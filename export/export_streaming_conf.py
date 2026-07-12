"""ICECS latency-ablation exports (conf branch) — single-frame streaming ONNX.

Exports latency-ablation variants of the deployed streaming SCNN, requested by
Tao Sun for the ICECS 2026 submission (deadline June 15):

  nobin               Remove the binarization (keep the proj 1x1 conv).
  noptwise            Remove the two kernel-size-1 convolutions: proj (after
                      LayerNorm) and mask (before Sigmoid). Binarization stays,
                      applied directly to the LayerNorm output. Needs N == B.
  plifro              Replace the ALIF readout membrane update (per-channel
                      alpha vector) with a PLIF readout update (scalar tau).
  combined            All three of the above (binarization disappears entirely
                      because proj is gone).
  decmatvec           Replace the decoder ConvTranspose with a mathematically
                      identical Dense (Gemm) layer. The streaming decoder input
                      has time-length 1, so ConvTranspose1d(N,1,K=80,stride=40)
                      is exactly an N->80 matrix-vector product. X-CUBE-AI
                      lowers ConvTranspose as zero-stuff + dense Conv2D
                      (+409,601 MACC at N=64 = 95.1% of the whole c-model, see
                      st_ai_output/dpsnn_streaming_n64_analyze_report.txt), so
                      this lossless rewrite removes ~95% of on-device compute.
  decmatvec_combined  decmatvec + combined.
  baseline            Unmodified wrapper (sanity reference; should reproduce
                      models/dpsnn_n{width}_streaming.onnx).

Latency does not depend on weight values, so any checkpoint (or random init)
gives valid timing. By default the trained checkpoint for the chosen width is
loaded so that the lossless decmatvec variant also preserves the published
quality (17.42 dB N=128 / 16.70 dB N=64). Use --random-init to skip loading.

Usage
-----
# One variant:
python export/export_streaming_conf.py --variant decmatvec --width 64

# Whole matrix for one width:
python export/export_streaming_conf.py --all --width 64

Outputs: export/conf_<variant>_n<width>.onnx (+ _xcubeai.onnx).
I/O tensor names are identical to the deployed model so the firmware glue and
the stedgeai validate harness need no changes.
"""
from __future__ import annotations

import argparse
import os
import sys

import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dpsnn.layers.spike_neuron import act_fun_adp
from dpsnn.models.dp_binary_net import StreamSpikeNet
from export.export_streaming import StreamingWrapper
from export.export_to_onnx import load_from_checkpoint, postprocess_for_xcubeai

DEFAULT_CKPTS = {
    128: "models/dpsnn_n128.ckpt",
    64: "models/dpsnn_n64.ckpt",
}

VARIANTS = {
    "baseline": {},
    "nobin": {"no_binarization": True},
    "noptwise": {"no_pointwise": True},
    "plifro": {"plif_readout": True},
    "combined": {"no_binarization": True, "no_pointwise": True,
                 "plif_readout": True},
    "decmatvec": {"decoder_matvec": True},
    "decmatvec_combined": {"no_binarization": True, "no_pointwise": True,
                           "plif_readout": True, "decoder_matvec": True},
    # Exp 12: checkpoints TRAINED with no_pointwise=True (proj/mask are None
    # in the model itself; ALIF readout kept). Use with
    # --ckpt models/dpsnn_n64_noptwise.ckpt.
    "exp12": {"no_binarization": True, "no_pointwise": True},
    "exp12_decmatvec": {"no_binarization": True, "no_pointwise": True,
                        "decoder_matvec": True},
}


class ConfStreamingWrapper(nn.Module):
    """StreamingWrapper with the ICECS ablations as constructor switches.

    Mirrors export_streaming.StreamingWrapper exactly when all switches are
    off. State tensor shapes and ONNX I/O names are unchanged in every
    configuration, so all variants are drop-in for the aiValidation firmware.
    """

    def __init__(self, model: StreamSpikeNet,
                 no_binarization: bool = False,
                 no_pointwise: bool = False,
                 plif_readout: bool = False,
                 decoder_matvec: bool = False) -> None:
        super().__init__()
        assert model.scnn_only and model.X == 1
        if no_pointwise:
            assert model.N == model.B, "removing proj/mask needs N == B"

        self.no_binarization = no_binarization
        self.no_pointwise = no_pointwise
        self.plif_readout = plif_readout
        self.decoder_matvec = decoder_matvec

        self.encoder_1d = model.encoder_1d
        self.encoder_act = model.encoder_act
        self.ln = model.ln
        self.proj = model.proj
        self.sconv1d = model.repeats[0][0]
        self.srnn_readout = model.srnn_readout
        self.mask = model.mask
        self.mask_act = model.mask_act
        self.decoder_1d = model.decoder_1d

        self.L = model.L
        self.stride = model.stride
        self.context_step = model.context_step

        plif = self.sconv1d.neuron
        self.plif_v_threshold = float(plif.v_threshold)
        self.plif_v_reset = float(plif.v_reset)
        self.plif_surrogate = plif.surrogate_function

        alif = self.srnn_readout.neuro
        self.alif_R_m = float(alif.R_m)

        # PLIF-readout tau parameter, initialised per PLIFNode convention
        # (init_tau=2.0 -> w=0 -> sigmoid(w)=0.5). Scalar instead of the ALIF
        # per-channel alpha vector.
        if plif_readout:
            self.register_buffer("plif_readout_w", torch.tensor(0.0))

        # Decoder as Dense: ConvTranspose1d(N,1,K,stride) on a length-1 input
        # is out[j] = bias + sum_c x[c] * W[c,0,j], i.e. Linear(N, L) with
        # weight = W.squeeze(1).T and the scalar conv bias broadcast over L.
        if decoder_matvec:
            dec = model.decoder_1d
            N, L = dec.weight.shape[0], dec.weight.shape[2]
            self.dec_linear = nn.Linear(N, L, bias=True)
            with torch.no_grad():
                self.dec_linear.weight.copy_(dec.weight.squeeze(1).t())
                self.dec_linear.bias.fill_(dec.bias.item())

    def forward(self, frame, context_win, v_plif, mem_readout, ola_tail):
        # ---- Encoder ----
        x = self.encoder_1d(frame)              # (1, N, 1)
        x = self.encoder_act(x)
        w = x

        x = self.ln(x)                          # (1, N, 1)

        # ---- Bottleneck: proj 1x1 conv and/or binarization ----
        if self.no_pointwise:
            if not self.no_binarization:
                # standalone learned-threshold binarization on the LN output
                x = act_fun_adp(x - self.proj.threshold)
        elif self.no_binarization:
            # plain conv, skip the spike threshold inside BinaryConv1D
            x = nn.Conv1d.forward(self.proj, x)  # (1, B, 1)
        else:
            x = self.proj(x)                     # conv + binarize (baseline)

        # ---- SCNN ----
        win_w = torch.cat([context_win, x], dim=2)
        y = self.sconv1d.dconv(win_w)            # (1, H, 1)

        alpha_plif = self.sconv1d.neuron.w.sigmoid()
        v_new = v_plif + (y - (v_plif - self.plif_v_reset)) * alpha_plif
        spike = self.plif_surrogate(v_new - self.plif_v_threshold)
        new_v_plif = v_new * (1.0 - spike.detach())

        x = spike
        new_context_win = win_w[:, :, 1:]

        # ---- Readout ----
        x_sq = x.squeeze(2)                      # (1, H)
        y_dense = self.srnn_readout.dense(x_sq)  # (1, B)
        if self.plif_readout:
            alpha_ro = self.plif_readout_w.sigmoid()
            new_mem_readout = mem_readout + (y_dense - mem_readout) * alpha_ro
        else:
            alpha_ro = self.srnn_readout.neuro.alpha.sigmoid()
            new_mem_readout = (mem_readout * alpha_ro
                               + (1.0 - alpha_ro) * self.alif_R_m * y_dense)
        x_readout = new_mem_readout.unsqueeze(2)  # (1, B, 1)

        # ---- Mask ----
        if self.no_pointwise:
            x_mask = self.mask_act(x_readout)     # sigmoid directly (B == N)
        else:
            x_mask = self.mask_act(self.mask(x_readout))
        x_out = w * x_mask                        # (1, N, 1)

        # ---- Decoder + OLA ----
        if self.decoder_matvec:
            decoded = self.dec_linear(x_out.squeeze(2))   # (1, L) via Gemm
        else:
            decoded = self.decoder_1d(x_out).squeeze(1)   # (1, L)

        enhanced = ola_tail + decoded[:, :self.stride]
        new_ola_tail = decoded[:, self.stride:]

        return enhanced, new_context_win, new_v_plif, new_mem_readout, new_ola_tail


def build_model(width: int, random_init: bool, ckpt_path: str | None) -> StreamSpikeNet:
    if random_init:
        torch.manual_seed(0)
        model = StreamSpikeNet(input_dim=16160, context_dim=160, sr=16000,
                               L=80, stride=40, N=width, B=width, H=width,
                               X=1, scnn_only=True)
    else:
        path = ckpt_path or DEFAULT_CKPTS[width]
        print(f"Loading checkpoint: {path}")
        model = load_from_checkpoint(path)
        assert model.N == width, f"checkpoint is N={model.N}, requested {width}"
    model.eval()
    return model


def compare_streams(model: StreamSpikeNet, wrapper: ConfStreamingWrapper,
                    n_frames: int = 400) -> float:
    """Max abs diff between the reference streaming wrapper and a conf variant
    over a streamed random utterance (state carried frame to frame in both).

    For checkpoints trained with no_pointwise=True the deployed
    StreamingWrapper cannot run (proj/mask are None), so the
    architecture-faithful ConfStreamingWrapper is the reference instead."""
    if getattr(model, "no_pointwise", False):
        baseline = ConfStreamingWrapper(model, no_binarization=True,
                                        no_pointwise=True)
    else:
        baseline = StreamingWrapper(model)
    baseline.eval()
    wrapper.eval()

    L, stride, B, ctx = model.L, model.stride, model.B, model.context_step
    torch.manual_seed(1)
    audio = torch.randn(1, (n_frames - 1) * stride + L)

    state_a = [torch.zeros(1, B, ctx), torch.zeros(1, B, 1),
               torch.zeros(1, B), torch.zeros(1, stride)]
    state_b = [t.clone() for t in state_a]

    max_diff = 0.0
    with torch.no_grad():
        for t in range(n_frames):
            frame = audio[:, t * stride: t * stride + L]
            out_a = baseline(frame, *state_a)
            out_b = wrapper(frame, *state_b)
            state_a, state_b = list(out_a[1:]), list(out_b[1:])
            max_diff = max(max_diff, float((out_a[0] - out_b[0]).abs().max()))
    return max_diff


def export_variant(variant: str, width: int, random_init: bool,
                   ckpt_path: str | None, out_dir: str = "export") -> None:
    flags = VARIANTS[variant]
    model = build_model(width, random_init, ckpt_path)
    wrapper = ConfStreamingWrapper(model, **flags)
    wrapper.eval()

    diff = compare_streams(model, wrapper)
    if variant in ("baseline", "decmatvec", "exp12", "exp12_decmatvec"):
        status = "PASS" if diff < 1e-4 else "FAIL"
        print(f"[{variant}] lossless check vs deployed wrapper: "
              f"max-abs-diff={diff:.2e} -> {status}")
        if status == "FAIL":
            raise RuntimeError(f"{variant} should be lossless but diff={diff}")
    else:
        print(f"[{variant}] diff vs deployed wrapper = {diff:.3f} "
              "(expected nonzero: the ablation changes the function)")

    L, stride, B, ctx = model.L, model.stride, model.B, model.context_step
    dummy = (torch.zeros(1, L), torch.zeros(1, B, ctx), torch.zeros(1, B, 1),
             torch.zeros(1, B), torch.zeros(1, stride))

    raw = os.path.join(out_dir, f"conf_{variant}_n{width}.onnx")
    xcube = raw.replace(".onnx", "_xcubeai.onnx")
    print(f"Exporting -> {raw}")
    torch.onnx.export(
        wrapper, dummy, raw,
        input_names=["frame", "context_win", "v_plif", "mem_readout", "ola_tail"],
        output_names=["enhanced", "new_context_win", "new_v_plif",
                      "new_mem_readout", "new_ola_tail"],
        opset_version=13,
    )
    postprocess_for_xcubeai(onnx_path=raw, output_path=xcube,
                            input_dim=L, output_dim=stride)
    print(f"[{variant} n{width}] done: {xcube}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--variant", choices=sorted(VARIANTS),
                        help="single variant to export")
    parser.add_argument("--all", action="store_true",
                        help="export every variant for the given width")
    parser.add_argument("--width", type=int, default=64, choices=(64, 128))
    parser.add_argument("--ckpt", default=None,
                        help=f"checkpoint override (default: {DEFAULT_CKPTS})")
    parser.add_argument("--random-init", action="store_true",
                        help="random parameters instead of a checkpoint "
                             "(latency-equivalent; supervisor-approved)")
    args = parser.parse_args()

    if not args.all and not args.variant:
        parser.error("pass --variant <name> or --all")

    names = sorted(VARIANTS) if args.all else [args.variant]
    for name in names:
        export_variant(name, args.width, args.random_init, args.ckpt)


if __name__ == "__main__":
    main()

"""ONNX export utilities for StreamSpikeNet.

Usage (from repo root):
    # DPSNN pretrained (N=256)
    python export/export_to_onnx.py \\
        --ckpt_path egs/voicebank/epoch=478-val_loss=81.5449-val_sisnr=-18.4556.ckpt \\
        --output_path export/dpsnn_pretrained.onnx

    # DPSNN scnn_only after training (hparams loaded from ckpt)
    python export/export_to_onnx.py \\
        --ckpt_path egs/voicebank/<scnn_ckpt>.ckpt \\
        --output_path export/dpsnn_scnn128.onnx
"""
from __future__ import annotations

import argparse
import os
import sys

import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dpsnn.models.dp_binary_net import StreamSpikeNet


class ExportWrapper(nn.Module):
    """Single-tensor interface around StreamSpikeNet or ConvTasNet.

    Both models expect a nested batch tuple; only noisy_x is used in the
    computation. This wrapper accepts only noisy_x, builds the required
    tuple internally, and returns enhanced audio as a 2-D tensor
    (batch, output_len).
    """

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, noisy_x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            noisy_x: (batch, seq_len) noisy audio including the context prefix.
        Returns:
            enhanced: (batch, output_len) denoised audio.
        """
        dummy_id = torch.zeros(1, device=noisy_x.device)
        dummy_len = torch.tensor(noisy_x.shape[-1], device=noisy_x.device)
        inputs = (dummy_id, noisy_x, dummy_len)
        dummy_targets = (dummy_id, torch.zeros_like(noisy_x), dummy_len)

        enhanced, _, _, _ = self.model((inputs, dummy_targets))
        return enhanced.reshape(noisy_x.shape[0], -1)


def load_from_checkpoint(ckpt_path: str) -> nn.Module:
    """Load a StreamSpikeNet from a PyTorch Lightning checkpoint."""
    ckpt = torch.load(ckpt_path, map_location="cpu")
    model = StreamSpikeNet(**ckpt["hyper_parameters"])
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model


def export_model(
    wrapper: ExportWrapper,
    dummy_input: torch.Tensor,
    output_path: str,
    opset_version: int = 13,
) -> None:
    """Export ExportWrapper to an ONNX file."""
    parent = os.path.dirname(output_path)
    if parent:
        os.makedirs(parent, exist_ok=True)

    # Note: dynamo=False is the default in PyTorch 2.1.0 (the param was added later).
    # Do NOT pass dynamo=True — it triggers onnxscript import errors on this version.
    torch.onnx.export(
        wrapper,
        dummy_input,
        output_path,
        input_names=["noisy_audio"],
        output_names=["enhanced_audio"],
        opset_version=opset_version,
        dynamic_axes={
            "noisy_audio":    {0: "batch_size"},
            "enhanced_audio": {0: "batch_size"},
        },
    )
    size_kb = os.path.getsize(output_path) / 1024
    print(f"Exported -> {output_path}  ({size_kb:.1f} KB)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Export StreamSpikeNet to ONNX")
    parser.add_argument("--ckpt_path", required=True)
    parser.add_argument("--output_path", default="export/model.onnx")
    parser.add_argument("--opset", type=int, default=13)
    args = parser.parse_args()

    print(f"Loading checkpoint: {args.ckpt_path}")
    model = load_from_checkpoint(args.ckpt_path)

    input_dim = model.hparams["input_dim"]
    print(f"Model: input_dim={input_dim}, "
          f"context_dim={model.hparams['context_dim']}, "
          f"N={model.hparams['N']}, B={model.hparams['B']}, H={model.hparams['H']}, "
          f"scnn_only={model.hparams.get('scnn_only', False)}")

    wrapper = ExportWrapper(model)
    dummy_input = torch.randn(1, input_dim)
    print(f"Dummy input shape: {dummy_input.shape}")

    print("Attempting ONNX export ...")
    try:
        export_model(wrapper, dummy_input, args.output_path, args.opset)
        print("Export SUCCEEDED.")
    except Exception as exc:
        print(f"\nExport FAILED: {type(exc).__name__}: {exc}")
        print("\nDocument the full traceback in results/experiment_log.md.")
        sys.exit(1)


if __name__ == "__main__":
    main()

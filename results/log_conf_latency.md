# Exp 11 — ICECS Latency Experiments (conf branch)

**Date started:** 2026-06-10
**Branch:** `conf` (branched from `clean` after the thesis freeze)
**Goal:** Cut per-frame inference time below the 2.5 ms real-time budget for the
ICECS 2026 submission (deadline extended to June 15).

## 1. Context — supervisor request

Email from Tao Sun (2026-06-10):

> Since the ICECS submission is extended to June 15th, can you try some more
> experiments to cut the inference time and make the model real-time? The
> possible options are
>
> 1. Remove the binarization.
> 2. Remove the two convolutions with kernel size 1 (the one after LayerNorm
>    and the one before Sigmoid.)
> 3. Change ALIF Readout to PLIF Readout.
>
> I think you do not need to train the model again to get the inference time
> and just use the random parameters.

Baseline (published in the thesis / ICECS draft): N=128 → 6.145 ms/frame
(RTF 2.46×), N=64 → 2.715 ms/frame (RTF 1.086×), both FP32 streaming on the
STM32U585 at 160 MHz.

## 2. Root cause first: where does the time actually go?

Before implementing the ablations we checked the existing per-layer compiler
report (`st_ai_output/dpsnn_streaming_n64_analyze_report.txt`, committed on
`clean`):

- Line 180: `model/c-model: macc=21,110/430,852 +409,742(+1941.0%)` — the ONNX
  graph needs only **21,110 MACC** per frame, but the generated C model
  executes **430,852 MACC**. The 20× inflation is added by X-CUBE-AI itself.
- Line 169/545: the inflation is one layer. The decoder
  `ConvTranspose` (`_decoder_1d_ConvTranspose_..._inflated_2`) is lowered as
  `Resize_/Conv2D` — zero-stuff upsampling followed by a **dense** conv over
  the stuffed buffer: 5,121 ONNX MACC → +409,601 C MACC = **95.1 % of the
  entire c-model**.
- The `-O time` build (`dpsnn_streaming_n64_otime_analyze_report.txt`) reports
  the identical 430,852 MACC → compiler flags cannot fix this; only a graph
  change can.

Meanwhile the layers targeted by the three requested ablations (proj conv,
mask conv, binarization chain, readout update) together account for **~2–4 %**
of c-model MACC.

**Key insight:** in the streaming formulation the decoder input has
time-length 1, so `ConvTranspose1d(N, 1, K=80, stride=40)` degenerates to a
plain matrix–vector product: `out[j] = b + Σ_c x[c]·W[c,0,j]`, i.e. a Dense
layer `(N → 80)`. Re-expressing it as `nn.Linear` / ONNX `Gemm` is
**mathematically identical** (same trained weights, transposed view) and
costs 5,200 MACC instead of 414,722. This became the fourth experiment
(`decmatvec`), and unlike the ablations it requires no retraining and incurs
**zero quality change**.

This mirrors the thesis's central finding one level deeper: first the memory
bottleneck was graph topology (OLA chain), not weights; now the latency
bottleneck is the compiler's lowering of one operator, not the SNN layers.

## 3. Variants

Implemented in `export/export_streaming_conf.py` as switches on a copy of the
deployed `StreamingWrapper` (I/O tensor names unchanged → drop-in for the
firmware):

| Variant | Meaning |
|---|---|
| `baseline` | Unmodified deployed wrapper (control). |
| `nobin` | Binarization removed: proj conv runs without the `act_fun_adp(x − threshold)` spike function (drops the Sub→Sign→ReLU chain). |
| `noptwise` | Both 1×1 convs removed: LN output is binarized directly (scalar learned threshold) and feeds the SCNN; Sigmoid applied directly to the readout membrane. Shape-safe because N=B=H. |
| `plifro` | ALIF readout update `mem·σ(α_vec) + (1−σ(α_vec))·y` replaced by PLIF update `mem + (y − mem)·σ(w_scalar)`. |
| `combined` | All three (binarization disappears entirely because proj is gone). |
| `decmatvec` | Decoder ConvTranspose → `nn.Linear(N, 80)` (Gemm), weights = `W.squeeze(1).T`, bias = scalar conv bias broadcast. Lossless. |
| `decmatvec_combined` | decmatvec + combined. |

All variants were exported from the **trained checkpoints**
(`models/dpsnn_n128.ckpt`, `models/dpsnn_n64.ckpt`); latency is independent of
weight values (supervisor approved random params — trained weights are a
superset of that: same timing, and the lossless variant keeps real quality).

**Equivalence checks** (400-frame random stream, state carried frame to
frame, vs the deployed wrapper):

- `baseline`: max abs diff **0.0** (both widths) — wrapper copy is exact.
- `decmatvec`: max abs diff **1.14e-05** (N=64), **9.54e-06** (N=128) — FP32
  accumulation-order noise only. PASS (lossless as designed).
- Ablation variants: diffs of 9.9–67.7 as expected — they change the function;
  quality numbers would require retraining and are out of scope per the
  supervisor's email.

## 4. Static analysis (stedgeai analyze v10.2.0, --target stm32u5 -O balanced)

Reports: `st_ai_output/conf_<variant>_<width>_analyze_report.txt`.

### N = 128

| Variant | c-model MACC | Δ vs baseline | Weights (B) | Activations (B) |
|---|---|---|---|---|
| baseline | 886,180 | — | 285,216 | 47,072 |
| nobin | 885,796 | −0.04 % | 285,212 | 47,072 |
| noptwise | 853,412 | −3.7 % | 154,144 | 47,072 |
| plifro | 886,308 | **+0.01 %** | 285,216 | 47,072 |
| combined | 853,156 | −3.7 % | 154,140 | 47,072 |
| **decmatvec** | **77,259** | **−91.3 %** | 285,532 | **8,672** |
| decmatvec_combined | 44,235 | −95.0 % | 154,456 | 8,672 |

### N = 64

| Variant | c-model MACC | Δ vs baseline | Weights (B) | Activations (B) |
|---|---|---|---|---|
| baseline | 430,852 | — | 93,472 | 23,264 |
| nobin | 430,660 | −0.04 % | 93,468 | 23,776 |
| noptwise | 422,660 | −1.9 % | 60,704 | 23,776 |
| plifro | 430,916 | **+0.01 %** | 93,472 | 23,264 |
| combined | 422,532 | −1.9 % | 60,700 | 23,264 |
| **decmatvec** | **26,411** | **−93.9 %** | 93,788 | **4,320** |
| decmatvec_combined | 18,091 | −95.8 % | 61,016 | 4,320 |

Observations:

1. **The three requested ablations barely move compute.** Removing the two
   1×1 convs saves 1.9–3.7 % MACC (though it does halve the weight storage,
   131 KB → useful for Flash, not latency). Removing binarization saves
   0.04 %. The PLIF readout is *not* cheaper at the graph level (+64/+128
   MACC — the scalar σ(w) broadcast costs the same as the vector version).
2. **The decoder rewrite removes 91–96 % of all compute.** In
   `conf_decmatvec_n64` the decoder appears as `Dense` with 5,200 MACC
   (report line 65/172) instead of 414,722 — a 79.7× reduction on that layer.
3. Activations collapse 47 KB → 8.7 KB (N=128) and 23 KB → 4.3 KB (N=64)
   because the zero-stuffed ConvTranspose scratch buffer disappears.

## 5. Predicted on-device latency (linear MACC scaling) — FALSIFIED by §6

Scaling from the measured baselines (N=128: 6.934 ns/MACC; N=64: 6.301
ns/MACC at 160 MHz), written down *before* the board runs:

| Variant | N=128 pred. ms | N=64 pred. ms |
|---|---|---|
| nobin | 6.14 | 2.71 |
| noptwise | 5.92 | 2.66 |
| plifro | 6.15 | 2.72 |
| combined | 5.92 | 2.66 |
| decmatvec | **0.54** (RTF 0.21) | **0.17** (RTF 0.07) |
| decmatvec_combined | 0.31 | 0.11 |

**These predictions are wrong** (kept here deliberately as a record). The
measurements in §6 show that c-model MACC is not a usable latency proxy for
this graph: the prediction for `decmatvec` was off by 15×, and for
`noptwise` in the *opposite* direction. See §6.2 for the cycle-level
analysis.

## 6. On-device measurements (DWT cycle counter, p232_009, 1,662 frames)

Measurement loop fully automated: `tools/run_conf_campaign.ps1` runs
stedgeai generate → STM32CubeIDE headless build (stm32cubeidec.exe) → flash
via STM32_Programmer_CLI → captures the firmware's `TIMING:` UART line
(`tools/read_timing.py`). Each variant is generated under the fixed network
name `dpsnn_streaming_n64` so the firmware glue never changes. Campaign run
2026-06-11 00:48–00:56; raw lines in `results/conf_timing_results.txt`.

### 6.1 Results

| Variant | Width | ms/frame | RTF | Δ cycles vs baseline | Real-time? |
|---|---|---|---|---|---|
| baseline (control) | 64 | 2.715 | 1.086 | — | ✗ |
| nobin | 64 | 2.683 | 1.073 | −5,120 (−1.2 %) | ✗ |
| noptwise | 64 | **2.282** | **0.913** | −69,280 (−16.0 %) | **✓** |
| plifro | 64 | 2.723 | 1.089 | +1,280 (+0.3 %) | ✗ |
| combined | 64 | **2.253** | **0.901** | −73,920 (−17.0 %) | **✓** |
| decmatvec | 64 | 2.564 | 1.025 | −24,160 (−5.6 %) | ✗ (2.5 % over) |
| decmatvec_combined | 64 | **2.101** | **0.840** | −98,240 (−22.6 %) | **✓ (best)** |
| baseline (published Exp 7) | 128 | 6.145 | 2.458 | — | ✗ |
| decmatvec | 128 | 5.991 | 2.396 | −24,640 (−2.5 %) | ✗ |

- The control reproduced the published Exp 10 number **exactly** (2.715 ms,
  RTF 1.086) → the automated pipeline and the firmware buffer-size change are
  validated.
- **Real-time is achieved at N=64**: removing the two 1×1 convs alone gives
  RTF 0.913; the supervisor's three changes combined give RTF 0.901; adding
  the lossless decoder rewrite gives **2.101 ms/frame, RTF 0.840** — 16 %
  under budget.
- N=128 remains far from real-time (best lossless config 5.991 ms,
  RTF 2.40).
- Effects compose almost perfectly additively: singles sum to −0.457 ms vs
  −0.462 ms measured for `combined`; `combined`+`decmatvec` sum to −0.613 ms
  vs −0.614 ms measured for `decmatvec_combined`.

### 6.2 Why the MACC prediction failed — cycle-level analysis

Cycle budgets (ms × 160,000 cycles/ms):

| Quantity | N=64 | N=128 |
|---|---|---|
| baseline cycles/frame | 434,400 | 983,200 |
| decmatvec cycles/frame | 410,240 | 958,560 |
| → real cost of the ConvTranspose decoder | **~24.2 K cycles** | **~24.6 K cycles** |
| nominal c_macc of that decoder | 414,722 | 829,522 |

1. **The compiler's c_macc wildly overstates the decoder's real cost.** The
   zero-stuffed ConvTranspose that the report bills at 95 % of all MACC
   actually executes in ~24 K cycles (≈0.06 cycles per *nominal* MACC) and is
   nearly width-independent — the runtime kernel clearly does not pay for the
   stuffed zero positions at the planner's nominal rate. The thesis-reported
   "1.12 cycles/MACC" was an artefact of this inflated denominator.
2. **The true bottleneck is small-tensor / non-MACC work.** After the decoder
   fix, N=64 still needs 410 K cycles for only 26 K MACC (≈15.5
   cycles/MACC): LayerNorm's reduce/normalise chain, ~30 elementwise spike
   ops on (1,64,1) tensors, per-node runtime dispatch, and state movement
   dominate.
3. **That is why the supervisor's `noptwise` works so well:** the two 1×1
   convs are only 8 K MACC but cost ~34.6 K cycles each (≈8.4 cycles/MACC,
   overhead-dominated small-conv kernels) → −0.433 ms.
4. `nobin` saves the Sub→Sign→ReLU binarization chain: −5 K cycles. `plifro`
   is cycle-neutral (+1.3 K, noise) — ALIF and PLIF readout updates cost the
   same per frame, confirming the graph-level expectation.

**Methodological takeaway for the paper:** static MACC reports from the
embedded compiler do not predict on-device latency for small streaming
graphs; per-variant *measurement* is mandatory. (This is the latency-side
analogue of the thesis's memory finding: the report metrics mislead, the
bottleneck is structural.)

### 6.3 On-device quality check (lossless variants only)

`tools/mcu_audio_receiver.py` against per-variant Python ONNX references
(`tools/wav_to_c_array.py` regenerated refs from the conf ONNX graphs —
both reproduce the published reference SI-SNRs exactly):

| Build | MCU SI-SNR (p232_009) | Python ONNX ref | Delta |
|---|---|---|---|
| decmatvec N=64 | 15.98 dB | 15.98 dB | **0.00 dB PASS** |
| decmatvec N=128 | 15.66 dB | 15.66 dB | **0.00 dB PASS** |

The decoder rewrite is confirmed lossless end-to-end on hardware. The
ablation variants (`nobin`, `noptwise`, `plifro`, `combined`) change the
network function; attaching quality numbers to them requires retraining,
which is out of scope per the supervisor's email ("just use the random
parameters").

### 6.4 Conclusions and recommended next steps

1. **Answer to the supervisor's question:** yes — the model can be made
   real-time at N=64. Removing the two 1×1 convolutions is the single most
   effective of the three suggestions (RTF 1.086 → 0.913); all three
   combined give RTF 0.901. Removing binarization helps marginally
   (−1.2 %); ALIF→PLIF readout does not change latency at all.
2. **The decoder ConvTranspose→Gemm rewrite should be adopted
   unconditionally**: it is mathematically lossless (0.00 dB on-device),
   needs no retraining, saves a further 0.15 ms, and cuts peak activation
   SRAM 23.3 → 4.3 KB (N=64) / 47 → 8.7 KB (N=128). Best overall config:
   `decmatvec_combined` at **2.101 ms/frame, RTF 0.840**.
3. **Quality cost of the ablations is the open item.** If time permits
   before camera-ready: retrain `combined` (or just `noptwise`) at N=64 with
   the Exp 10 recipe and evaluate on the 824-utterance test set.
4. Remaining latency at N=64 (336 K cycles for 14 K MACC in
   `decmatvec_combined`) is dominated by LayerNorm + elementwise chains —
   per-node on-device profiling (aiValidation observer) is the next
   optimisation lever if more speed is ever needed.

## 7. Firmware change (outside repo)

`Stm_deployment/X-CUBE-AI/App/app_x-cube-ai.c`: the ping-pong state buffers
(`ctx_a/b`, `vp_a/b`, `mem_a/b`) were hardcoded for N=64 (256/64/64 floats);
they are now sized from the generated header macros
(`AI_DPSNN_STREAMING_N64_IN_2..4_SIZE`) so any regenerated variant — including
N=128 — compiles without edits. Behaviour for the deployed N=64 model is
bit-identical (macros equal the old constants).

## 8. Reproduction

```powershell
# Export all variants for one width (trained ckpt; --random-init also works)
python export/export_streaming_conf.py --all --width 64
python export/export_streaming_conf.py --all --width 128

# Static analysis (per variant)
stedgeai analyze --model export/conf_<v>_<w>_xcubeai.onnx --target stm32u5 `
    --optimization balanced --compression none --name conf_<v>_<w> `
    --workspace st_ai_ws --output st_ai_output

# Full on-device campaign (board connected)
powershell -File tools\run_conf_campaign.ps1 -Port COM3
```

### 6.5 Validation of the campaign (2026-06-11)

Five independent checks confirm the results are valid:

1. **Control**: the re-measured baseline N=64 build reproduced the published
   Exp 10 numbers exactly (2.715 ms/frame, RTF 1.086) through the full
   automated generate->build->flash->measure pipeline.
2. **Structural**: greps over the per-layer analyze reports confirm each
   compiled graph contains exactly its claimed change. `nobin` keeps the
   proj conv (renamed `_Conv_output_0` by the tracer since the module
   __call__ is bypassed) and drops the binarization Sign chain (4->2 Sign
   layers); `noptwise` has no `_proj_Conv`/`_mask_Conv` and keeps 4 Sign
   layers; `combined` has neither convs nor binarization (2 Sign);
   `decmatvec` replaces ConvTranspose with a `Dense` (Gemm) layer.
3. **Losslessness (decmatvec)**: PyTorch stream diff vs deployed wrapper
   <= 1.1e-05; per-variant ONNX references reproduce the published p232_009
   SI-SNRs exactly (15.66 / 15.98 dB); on-device MCU output matches the
   ONNX reference to 0.00 dB at both widths.
4. **Repeatability**: independent re-runs (fresh generate, build, flash) of
   noptwise, plifro and decmatvec_combined reproduced 2.282 / 2.723 /
   2.101 ms to the displayed precision - DWT timing over 1,662 frames is
   deterministic. The +0.008 ms of plifro vs baseline is reproducible but
   still only +0.3% (latency-neutral for practical purposes).
5. **Internal consistency**: single-change effects compose additively into
   the combined configs (singles sum -0.457 ms vs -0.462 measured;
   combined+decmatvec sum -0.613 ms vs -0.614 measured).

Latency is weight-independent for these graphs (dense FP32 kernels, no
data-dependent branches), so measuring with trained weights also covers the
supervisor's random-parameter premise.

## 7. Exp 12 — retrained no-pointwise model (Tao-approved config, 2026-06-12)

Tao's reply to the latency results: "You can remove the first two
(binarization and two convs) together and train it." ALIF readout stays.

**Training:** recipe identical to Exp 10 plus `--no_pointwise`
(new flag in `dpsnn/models/dp_binary_net.py` / `vctk_trainer.py`): N=B=H=64,
scnn_only, 100 epochs, lr 1e-2, batch 64, bf16-mixed, RTX 4060, 911 min
(15.2 h). 15,042 trainable params (the two 1x1 convs + threshold removed:
-8,321 vs the Exp 10 architecture). Best ckpt epoch 99
(`models/dpsnn_n64_noptwise.ckpt`); val SI-SNR plateaued from ~epoch 82
(16.57-16.59 dB band), so the run is converged, not truncated.

**Quality (batch test over 824 utterances, trainer test pass):**

| Metric | Exp 10 baseline N=64 | Exp 12 no-pointwise | delta |
| --- | --- | --- | --- |
| SI-SNR | 16.70 dB | 16.23 dB | -0.47 dB |
| PESQ (wb) | 2.022 | 1.959 | -0.063 |
| STOI | 0.922 | 0.917 | -0.005 |

Honest note: PESQ 1.959 is marginally below the noisy input (1.971), while
SI-SNR improves it by +7.79 dB. The full *streaming* eval
(`results/eval_exp12_noptwise_streaming.txt`, the deployment-faithful
number) confirms: **SI-SNR 16.24 dB**, PESQ 1.973, STOI 0.917 — matches the
batch test within 0.01 dB, as in Exp 7/10.

**On-device (same firmware loop, p232_009, 1,662 frames):**

| Build | ms/frame | RTF | MCU vs ONNX | MCU SI-SNR |
| --- | --- | --- | --- | --- |
| exp12 (ConvTranspose decoder) | 2.247 | 0.899 | - | - |
| exp12 + decoder Gemm | **2.097** | **0.839** | **0.00 dB** | 15.89 dB |

- Latencies confirm weight-independence: the random-weight equivalents
  measured 2.253 / 2.101 ms (combined incl. the latency-neutral PLIF swap).
- p232_009 quality vs the other deployed models: 15.89 dB (Exp 12) vs
  15.98 dB (Exp 10 N=64) vs 15.66 dB (Exp 7 N=128).
- Footprint (analyze): weights 61,016 B, activations 4,320 B with the Gemm
  decoder -> the deployable real-time config uses <0.6 % of SRAM.

**Conclusion:** the supervisor-approved config is real-time on the
STM32U585 with trained weights: **2.097 ms/frame (RTF 0.839) at 16.23 dB
test SI-SNR**, costing 0.47 dB vs the non-real-time baseline. Exported
ONNX: `export/conf_exp12_n64_xcubeai.onnx`,
`export/conf_exp12_decmatvec_n64_xcubeai.onnx` (lossless checks: 0.0 and
1.14e-05).

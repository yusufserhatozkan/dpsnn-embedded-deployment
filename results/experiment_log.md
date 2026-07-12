# Experiment Log

Full records are split across three files. Open the relevant one for details.

| File | Contents |
| --- | --- |
| [log_baseline.md](log_baseline.md) | Pretrained N=256 inference, ONNX export, numerical validation, pipeline decision |
| [log_training.md](log_training.md) | SCNN-only N=128 model design, both training runs, final test-set evaluation |
| [log_quantization.md](log_quantization.md) | All INT8 experiments, root cause analysis, footprint numbers |

---

## X-CUBE-AI Compatibility Fix (2026-05-13)

### Error

`INTERNAL ERROR: Trying to remove _model_Equal_output_0 which has multiple inputs`

X-CUBE-AI fails on any ONNX that contains dead `Equal` nodes — even after If-node inlining.

### Root Cause

The `torch.onnx` tracer emits 798 `If` nodes (one per unrolled time step) where both branches are identical `Squeeze` ops. Each `If` node's condition is produced by a `Shape → Gather → Equal(gathered_dim, const_0)` chain. After `inline_if_nodes.py` inlines the If bodies and removes the `If` nodes, those 3-node chains become dead code — no consumer reads `Equal`'s output. X-CUBE-AI's internal graph simplifier tries to remove these dead `Equal` nodes but crashes because its code assumes Equal nodes only have one input tensor.

### Fix

Added `remove_dead_nodes()` to `tools/inline_if_nodes.py`. It performs a backward reachability pass from graph outputs to identify all live nodes, then removes everything unreachable. On the FP32 baked model it removes **3990 dead nodes**:

| Op type | Count removed |
| --- | --- |
| Equal | 798 |
| Shape | 798 |
| Gather | 798 |
| Constant | 1596 |

The fix is also integrated into `inline_if_nodes.py`'s `main()`, so it runs automatically when you use that script.

### Integrated Pipeline

`export/export_to_onnx.py` now runs the full pipeline automatically after the raw ONNX export and produces `<stem>_xcubeai.onnx`:

```text
torch.onnx.export (no dynamic_axes)
  → inline If nodes
  → remove dead nodes        ← NEW
  → fix concrete shapes
  → ONNX shape inference
  → bake shapes via ORT (~5 min)
```

### Second Error (after dead-node removal): `TOOL ERROR: list index out of range`

Every `Pad` node has an empty string `""` as its optional third input (`constant_value`). ONNX uses `""` as a placeholder for absent optional inputs. X-CUBE-AI looks up `""` as a tensor name, gets an empty result, then accesses `result[0]` → crash.

**Fix:** Added `strip_trailing_empty_inputs()` to `tools/inline_if_nodes.py`. It deletes trailing `""` entries from every node's input list (safe because the ONNX spec defaults absent optional inputs to zero/identity values). Removed 399 empty inputs on the 399 Pad nodes.

The two fixes together:

| Fix | Nodes affected |
| --- | --- |
| `remove_dead_nodes` | 3990 nodes (798 Equal + 798 Shape + 798 Gather + 1596 Constant) |
| `strip_trailing_empty_inputs` | 399 Pad nodes (3→2 inputs each) |

### Third fix: ORT constant-folding (clean3)

After removing dead nodes and empty inputs, the graph still contained `ConstantOfShape` (401), `Shape` (6), and `Gather` (6) nodes from dynamic shape computations introduced by the original `dynamic_axes` export. X-CUBE-AI cannot evaluate these at compile time and may crash.

**Fix:** Run ORT graph optimisation at `ORT_ENABLE_BASIC` level (constant folding + dead-code elimination, no op fusion). This folds all `Shape → Gather → ConstantOfShape` chains into plain constants and removes the ~11 K `Constant` initialiser nodes that become redundant. `ORT_ENABLE_ALL` was tried first but rejected — it introduces `FusedConv` (ORT-internal op that X-CUBE-AI does not support).

Result on `dpsnn_fp32_baked_clean2.onnx`:

| Op type | Before | After |
| --- | --- | --- |
| ConstantOfShape | 401 | 0 |
| Shape | 6 | 0 |
| Gather | 6 | 0 |
| Constant | ~11 216 | 0 |
| Cast | 1997 | 802 |
| Concat | 1200 | 798 |
| Reshape | 1198 | 400 |
| Sigmoid | 401 | 399 |

File: `C:/ai/dpsnn_fp32_clean3.onnx` (4929 KB) — use this for the next X-CUBE-AI Analyse attempt.

### Summary — all rounds

| Round | Fix | Trigger |
| --- | --- | --- |
| clean | `remove_dead_nodes` — removes 3990 orphaned Equal/Shape/Gather/Constant nodes | `INTERNAL ERROR: Trying to remove _model_Equal_output_0` |
| clean2 | `strip_empty_inputs` — removes 399 empty-string Pad inputs | `TOOL ERROR: list index out of range` |
| clean3 | ORT BASIC constant-fold — eliminates ConstantOfShape, Shape, Gather, redundant Constants | proactive (anticipating further crashes) |
| final | shape inference POST-ORT — resolves 2420 missing intermediate tensor shapes | (intermediate `Unknown dimensions: H` while integrating ORT step) |
| stripped | strip foreign opset imports — keeps only standard ONNX domain | `INTERNAL ERROR: Unknown dimensions: H` on `dpsnn_fp32_final.onnx` (did not fix) |
| simplified | onnxsim — removes 400 no-op Reshape and 399 redundant Concat ops | `INTERNAL ERROR: Unknown dimensions: H` on `dpsnn_fp32_final_stripped.onnx` (did not fix) |
| 4d | inflate every `Pad` and `ConvTranspose1d` to rank-4 NCHW (Unsqueeze → 2D op → Squeeze) | `INTERNAL ERROR: Unknown dimensions: H` on `dpsnn_fp32_simplified.onnx` (cleared H; surfaced BOOL error) |
| nobool | replace `Greater/GreaterOrEqual+Cast` with `Sub→Sign→Relu` and `Where(bool,A,B)` with arithmetic | `INTERNAL ERROR: Unsupported operands type: BOOL` on `dpsnn_fp32_4d.onnx` |

### Fifth fix: strip foreign opset imports (2026-05-13)

After all previous fixes `dpsnn_fp32_final.onnx` still produced
`INTERNAL ERROR: Unknown dimensions: H`. Inspection showed:

- 0 symbolic dims, 0 tensors missing shape annotations
- All 18 433 nodes use the standard ONNX domain (`''`)
- BUT `opset_import` declared 8 domains: `''`, `ai.onnx.ml`, `ai.onnx.training`,
  `ai.onnx.preview.training`, `com.microsoft`, `com.microsoft.experimental`,
  `com.microsoft.nchwc`, `org.pytorch.aten`

The 7 non-standard domains were injected by ORT's `ORT_ENABLE_BASIC` pass
(ORT advertises all domains it knows, even when the saved model uses none of
them). X-CUBE-AI's parser walks the opset list, fails to load the unknown
domains, and surfaces the failure as the misleading "Unknown dimensions: H".

**Fix:** new Step 6 in `postprocess_for_xcubeai()` drops every opset entry
whose domain is not actually used by a node — in practice keeping only
`('' , 13)`. Output: `dpsnn_fp32_final_stripped.onnx` (5036 KB, identical
graph, single opset import).

**Result:** X-CUBE-AI still emitted `INTERNAL ERROR: Unknown dimensions: H`
on the stripped model. The opset list was a real defect but not the actual
trigger of the H error. See sixth fix.

### Sixth fix: onnxsim — graph simplification (2026-05-13)

Located the error template by extracting strings from
`irs/objects/ai_shape.cp39-win_amd64.pyd`:

```text
b'Unknown dimensions: '
b'AIShape.get_space_dimensions'
b'AIShape.get_non_batch_dimensions'
b'CHANNEL', b'HEIGHT', b'channel_first', b'channel_last'
```

So "H" is X-CUBE-AI's internal layout slot for HEIGHT. The parser
constructs an `AIShape` per tensor and tries to assign each ONNX dim to a
named slot (BATCH/CHANNEL/HEIGHT/WIDTH). It is rejecting some tensor
whose dims it cannot map.

Inspection of `dpsnn_fp32_final_stripped.onnx` found 400 no-op `Reshape`
nodes: 399 with shape `(-1, 128)` on a `(1, 128)` input and 1 with shape
`(1, -1)` on a `(1, 16000)` input. They reshape to the same shape they
already have, but X-CUBE-AI's parser does not simplify them away and the
`-1` apparently confuses the layout assignment.

**Fix:** ran the `onnxsim` bundled inside X-CUBE-AI itself
(`Lib/site-packages/onnxsim` v0.4.36) on the model. Using the same
simplifier the ST toolchain itself uses removes any doubt about
compatibility.

Result on `dpsnn_fp32_final_stripped.onnx` → `dpsnn_fp32_simplified.onnx`:

| Op | Before | After | Δ |
| --- | --- | --- | --- |
| Reshape | 400 | 0 | −400 |
| Concat | 798 | 399 | −399 |
| Add | 2002 | 2000 | −2 |
| Sub | 1604 | 1603 | −1 |
| Total | 18 433 | 17 631 | −802 |

File: `C:/ai/dpsnn_fp32_simplified.onnx` (4068 KB, down from 5036 KB).

The pipeline now applies onnxsim as the final step (`Step 7` in
`export/export_to_onnx.py`'s `postprocess_for_xcubeai()`).

**Result:** still `Unknown dimensions: H`. The graph was textually clean
but X-CUBE-AI's AIShape parser kept rejecting *something*. Time to bisect.

### Seventh fix: bisect with minimal models — Pad/ConvTranspose need NCHW (2026-05-13)

Built one-node test ONNX files for every distinct op pattern in the
model and ran each through `stedgeai analyze`:

| Test | Op | Input shape | Result |
| --- | --- | --- | --- |
| test1_conv | Conv1d | (1, 1, 80) → (1, 128, 1) | (n/a — batch EOF) |
| test2_pointwise | Conv1d k=1 | (1, 128, 1) → (1, 128, 1) | ✓ PASS |
| test6_depthwise | Conv1d group=128 | (1, 128, 5) → (1, 128, 1) | ✓ PASS |
| test4_slice | Slice rank-2 | (1, 16160) → (1, 80) | ✓ PASS |
| testG_add2d | Add rank-2 | (1, 16000) + (1, 16000) | ✓ PASS |
| **test3_pad** | **Pad rank-2** | **(1, 80) → (1, 16000)** | **✗ `Unknown dimensions: H`** |
| **testA_pad3d** | **Pad rank-3** | **(1, 1, 80) → (1, 1, 16000)** | **✗ `Unknown dimensions: W`** |
| testE_pad4d | Pad rank-4 NCHW | (1, 1, 1, 80) → (1, 1, 1, 16000) | ✓ PASS |
| **test5_convtrans** | **ConvTranspose1d** | **(1, 128, 1) → (1, 1, 80)** | **✗ `TOOL ERROR: tuple index`** |
| testF_convtrans2d | ConvTranspose2d NCHW | (1, 128, 1, 1) → (1, 1, 1, 80) | ✓ PASS |

**Root cause:** X-CUBE-AI v2.2.0 (ST Edge AI Core 2.2.0-20266) requires
`Pad` and `ConvTranspose` to be 4D NCHW. Rank-2 Pad errors as "Unknown H",
rank-3 Pad as "Unknown W", and any rank-3 ConvTranspose errors as
"tuple index out of range" in its parser. Conv1d (3D) is fine.

Our model uses 399 `Pad`s for overlap-add reconstruction (each on a
rank-2 (1, 80)) and 399 `ConvTranspose1d`s for the decoder.

**Fix:** new post-processing step `inflate_to_4d()` that walks the graph
and rewrites every offending `Pad`/`ConvTranspose` as:

```text
input (rank-r)
  → Unsqueeze(axes=[…])  → 4D input
  → 2D Pad/ConvTranspose (attrs prepended with 1s)
  → Squeeze(axes=[…])    → original rank
  → downstream
```

For rank-2 Pad: `axes=[1,2]` (insert H,C of size 1, treating last dim as W).
For rank-3 ConvTranspose: `axes=[2]` (insert H of size 1 between C and W).
The `pads` initializer and ConvTranspose weight tensor are also inflated.

After the inflation, onnxsim is rerun to fold the new Squeeze/Unsqueeze
pairs into adjacent ops where possible.

Result on `dpsnn_fp32_simplified.onnx`:

| Metric | Before | After |
| --- | --- | --- |
| Total nodes | 17 631 | 18 828 |
| File size | 4068 KB | 4343 KB |
| Pad rank-2 | 399 | 0 |
| Pad rank-4 | 0 | 399 |
| ConvTranspose rank-3 | 399 | 0 |
| ConvTranspose rank-4 | 0 | 399 |

File: `C:/ai/dpsnn_fp32_4d.onnx`.

**Result:** X-CUBE-AI accepted the Pad and ConvTranspose nodes — the
"Unknown dimensions: H" error is gone. New error surfaced further into
parsing: `INTERNAL ERROR: Unsupported operands type: BOOL`.

### Eighth fix: eliminate BOOL tensors (2026-05-13)

Individual bisect tests (testI–testQ) showed that
`Greater`/`GreaterOrEqual` + `Cast(bool→float)`, plain `Where(bool, A, B)`,
and even the Cast+Where fan-out pattern all parse fine in isolation. The
failure only manifests in the full graph, so we eliminate BOOL tensors
entirely.

**Rewrite:**

| Original pattern | Replacement (float only) |
| --- | --- |
| `Greater(x, c) -> bool` | `Sub(x, c) -> Sign -> Relu` |
| `GreaterOrEqual(x, c) -> bool` | same as above (drift only at exact equality) |
| `Cast(bool -> float)` | `Identity(replacement_float)` — the bool tensor is already gone |
| `Where(bool, A, B)` | `bool_float * A + (1 - bool_float) * B` |

After rewrite + onnxsim:

| Metric | Before (`dpsnn_fp32_4d.onnx`) | After (`dpsnn_fp32_final_v2.onnx`) |
| --- | --- | --- |
| Total nodes | 18 828 | 20 022 |
| File size | 4343 KB | 4611 KB |
| BOOL tensors | 802 | 0 |
| Greater/GreaterOrEqual | 802 | 0 |
| Cast | 802 | 0 |
| Where | 398 | 0 |
| Sign | 0 | 802 |
| Relu (added) | — | +802 |

Pipeline updated: new `_eliminate_bool_tensors()` helper runs as Step 10
in `postprocess_for_xcubeai()`. The function also re-runs onnxsim
afterwards so the new arithmetic can be fused where possible.

File: `C:/ai/dpsnn_fp32_final_v2.onnx`.

**Result:** X-CUBE-AI fully parsed the model and produced an analyze
report (`Computing AI RT data/code size...` stage). After 1 h wall-clock,
STM32CubeMX's GUI crashed trying to render the per-layer view of the
20 022 nodes (Java Swing doesn't scale to that), but the CLI summary
came through:

| Metric | Value | Budget (B-U585I-IOT02A) | Status |
| --- | --- | --- | --- |
| Weights (Flash) | 347.53 KB | 2048 KB | ✓ 17 % |
| Activations (RAM) | 1 433 408 B = 1.37 MiB | 786 KB | ✗ over by 78 % |
| MACC / inference | 363 M | — | — |

(`params # = 21,830,805 items (83.28 MiB)` counts every constant in the
graph including the 399 inflated zero-pad initializers; the 99.6 % drop
vs "float model" is X-CUBE-AI deduplicating them down to 347 KB.)

Flash fits with massive headroom; RAM is the bottleneck. Path forward is
INT8 quantization — already validated at −0.29 dB SI-SNR drop on Exp 5
(Round 6) — which should bring activations down ~4×.

### Pipeline applied to Exp 7 INT8 ONNX (2026-05-14)

Re-ran the same 10-step `postprocess_for_xcubeai()` on the existing
spike-aware INT8 ONNX (`export/dpsnn_scnn128_exp7_int8_pct95.onnx`, 6.3 MB,
20 850 nodes). All graph rewrites preserve QDQ pairs: Pad/ConvTranspose
inflation only adds Unsqueeze/Squeeze around the op (the QDQ wraps stay
intact), and BOOL elimination rewrites the spike-threshold branch which
is downstream of QDQ.

One small adjustment was needed: ConvTranspose's first input is a
`DequantizeLinear` output (no shape annotation in value_info), so
`rank_of(input[0])` returned `None` and inflation skipped them. Fix:
fall back to the rank of the weight initializer (always present, always
correct rank). After fix, all 399 ConvTransposes inflated.

Resulting graph:

| Op family | Count after |
| --- | --- |
| Conv | 1604 |
| ConvTranspose (now 4D) | 399 |
| Pad (now 4D) | 399 |
| QuantizeLinear / DequantizeLinear | 802 / 808 (preserved) |
| Greater / GreaterOrEqual / Cast / Where | 0 (all eliminated) |
| Sign / Identity | 802 / 802 (replacement) |
| Total nodes | 20 022 |
| File | `C:/ai/dpsnn_int8_final_v2.onnx` (4737 KB) |

Next: stedgeai analyze on the INT8 file — expecting activations ≈ 350 KB.

### Ninth fix: ConvTranspose time-as-H (2026-05-14)

First stedgeai analyze on `dpsnn_int8_final_v2.onnx` ran for ~25 min and
got to layer 47/167 before failing with:

```text
INTERNAL ERROR: Mismatch between filter size and weights: (1, 80) vs. (80, 1)
```

X-CUBE-AI computes the "filter size" by reading the ConvTranspose weight
spatial dims and compares them to the `kernel_shape` attribute. Our
inflated layout used **time-as-W** (kernel `(1, 80)`, weight
`(in_C, out_C, 1, K=80)`) — kernel and weight agree, but X-CUBE-AI
internally expects ConvTranspose with kernel along the H axis, i.e.
**time-as-H** (kernel `(80, 1)`, weight `(in_C, out_C, K=80, 1)`).
Bisect tests with single ConvTranspose nodes passed both orientations,
so the constraint only triggers inside the full graph.

**Fix:** `_inflate_pad_and_convtranspose()` now uses different inflation
orientations:

| Op | Unsqueeze axes (rank 3 input) | Kernel layout |
| --- | --- | --- |
| Pad | `[2]` (insert H before W) | spatial dim stays as W |
| ConvTranspose | `[3]` (append W after H) | spatial dim becomes H |

Also fixed weight detection: in INT8 mode the ConvTranspose weight comes
through a `DequantizeLinear` (so `node.input[1]` is the DQ output, not
the initializer). Follow the DQ producer chain to find the INT8 weight
initializer and rewrite *that* in place (the DQ's input). Output shape
follows once shape inference re-runs.

Result on `dpsnn_scnn128_exp7_int8_pct95.onnx`:

| Metric | Value |
| --- | --- |
| Pads inflated | 399 (time-as-W, kernel kept aligned with original) |
| ConvTransposes inflated | 399 (time-as-H, kernel `(80, 1)`) |
| Weight shapes after | `(128, 1, 80, 1)` (in_C, out_C, kH, kW) |
| QDQ pairs preserved | 802 Q + 808 DQ |
| Total nodes | 20 022 |
| File | `C:/ai/dpsnn_int8_final_v3.onnx` (4728 KB) |

Same FP32 rebuild: `C:/ai/dpsnn_fp32_final_v3.onnx` (4594 KB).

Next: stedgeai analyze on `dpsnn_int8_final_v3.onnx` — passed layer 17
within 5 min (previously failed at 47 after 25 min).

---

## Experiment Outcomes at a Glance

| Experiment | What | Outcome |
| --- | --- | --- |
| 0 | Pretrained N=256 inference | 18.08 dB SI-SNR, PESQ 2.264 — desktop ceiling |
| 1 | ONNX export of pretrained model | PASS, diff 4.01e-05; two upstream bugs fixed |
| 2 | ONNX numerical validation | PASS < 1e-3 |
| 1b | Standard INT8 on pretrained N=256 | FAILED — 6.53 dB (worse than noisy input) |
| 1c | Extended INT8 strategies on pretrained N=256 | Weight-only INT8 viable (−0.30 dB); all activation INT8 fails |
| 3 | SCNN-only N=128 first run (frame_dur=0.5) | DISCARDED — 9.52 dB ceiling; root cause: too few time steps |
| 4 | SCNN-only ONNX export of discarded checkpoint | PASS — confirms export pipeline works for SCNN-only |
| 5 | SCNN-only N=128 retrain (frame_dur=1.0, gradient clipping) | **17.60 dB val / 17.23 dB test** — first deployable model |
| 6 | INT8 quantization of SCNN-only N=128 (pct-95 spike-aware) | **16.99 dB, 140.9 KB Flash** — best INT8; quantization study contribution |
| 7 | SCNN-only N=128 retrain (bf16, no sparsity losses, 100 epochs) | **17.775 dB val / 17.41 dB test** — final trained model |
| 8 | W(INT8)+A(INT16) fake-quant study | **17.09 dB** — W8A16 quality ceiling; NOT deployable (opset 13 has no INT16 QDQ) |
| 9 | BatchNorm + stride=kernel=80 (no overlap) | **16.486 dB** — first batch model that fits 786 KB SRAM (501 KiB activations) |
| S | Streaming wrapper export (Exp 7, single-frame + state) | **17.42 dB**, max abs diff vs batch = 0.0; **46 KB activations** |
| D | On-device deployment + validation (Exp 7 streaming FP32, B-U585I-IOT02A) | **MCU 15.66 dB on p232_009 = Python ref 15.66 dB**, RTF 2.459× — deployment complete |

---

## FP32 Deployment Candidates — Exp 5 vs Exp 7 (824-utterance test set)

Same architecture (SCNN-only N=B=H=128, 71,299 params). Exp 7 retrained
with bf16 mixed precision, sparsity losses removed, and 100 epochs instead
of 49. The training speed-ups also *simplified* the inference graph by
removing the `readout_threshold` scalar and its `torch.where` op — Exp 7's
exported ONNX is 194 KB smaller than Exp 5's despite identical layer
shapes.

| Metric | Noisy | **Exp 7 (recommended)** | Exp 5 (previous) | Delta | Pretrained N=256 |
| --- | --- | --- | --- | --- | --- |
| SI-SNR (dB) | 8.44 | **17.41** | 17.23 | **+0.18** | 18.08 |
| PESQ (wb) | 1.971 | **2.115** | 2.089 | **+0.026** | 2.264 |
| STOI | 0.921 | **0.923** | 0.920 | **+0.003** | 0.925 |
| Composite OVRL | 2.637 | **2.591** | 2.480 | **+0.111** | 2.798 |
| Composite SIG | 3.357 | **3.115** | 2.935 | **+0.180** | 3.366 |
| Composite BAK | 2.445 | **2.947** | 2.909 | **+0.038** | 2.246 |
| Val SI-SNR (best) | — | 17.775 dB (ep 87) | 17.60 dB (ep 39) | +0.175 | — |
| ONNX size (FP32) | — | **5437 KB** | 5631 KB | −194 KB | — |
| Epochs to best | — | 87 (of 100) | 39 (of 49) | — | — |
| Train precision | — | bf16-mixed | fp32 | — | — |

**Verdict:** Exp 7 wins on every metric and produces a smaller ONNX. The
deployment issues we are currently hitting in X-CUBE-AI (If nodes, dead
nodes, empty Pad inputs, foreign opset imports) are artefacts of
`torch.onnx.export` + ORT, not of the training changes — they affect Exp 5
and Exp 7 identically. Continue with **Exp 7** as the deployment target.

---

## Full Quantization Comparison (824-utterance VoiceBank-DEMAND test set)

All numbers are for the **SCNN-only N=128** model. Noisy and FP32 rows are the reference.
Flash = weight bytes only (not ONNX file size).

| Model | Approach | SI-SNR (dB) | PESQ (wb) | STOI | OVRL | SIG | BAK | Flash |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Noisy input | — | 8.44 | 1.971 | 0.921 | 2.637 | 3.357 | 2.445 | — |
| FP32 ONNX | baseline | 17.23 | 2.089 | 0.920 | 2.480 | 2.935 | 2.909 | 278.5 KB |
| Naive INT8 v1 | ORT quantize_static, spike_map exclusion only | 7.64 | 1.328 | 0.846 | 1.341 | 1.456 | 2.197 | — |
| Naive INT8 v2 | ORT quantize_static, op_types restricted | 7.92 | 1.309 | 0.840 | 1.340 | 1.490 | 2.167 | — |
| Weight-only INT8 | All weights INT8, all activations FP32 | 16.92 | 1.757 | 0.916 | 1.825 | 2.007 | 2.655 | 140.9 KB |
| v3 corrected | QDQ after ReLU/Sigmoid, per-step scale | 15.73 | 1.801 | 0.916 | 2.011 | 2.311 | 2.689 | 140.9 KB |
| **v4 corrected** | **QDQ after ReLU/Sigmoid, shared scale** | **16.94** | **1.759** | **0.915** | **1.928** | **2.180** | **2.700** | **140.9 KB** |

**Drop vs FP32 (v4 recommended model):**

| SI-SNR | PESQ | STOI | OVRL | SIG | BAK |
| --- | --- | --- | --- | --- | --- |
| −0.29 dB | −0.330 | −0.005 | −0.552 | −0.755 | −0.209 |

---

## Exp9: BatchNorm + Non-Overlapping Decoder (2026-05-15)

### Motivation

Exp7 (17.41 dB, 17 KB SRAM over budget at 1.37 MB activations) was blocked from deployment by two structural issues: (1) ChannelWiseLayerNorm forces an FP32 island that X-CUBE-AI cannot quantize across, and (2) the 399-step unrolled OLA chain fills 1,196 × 64 KB buffers even after the memory planner rotates them. Exp9 attacks both simultaneously: replace LayerNorm with BatchNorm (folds at quant time) and set stride=kernel=80 (no overlap, no OLA chain).

### Training

- Command: `python vctk_trainer.py --scnn_only -L 80 --stride 80 -N 128 -B 128 -H 128 -X 1 --frame_dur 1.0 --norm_type bnorm --exp_name exp9_bn_nooverlap --batch_size 64 --max_epochs 100 --precision bf16-mixed`
- Checkpoint dir: `egs/voicebank/lightning_logs/version_4/checkpoints/`
- Best checkpoint: `epoch=99-val_loss=83.1804-val_sisnr=-16.8212.ckpt`
- Training time: ~13 hours (100 epochs × ~7.5 min/epoch)
- time_steps: 200 (down from 399; stride=kernel=80 means no overlap-add needed)

### Test-Set Results (824-sample VoiceBank-DEMAND test set)

| Metric | Exp7 (baseline) | Exp9 | Delta |
| --- | --- | --- | --- |
| SI-SNR (dB) | 17.41 | **16.486** | −0.93 dB |
| PESQ (wb) | 2.149 | 1.987 | −0.162 |
| STOI | 0.923 | 0.917 | −0.006 |
| OVRL | 2.584 | 2.509 | −0.075 |

Success criterion was ≥ 16.0 dB — **PASSED** (−0.93 dB vs Exp7, within the ≤1 dB budget).

### X-CUBE-AI Analysis

| Model | weights | activations | Budget | Notes |
| --- | --- | --- | --- | --- |
| Exp7 true-INT8 (best prev.) | 282 KiB | 1.37 MB | ❌ 2× over | OLA chain + LN island |
| **Exp9 FP32** | **280 KiB** | **501 KiB** | **✓ fits** | OLA eliminated; 285 KiB headroom |
| **Exp9 true-INT8** | **280 KiB** | **501 KiB** | **✓ fits** | Spike-path FP32 islands prevent further reduction |

Both Exp9 variants fit within the 786 KiB SRAM budget. INT8 saves only ~200 bytes over FP32 because the spike-path membrane potentials remain FP32 (intentional — quantizing them collapses spikes). The BN layer appears as `BatchNorm1d(128)` in the model; X-CUBE-AI is expected to fold it with adjacent Conv at code-gen time.

### ONNX Files

- FP32 raw: `export/dpsnn_exp9_bn_nooverlap.onnx` (2230.2 KB)
- FP32 X-CUBE-AI: `export/dpsnn_exp9_bn_nooverlap_xcubeai.onnx` (2075.7 KB)
- INT8 raw: `export/dpsnn_exp9_bn_nooverlap_true_int8.onnx` (2290.6 KB)
- INT8 X-CUBE-AI: `export/dpsnn_exp9_bn_nooverlap_true_int8_xcubeai.onnx` (2189.6 KB)
- Spike map: `export/dpsnn_exp9_bn_nooverlap.onnx.spike_map.json` (802 SAFE / 402 SPIKE nodes)

### Next Step

Run `stedgeai generate` on `C:/ai/dpsnn_exp9_bn_nooverlap.onnx` to produce C code for STM32U585. See `results/log_xcube_ai.md` §6 for updated next-steps.

---

## Experiment 10: SCNN-only N=64 — Channel Reduction Study (2026-06-03)

**Goal:** Measure the quality-latency trade-off of halving the channel width from N=128 to N=64. Motivated by supervisor feedback (meeting 2026-05-29) to address RTF=2.46× on the deployed N=128 model.

**Config:** Identical to Exp 7 except N=B=H=64. scnn_only=True, L=80, stride=40, frame_dur=1.0, bf16-mixed, 100 epochs, lr=1e-2, batch_size=64, no early stopping.

**Checkpoint:** `egs/voicebank/lightning_logs/version_5/checkpoints/epoch=92-val_loss=82.9734-val_sisnr=-17.0281.ckpt`
**Best val SI-SNR:** 17.03 dB (epoch 92)

### Test-Set Results (824-utterance VoiceBank-DEMAND)

| Metric | N=64 (Exp 10) | N=128 (Exp 7) | Delta |
|---|---|---|---|
| SI-SNR (dB) | **16.70** | 17.42 | −0.72 dB |
| PESQ (wb) | **2.022** | 2.149 | −0.127 |
| STOI | **0.922** | 0.923 | −0.001 |

### Deployment Footprint (stedgeai analyze, stm32u5, -O balanced -c none)

| Metric | N=64 | N=128 | Ratio |
|---|---|---|---|
| Weights (Flash) | 91 KB | 285 KB | 3.1× smaller |
| Activations (SRAM) | **23 KB** | 46 KB | 2.0× smaller |
| MACC/inference | 430,852 | 886,000 | 2.06× fewer |
| Predicted RTF | ~1.20× | 2.46× (measured) | ~2× faster |

**RTF prediction:** N=128 measured 1.12 cycles/MACC on 160 MHz Cortex-M33. Applying same ratio: 430,852 × 1.12 = 482,554 cycles → 3.02 ms/frame → RTF = 3.02/2.5 = **~1.20×**. Not real-time but approximately half the N=128 RTF.

### ONNX/Streaming Files

- Batch ONNX: `export/dpsnn_scnn64_exp10.onnx`
- Streaming ONNX: `export/dpsnn_streaming_n64.onnx`
- X-CUBE-AI ready: `export/dpsnn_streaming_n64_xcubeai.onnx`
- C:/ai copy: `c:/ai/dpsnn_streaming_n64.onnx`

### Conclusion

N=64 cuts MACC count by 2× and SRAM by 2×, at a cost of 0.72 dB SI-SNR. Measured RTF is 1.086× (8.6% over real-time), a 2.26× speed-up over N=128. Both configurations are deployed and validated on the B-U585I-IOT02A: N=128 is the quality-priority option (17.42 dB, RTF 2.46×), N=64 is the near-real-time option (16.70 dB, RTF 1.086×). Reaching RTF<1 on this MCU would require approximately N=32, which would likely carry a significant further quality penalty.

### On-Device Measurement — N=64 deployed (2026-06-03)

Built N=64 firmware (Stm_deployment.elf), flashed to B-U585I-IOT02A, ran p232_009 audio loop:

| Metric | Measured | Predicted | N=128 baseline |
|---|---|---|---|
| Per-frame latency | **2.715 ms** | 3.02 ms | 6.145 ms |
| RTF (vs 2.5 ms stride) | **1.086×** | 1.20× | 2.46× |
| MCU SI-SNR (p232_009) | **15.98 dB** | — | 15.66 dB |
| Python ONNX ref | 15.98 dB | — | 15.66 dB |
| MCU vs Python delta | **0.00 dB** | — | 0.00 dB |
| Speedup vs N=128 | **2.26×** | — | baseline |

**Key finding:** N=64 reaches RTF = 1.086× on the STM32U585 — within 0.215 ms of real-time per frame. The measured speedup (2.26×) is slightly better than predicted from MACC count alone (2.06×), likely because smaller weights also reduce L1 cache misses on the Cortex-M33.

MCU output is numerically identical to Python ONNX reference (delta = 0.00 dB on real audio), confirming the C code is correct.

RTF dropped from 2.46× to 1.09× by halving channel width, at a cost of −0.72 dB SI-SNR. Two operating points now on the same MCU: (N=128, 17.42 dB, 2.46×) and (N=64, 16.70 dB, 1.09×).

### Deployment Process — N=64 firmware (2026-06-03)

**1. C-code generation:** `stedgeai generate` on `c:/ai/dpsnn_streaming_n64.onnx` produced 7 files under `Stm_deployment/X-CUBE-AI/App/`:
`dpsnn_streaming_n64.{c,h}`, `dpsnn_streaming_n64_data.{c,h}`, `dpsnn_streaming_n64_data_params.{c,h}`, `dpsnn_streaming_n64_config.h`. Generation completes in 48 s, model_hash = 0xe5202bc8af9c81cd04c28fed1f851b8e.

**2. Firmware port from N=128 to N=64.** Eight edits in `Stm_deployment/X-CUBE-AI/App/`:

| File | What changed |
|---|---|
| `app_x-cube-ai.h:26-27` | include `dpsnn_streaming_n64.h` and `..._data.h` |
| `app_x-cube-ai.h:75` | `AI_MNETWORK_DATA_ACTIVATIONS_INT_SIZE` → N64 variant |
| `app_x-cube-ai.c:pool0[]` | `AI_DPSNN_STREAMING_N64_DATA_ACTIVATION_1_SIZE` (23,264 B) |
| `app_x-cube-ai.c:ctx_a/b[]` | `[512]` → `[256]` (B×4 channel-time slots, B halved) |
| `app_x-cube-ai.c:vp_a/b[]` and `mem_a/b[]` | `[128]` → `[64]` |
| `app_x-cube-ai.c:MX_X_CUBE_AI_Init` | all 3 calls renamed to `ai_dpsnn_streaming_n64_*` |
| `app_x-cube-ai.c:run` call in loop | `ai_dpsnn_streaming_n64_run` |
| `app_x-cube-ai.c:networks[]` table | all 10 entries renamed to `_n64` variants |

`tail_a/b[40]` unchanged — OLA tail is fixed at L−stride = 40 samples regardless of N.

**3. Build:** CubeIDE clean build → `Debug/Stm_deployment.elf` (2,804,896 bytes, 2026-06-03 12:19). 0 errors, 0 warnings.

**4. Flash:** Closed CubeIDE debug session, used STM32CubeProgrammer with ST-LINK + full erase + verify → Programming OK.

**5. aiValidation attempt: SKIPPED.** The firmware in `MX_X_CUBE_AI_Process()` runs the custom 1,662-frame audio loop on boot, not the X-CUBE-AI protobuf listener. `stedgeai validate --mode target` returns `E801(HwIOError): Invalid firmware - COM3:115200`. To enable aiValidation, the audio loop would need to be replaced with the original protobuf handler. Skipped because the audio-loop test alone proves numerical correctness (MCU vs Python ONNX delta of 0.00 dB on real audio is a strictly stronger check than aiValidation on random inputs).

**6. Reference SI-SNR regeneration:** `tools/wav_to_c_array.py` had hardcoded `N=128` for state-tensor shape allocation. Patched to infer `N` from `context_win` input shape:
```python
inputs = {i.name: i.shape for i in sess.get_inputs()}
N = inputs["context_win"][1]
```
After the patch: Python ONNX reference SI-SNR for N=64 on p232_009 = **15.98 dB** (vs 15.66 dB for N=128 on the same utterance — N=64 happens to be slightly better on this one clip even though it's worse on the 824-set average).

**7. Audio test:** `tools/mcu_audio_receiver.py --clean deploy/test_clean.wav --ref_bin deploy/test_enhanced_ref.bin` over COM3 @ 115200 baud. Reset board → received 66,480 samples (4.16 s) → MCU SI-SNR = 15.98 dB matched Python ref exactly. Latency from firmware-side DWT counter: 2.715 ms/frame, RTF 1.086×.

### Architectural artifact: broadband residual hiss (both N=128 and N=64)

There is a "loud rustling" / broadband background hiss audible in the N=64 MCU output. I initially thought it was reduced mask resolution from halving the channel count — that was wrong. The same hiss is audible in the N=128 deployed output too. It is a DPSNN-family architectural characteristic, not a deployment or channel-reduction artifact.

**Diagnostic evidence (already in the data, not previously highlighted):**

Composite BAK (background-noise quality) from the 824-utterance test set:

| Model | Composite BAK | vs noisy (2.445) |
|---|---|---|
| Noisy input | 2.445 | — |
| Pretrained DPSNN N=256 (Sun & Bohté) | 2.246 | **−0.20 (worse than noisy)** |
| Exp 7 N=128 streaming | 2.947 | +0.50 |
| Exp 10 N=64 streaming | 2.080 | **−0.37 (worse than noisy)** |

The pretrained N=256 reference model **also** scores worse on BAK than the noisy input. This is a known property of the SCNN-only DPSNN architecture trained with an SI-SNR + small-MSE objective: the loss is scale-invariant and rewards alignment, but does not include any perceptual or noise-floor penalty. Spike-based binary masking is also inherently discrete, producing structured high-frequency artefacts during decoder OLA reconstruction.

**Implication:**
- The artifact is real and audible but is a model-level limitation of DPSNN, not a deployment bug.
- MCU output matches Python ONNX output sample-for-sample (delta 0.00 dB on real audio for both N=128 and N=64), so the C code is correct.
- N=64's perceptual quality is somewhat worse than N=128's (PESQ −0.127, BAK −0.87), but the *baseline* broadband residual is present in both.

**Thesis Discussion sentence to add (one-line scope marker):**

> The deployed model retains a low-level broadband residual in its output. This is an architectural characteristic of the SCNN-only DPSNN family: the SI-SNR + MSE training objective contains no perceptual or noise-floor term, and binary spike masking produces structured high-frequency artefacts. The same residual is present in the offline ONNX reference and in the pretrained N=256 baseline (composite BAK 2.95 and 2.25 respectively); it is a model-level limitation rather than a deployment artefact.

### Exp 10 — Summary of thesis-relevant numbers

| Quantity | N=128 (Exp 7, deployed) | N=64 (Exp 10) |
|---|---|---|
| Trainable params | 71,299 | 23,430 |
| Weights (Flash) | 285 KB | 91 KB |
| Activations (SRAM) | 46 KB | 23 KB |
| MACC / inference | 886,000 | 430,852 |
| Test SI-SNR (dB) | 17.42 | 16.70 |
| Test PESQ (wb) | 2.149 | 2.022 |
| Test STOI | 0.923 | 0.922 |
| Test composite BAK | 2.947 | 2.080 |
| MCU latency / frame | 6.145 ms | 2.715 ms |
| RTF | 2.46× | **1.086×** |
| MCU vs Python ONNX delta | 0.00 dB | 0.00 dB |


---

## Exp 11 - ICECS Latency Experiments (conf branch, 2026-06-10/11)

Full record: [log_conf_latency.md](log_conf_latency.md). Branch: `conf`.

**Trigger:** ICECS deadline extended to June 15; Tao asked for inference-time
experiments (remove binarization; remove the two 1x1 convs; ALIF->PLIF
readout), random parameters OK. We added a fourth, lossless change found via
the per-layer compiler report: the streaming decoder ConvTranspose acts on a
length-1 frame, so it equals an N->80 matrix-vector product (Dense/Gemm).

**Measured on-device (DWT, p232_009, 1,662 frames, 160 MHz):**

| Variant | N=64 ms/frame | RTF |
| --- | --- | --- |
| baseline (control - matches Exp 10 exactly) | 2.715 | 1.086 |
| remove binarization | 2.683 | 1.073 |
| remove two 1x1 convs | 2.282 | 0.913 |
| ALIF->PLIF readout | 2.723 | 1.089 |
| all three combined | 2.253 | 0.901 |
| decoder Gemm rewrite (lossless) | 2.564 | 1.025 |
| decoder rewrite + all three | **2.101** | **0.840** |

N=128 decoder rewrite: 5.991 ms (RTF 2.40) - still not real-time.

**Key findings:**

1. **Real-time achieved at N=64** (RTF 0.840 best config; the 1x1-conv
   removal alone already crosses the line at 0.913).
2. **The compiler MACC report does not predict latency.** The decoder
   ConvTranspose billed at 95% of c-model MACC costs only ~24K cycles
   (~0.15 ms); the two 1x1 convs billed at 1.9% cost 16% of cycles. The
   pre-registered MACC-based predictions in log_conf_latency.md SS5 were off
   by 15x - kept as a record. True bottleneck after the decoder fix:
   LayerNorm + elementwise spike chains on tiny tensors (15.5 cycles/MACC).
3. **Decoder Gemm rewrite is lossless on hardware**: MCU SI-SNR delta
   0.00 dB at both widths (15.98 / 15.66 dB on p232_009); also cuts
   activation SRAM 23.3->4.3 KB (N=64) and 47->8.7 KB (N=128).
4. Ablation quality is unknown without retraining (out of scope per Tao's
   email); recommended next step is retraining `combined` at N=64.

**Infrastructure:** measurement loop fully automated
(tools/run_conf_campaign.ps1: stedgeai generate -> headless CubeIDE build ->
CLI flash -> UART timing capture; ~70 s per variant). Firmware state buffers
now sized from generated header macros so any width builds unchanged.

---

## Exp 12 - Retrained no-pointwise N=64 (Tao-approved real-time config, 2026-06-12)

Full record: [log_conf_latency.md](log_conf_latency.md) SS7. Branch: `conf`.

Tao approved removing binarization + both 1x1 convs together (ALIF kept) and
training it. New `--no_pointwise` flag; recipe otherwise identical to Exp 10.
15,042 params (-8,321). 100 epochs, 15.2 h on RTX 4060, best ckpt epoch 99
(val plateau from ~82) -> `models/dpsnn_n64_noptwise.ckpt`.

| Quantity | Exp 10 baseline | Exp 12 no-pointwise |
| --- | --- | --- |
| Test SI-SNR | 16.70 dB | **16.23 dB** (-0.47) |
| Test PESQ (wb) | 2.022 | 1.959 (-0.063) |
| Test STOI | 0.922 | 0.917 (-0.005) |
| MCU ms/frame | 2.715 | **2.247** (2.097 with decoder Gemm) |
| RTF | 1.086 | **0.899** (**0.839** with decoder Gemm) |
| MCU vs ONNX (p232_009) | 0.00 dB | 0.00 dB (15.89 dB) |

**Bottom line: real-time speech enhancement on the STM32U585 with trained
weights - 2.097 ms/frame (RTF 0.839) at 16.23 dB test SI-SNR**, costing
0.47 dB vs the non-real-time Exp 10 baseline. Measured latencies match the
random-weight predictions from Exp 11 (2.253/2.101), confirming
weight-independence of the timing.

# Experiment Log

## Setup
- Repo: dpsnn-embedded-deployment
- Strategy: Validate-First — test ONNX export before training simplified models

---

## Experiment 0: Pretrained Model Inference
- Date: [fill in]
- Checkpoint: egs/voicebank/epoch=478-val_loss=81.5449-val_sisnr=-18.4556.ckpt
- Params: N=256, B=256, H=256, L=80, stride=40, context_dur=0.01, X=1
- Status: [PENDING]

### Command Used
```bash
cd egs/voicebank
python -u vctk_trainer.py --config vctk.yaml \
    -L 80 --stride 40 -N 256 -B 256 -H 256 \
    --context_dur 0.01 --max_epochs 500 -X 1 --lr 1e-2 \
    --test_ckpt_path ./epoch=478-val_loss=81.5449-val_sisnr=-18.4556.ckpt
```

### Desktop Baseline Metrics (Pretrained)
| Metric | Noisy (input) | Enhanced | Clean (reference) |
|--------|--------------|---------|------------------|
| SI-SNR (dB) | — | — | — |
| PESQ (wb) | — | — | — |
| STOI | — | — | — |

### Any Issues Encountered
[None / describe fixes needed]

---

## Experiment 1: ONNX Export of Pretrained Model
- Date: [fill in]
- Script: `python export/export_to_onnx.py --ckpt_path egs/voicebank/epoch=478-... --output_path export/dpsnn_pretrained.onnx`
- Status: [PENDING]

### Export Result
- [ ] SUCCEEDED — proceed with DPSNN as primary target
- [ ] SUCCEEDED WITH WARNINGS — investigate warnings, proceed cautiously
- [ ] FAILED ON SPECIFIC OPS — see error below, attempt inference-mode rewrite
- [ ] FAILED (fundamental) — pivot to Conv-TasNet, document as thesis finding

### Error (if any)
```
[paste full traceback here]
```

### File Size (if succeeded)
- FP32 ONNX: [X] KB

---

## Experiment 2: ONNX Numerical Validation
- Date: [fill in]
- Script: `python export/validate_onnx.py --ckpt_path ... --onnx_path export/dpsnn_pretrained.onnx`
- Status: [PENDING]

### Validation Result
- Max absolute difference (PyTorch vs ONNX Runtime): [X.XXe-XX]
- [ ] PASS (diff < 1e-3)
- [ ] WARN (1e-3 ≤ diff < 1e-1)
- [ ] FAIL (diff ≥ 1e-1)

---

## Decision After Phase 1
- [ ] **DPSNN path** — ONNX export works, proceed to simplify DPSNN and train
- [ ] **Conv-TasNet fallback** — ONNX export failed, Conv-TasNet becomes primary deployment target
- Next step: [Phase 2 plan — link once written]
---

"""Sanity check: enable_fused_ola() produces bit-equivalent output to the
original per-frame OLA path.

Loads the Exp 7 best checkpoint, runs both forward variants on the same
input, prints max-abs-diff. Expect < 1e-5 (fp32 reduction-order rounding).
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from export.export_to_onnx import enable_fused_ola, load_from_checkpoint

CKPT = "egs/voicebank/lightning_logs/version_2/checkpoints/epoch=87-val_loss=82.2283-val_sisnr=-17.7750.ckpt"

torch.manual_seed(0)

# Load twice so we have an untouched copy.
m_ref = load_from_checkpoint(CKPT)
m_fused = load_from_checkpoint(CKPT)

input_dim = m_ref.hparams["input_dim"]
print(f"input_dim={input_dim} time_steps={m_ref.time_steps} stride={m_ref.stride} L={m_ref.L}")

dummy_id = torch.zeros(1)
dummy_len = torch.tensor(input_dim)
x = torch.randn(1, input_dim)

inputs = (dummy_id, x, dummy_len)
targets = (dummy_id, torch.zeros_like(x), dummy_len)
batch = (inputs, targets)

with torch.no_grad():
    out_ref, *_ = m_ref(batch)

enable_fused_ola(m_fused)
with torch.no_grad():
    out_fused, *_ = m_fused(batch)

print(f"out_ref   shape={tuple(out_ref.shape)}  min={float(out_ref.min()):.5f}  max={float(out_ref.max()):.5f}")
print(f"out_fused shape={tuple(out_fused.shape)}  min={float(out_fused.min()):.5f}  max={float(out_fused.max()):.5f}")

diff = (out_ref - out_fused).abs()
print(f"max-abs-diff = {float(diff.max()):.6e}")
print(f"mean-abs-diff= {float(diff.mean()):.6e}")
rel = diff.max() / out_ref.abs().max().clamp_min(1e-9)
print(f"max-rel-diff = {float(rel):.6e}")

if float(diff.max()) < 1e-4:
    print("PASS: fused output matches per-frame within fp32 tolerance.")
else:
    print("FAIL: fused output diverges from per-frame OLA!")
    sys.exit(1)

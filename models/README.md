# Deployed models

Ready-to-run artifacts for the SCNN-only DPSNN deployed on the STM32U585. These
let you run inference / re-compile without retraining.

| File | What it is |
|------|------------|
| `dpsnn_n128.ckpt` | Trained PyTorch checkpoint, N=B=H=128 (Exp 7, best validation epoch). 17.42 dB SI-SNR on the VoiceBank-DEMAND test set. |
| `dpsnn_n64.ckpt` | Trained PyTorch checkpoint, channel-reduced N=B=H=64 (Exp 10, best validation epoch). 16.70 dB SI-SNR. |
| `dpsnn_n128_streaming.onnx` | Single-frame streaming ONNX export of the N=128 model (five explicit state tensors at the graph boundary). |
| `dpsnn_n128_streaming_xcubeai.onnx` | N=128 streaming graph after the X-CUBE-AI compatibility postprocess (the graph actually compiled to the device). |
| `dpsnn_n64_streaming.onnx` | Single-frame streaming ONNX export of the N=64 model. |
| `dpsnn_n64_streaming_xcubeai.onnx` | N=64 streaming graph after the X-CUBE-AI compatibility postprocess. |

The pretrained N=256 baseline checkpoint released by Sun & Bohté is kept separately
at `egs/voicebank/epoch=478-val_loss=81.5449-val_sisnr=-18.4556.ckpt`.

The VoiceBank-DEMAND dataset is **not** bundled (it is ~21 GB); see the top-level
`README.md` for how to obtain it.

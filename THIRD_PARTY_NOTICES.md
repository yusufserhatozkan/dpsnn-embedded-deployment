# Third-party notices

This repository builds on the following external work. The MIT license in
`LICENSE` covers the original code in this repository; the components below
retain the terms of their respective sources.

## DPSNN (Sun & Bohté)

The core network code under `dpsnn/` and the training recipe under
`egs/voicebank/` are derived from
[`tao-sun/dpsnn`](https://github.com/tao-sun/dpsnn) by Tao Sun and
Sander Bohté (CWI Amsterdam), which accompanies:

> T. Sun and S. Bohté, "DPSNN: Spiking neural network for low-latency
> streaming speech enhancement," Neuromorphic Computing and Engineering, 2024.

The pretrained N=256 baseline checkpoint at
`egs/voicebank/epoch=478-val_loss=81.5449-val_sisnr=-18.4556.ckpt` was
released by the DPSNN authors and is redistributed here for reproducibility.
This work was carried out under the supervision of Tao Sun.

## Composite speech-quality metrics

`dpsnn/data/metrics.py` (CSIG / CBAK / COVL) is taken from
[`facebookresearch/denoiser`](https://github.com/facebookresearch/denoiser)
(`scripts/matlab_eval.py`), which is in turn adapted from
[`santi-pdp/segan_pytorch`](https://github.com/santi-pdp/segan_pytorch) and
distributed under the MIT License.

## DNSMOS

`dpsnn/data/dnsmos.py` and the bundled model `dpsnn/data/sig_bak_ovr.onnx`
come from Microsoft's
[`DNS-Challenge`](https://github.com/microsoft/DNS-Challenge) repository
(code under the MIT License). If you use DNSMOS results in research, cite:

> C. K. A. Reddy, V. Gopal and R. Cutler, "DNSMOS P.835: A non-intrusive
> perceptual objective speech quality metric to evaluate noise suppressors,"
> ICASSP 2022.

## VoiceBank-DEMAND dataset

The evaluation audio under `deploy/` and `deploy/test_audio/` is drawn from
the noisy VoiceBank-DEMAND corpus (Valentini-Botinhao et al.), available from
the [Edinburgh DataShare](https://datashare.ed.ac.uk/handle/10283/2791) under
the Creative Commons Attribution 4.0 licence.

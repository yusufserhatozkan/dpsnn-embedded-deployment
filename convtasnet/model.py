"""
Conv-TasNet for speech enhancement (Luo & Mesgarani, 2018).

API-compatible with StreamSpikeNet so it plugs into vctk_trainer.py via
  --model convtasnet -X 1 -N 48 -B 48 -H 96 --P 3 --tcn_depth 3 --tcn_repeats 1
The X arg must be 1 for Conv-TasNet (controls context stages in the trainer,
not TCN depth); use --tcn_depth for the number of TCN blocks per repeat.

Default N=48, B=48, H=96, P=3, tcn_depth=3, tcn_repeats=1 → ~57K params,
matched to the SCNN-only DPSNN for a fair parameter comparison.
"""
import numpy as np
import torch
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning.utilities import rank_zero_info

from dpsnn.layers.sdr import singlesrc_neg_sisdr


class _TCNBlock(nn.Module):
    """One depthwise-separable TCN block (residual + skip)."""

    def __init__(self, B: int, H: int, P: int, dilation: int):
        super().__init__()
        padding = (P - 1) * dilation // 2
        self.net = nn.Sequential(
            nn.Conv1d(B, H, 1),
            nn.PReLU(),
            nn.GroupNorm(1, H, eps=1e-8),
            nn.Conv1d(H, H, P, dilation=dilation, padding=padding, groups=H),
            nn.PReLU(),
            nn.GroupNorm(1, H, eps=1e-8),
        )
        self.res_conv = nn.Conv1d(H, B, 1)
        self.skip_conv = nn.Conv1d(H, B, 1)

    def forward(self, x):
        out = self.net(x)
        return self.res_conv(out) + x, self.skip_conv(out)


class ConvTasNet(pl.LightningModule):
    """
    Conv-TasNet speech enhancer, StreamSpikeNet-compatible LightningModule.

    Parameters
    ----------
    input_dim     : total input samples (output_size + context_dim, e.g. 8160)
    context_dim   : leading context samples to strip before encoding (e.g. 160)
    N             : encoder channels
    B             : bottleneck / TCN channels
    H             : TCN hidden (depthwise) channels
    P             : depthwise kernel size
    tcn_depth     : TCN blocks per repeat (dilation 1,2,4,...,2^(tcn_depth-1))
    tcn_repeats   : number of repeats of the full TCN block sequence
    """

    def __init__(self, input_dim, context_dim, sr=16000,
                 L=80, stride=40,
                 N=48, B=48, H=96, P=3,
                 tcn_depth=3, tcn_repeats=1,
                 learning_rate=1e-3):
        super().__init__()
        self.save_hyperparameters()

        self.context_dim = context_dim
        self.sr = sr
        self.L = L
        self.stride = stride
        self.X = 1          # always 1 — trainer uses this only for context stage count
        self.lr = learning_rate

        # Encoder / decoder
        self.encoder = nn.Conv1d(1, N, L, stride=stride, bias=False)
        self.encoder_norm = nn.GroupNorm(1, N, eps=1e-8)
        self.decoder = nn.ConvTranspose1d(N, 1, L, stride=stride, bias=False)

        # Separator
        self.bottleneck = nn.Conv1d(N, B, 1)
        self.tcn_blocks = nn.ModuleList([
            _TCNBlock(B, H, P, dilation=2 ** x)
            for _ in range(tcn_repeats)
            for x in range(tcn_depth)
        ])
        self.mask_net = nn.Sequential(
            nn.PReLU(),
            nn.Conv1d(B, N, 1),
            nn.Sigmoid(),
        )

        # Ops dict for TestCallback compatibility (no sparsity — all ops fire)
        sep_ops = tcn_repeats * tcn_depth * (B * H + H * P + 2 * H * B)
        self.ops = {
            'fixed_syn_ops': N * L + B * N + sep_ops + N * L,
            'fixed_sep_syn_ops': B * N + sep_ops,
            'event_syn_ops': [sep_ops],   # single entry, event rate = 1.0
            'neuron_ops': 0,
        }

        total = sum(p.numel() for p in self.parameters())
        rank_zero_info(
            f"ConvTasNet  N={N} B={B} H={H} P={P} "
            f"tcn_depth={tcn_depth} tcn_repeats={tcn_repeats}  "
            f"params={total:,}"
        )

    # ------------------------------------------------------------------
    # Core forward
    # ------------------------------------------------------------------

    def _separate(self, signal: torch.Tensor) -> torch.Tensor:
        """signal: (batch, T) → enhanced: (batch, T)"""
        x = self.encoder(signal.unsqueeze(1))   # (B, N, frames)
        x = self.encoder_norm(x)
        w = x                                    # save for mask application
        x = self.bottleneck(x)                  # (B, B_ch, frames)
        skip_sum = torch.zeros_like(x)
        for block in self.tcn_blocks:
            x, skip = block(x)
            skip_sum = skip_sum + skip
        mask = self.mask_net(skip_sum)           # (B, N, frames)
        return self.decoder(w * mask).squeeze(1) # (B, T)

    def forward(self, batch):
        inputs, _ = batch
        _, noisy_x, _ = inputs                   # (batch, input_dim)
        signal = noisy_x[:, self.context_dim:]   # (batch, output_size)
        enhanced = self._separate(signal)

        # Dummy event rates: ANN fires every step → rate = 1.0
        event_rates = torch.ones(1, device=noisy_x.device)
        # Return (enhanced, event_rates, dummy, dummy) matching StreamSpikeNet
        return enhanced, event_rates, signal, signal

    def common_step(self, batch, batch_idx):
        return self(batch)

    # ------------------------------------------------------------------
    # Lightning steps
    # ------------------------------------------------------------------

    def training_step(self, batch, batch_idx):
        inputs, targets = batch
        _, y, _ = targets
        batch_size = y.shape[0]
        out, _, _, _ = self.common_step(batch, batch_idx)
        y = torch.squeeze(y, dim=1)[:, self.context_dim:]
        sisnr_loss = singlesrc_neg_sisdr(out, y).mean()
        mse_loss = nn.MSELoss()(out, y)
        loss = 0.001 * mse_loss + 100 + sisnr_loss

        self.log("loss", loss,      prog_bar=True, sync_dist=True, batch_size=batch_size)
        self.log("sisnr", sisnr_loss, prog_bar=True, sync_dist=True, batch_size=batch_size)
        self.log("mse", mse_loss,   prog_bar=True, sync_dist=True, batch_size=batch_size)
        self.log("lr", self.lr_schedulers().get_last_lr()[0],
                 prog_bar=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        inputs, targets = batch
        _, y, _ = targets
        batch_size = y.shape[0]
        out, _, _, _ = self.common_step(batch, batch_idx)
        y = torch.squeeze(y, dim=1)[:, self.context_dim:]
        sisnr_loss = singlesrc_neg_sisdr(out, y).mean()
        mse_loss = nn.MSELoss()(out, y)
        loss = 0.001 * mse_loss + 100 + sisnr_loss

        self.log("val_loss",  loss,       on_epoch=True, prog_bar=True,
                 sync_dist=True, batch_size=batch_size)
        self.log("val_sisnr", sisnr_loss, on_epoch=True, prog_bar=True,
                 sync_dist=True, batch_size=batch_size)
        return loss

    def test_step(self, batch, batch_idx):
        out, event_rates, _, _ = self.common_step(batch, batch_idx)
        return out, event_rates

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        stepping_batches = self.trainer.estimated_stepping_batches
        min_lr = 1e-4
        est_final = self.lr * (0.9999 ** stepping_batches)
        gamma = (0.9999 if est_final >= min_lr
                 else (min_lr / self.lr) ** (1 / stepping_batches))
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }

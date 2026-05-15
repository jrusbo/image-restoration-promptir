import torch
import torch.nn as nn
import torch.fft


class CharbonnierLoss(nn.Module):
    """Stable, edge-preserving spatial penalty."""

    def __init__(self, eps=1e-3):
        super().__init__()
        self.eps = eps

    def forward(self, pred, target):
        return torch.mean(torch.sqrt((pred - target) ** 2 + self.eps ** 2))


class FFTLoss(nn.Module):
    """Suppresses high-frequency peaks in the spectral domain."""

    def __init__(self):
        super().__init__()

    def forward(self, pred, target):
        pred_fft = torch.fft.fft2(pred, norm='backward')
        target_fft = torch.fft.fft2(target, norm='backward')

        pred_amp = torch.abs(pred_fft)
        target_amp = torch.abs(target_fft)

        return torch.mean(torch.abs(pred_amp - target_amp))


class CompositeLoss(nn.Module):
    """L_Total = L_Char + lambda * L_FFT"""

    def __init__(self, fft_weight=0.1):
        super().__init__()
        self.charbonnier = CharbonnierLoss(eps=1e-3)
        self.fft = FFTLoss()
        self.fft_weight = fft_weight

    def forward(self, pred, target):
        loss_char = self.charbonnier(pred, target)
        loss_fft = self.fft(pred, target)
        total_loss = loss_char + (self.fft_weight * loss_fft)

        # Return total loss and a dictionary of individual metrics for W&B
        return total_loss, {
            "loss_char": loss_char.item(),
            "loss_fft": loss_fft.item()
        }
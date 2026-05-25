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
        pred_fft = torch.fft.rfft2(pred, norm='backward')
        target_fft = torch.fft.rfft2(target, norm='backward')

        pred_amp = torch.abs(pred_fft)
        target_amp = torch.abs(target_fft)

        return torch.mean(torch.abs(pred_amp - target_amp))


class EdgeLoss(nn.Module):
    """Penalizes structural differences using Sobel operators."""

    def __init__(self):
        super().__init__()
        # Sobel kernels
        k_x = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]]).view(1, 1, 3, 3)
        k_y = torch.tensor([[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]]).view(1, 1, 3, 3)

        self.register_buffer('weight_x', k_x.repeat(3, 1, 1, 1))  # For RGB
        self.register_buffer('weight_y', k_y.repeat(3, 1, 1, 1))
        self.criterion = nn.L1Loss()

    def forward(self, pred, target):
        # Extract gradients (edges)
        pred_grad_x = nn.functional.conv2d(pred, self.weight_x, padding=1, groups=3)
        pred_grad_y = nn.functional.conv2d(pred, self.weight_y, padding=1, groups=3)

        target_grad_x = nn.functional.conv2d(target, self.weight_x, padding=1, groups=3)
        target_grad_y = nn.functional.conv2d(target, self.weight_y, padding=1, groups=3)

        loss_x = self.criterion(pred_grad_x, target_grad_x)
        loss_y = self.criterion(pred_grad_y, target_grad_y)

        return loss_x + loss_y


class CompositeLoss(nn.Module):
    """L_Total = L_Char + lambda_fft * L_FFT + lambda_edge * L_Edge"""

    def __init__(self, fft_weight=0.1, edge_weight=0.1):
        super().__init__()
        self.charbonnier = CharbonnierLoss(eps=1e-3)
        self.fft = FFTLoss()
        self.edge = EdgeLoss()
        self.fft_weight = fft_weight
        self.edge_weight = edge_weight

    def forward(self, pred, target):
        loss_char = self.charbonnier(pred, target)
        loss_fft = self.fft(pred, target)
        loss_edge = self.edge(pred, target)
        total_loss = loss_char + (self.fft_weight * loss_fft) + (self.edge_weight * loss_edge)

        # Return total loss and a dictionary of individual metrics for W&B
        # We return tensors; the training loop will call .item() to avoid graph breaks
        return total_loss, {
            "loss_char": loss_char,
            "loss_fft": loss_fft,
            "loss_edge": loss_edge
        }
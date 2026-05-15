import argparse
import os
import random

import numpy as np

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import wandb

from accelerate import Accelerator
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure

from model.promptir import PromptIR
from dataset import RestorationDataset
from metrics import CompositeLoss


def set_seed(seed: int) -> None:
    """Fix seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def apply_dense_mixup_cutmix(x: torch.Tensor, y: torch.Tensor, alpha: float = 1.2):
    """Custom dense blending for Image-to-Image tasks."""
    batch_size = x.size(0)
    index = torch.randperm(batch_size, device=x.device)
    lam = np.random.beta(alpha, alpha)

    if random.random() > 0.5:  # MixUp
        mixed_x = lam * x + (1 - lam) * x[index]
        mixed_y = lam * y + (1 - lam) * y[index]
    else:  # CutMix
        W, H = x.size(2), x.size(3)
        cut_rat = np.sqrt(1. - lam)
        cut_w, cut_h = int(W * cut_rat), int(H * cut_rat)
        cx, cy = np.random.randint(W), np.random.randint(H)

        bbx1, bby1 = np.clip(cx - cut_w // 2, 0, W), np.clip(cy - cut_h // 2, 0, H)
        bbx2, bby2 = np.clip(cx + cut_w // 2, 0, W), np.clip(cy + cut_h // 2, 0, H)

        mixed_x, mixed_y = x.clone(), y.clone()
        mixed_x[:, :, bbx1:bbx2, bby1:bby2] = x[index, :, bbx1:bbx2, bby1:bby2]
        mixed_y[:, :, bbx1:bbx2, bby1:bby2] = y[index, :, bbx1:bbx2, bby1:bby2]

    return mixed_x, mixed_y


def main():
    parser = argparse.ArgumentParser(description="Train PromptIR with Accelerate")
    parser.add_argument("--data_dir", type=str, default="dataset/train")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--min_lr", type=float, default=1e-6)
    parser.add_argument("--save_dir", type=str, default="checkpoints")
    args = parser.parse_args()

    # LIBRARY: Initialize Accelerator (replaces mp.spawn and wandb.init manual logic)
    accelerator = Accelerator(log_with="wandb", mixed_precision="fp16")
    set_seed(42 + accelerator.process_index)

    if accelerator.is_main_process:
        os.makedirs(args.save_dir, exist_ok=True)
        accelerator.init_trackers("PromptIR-Restoration", config=vars(args))

    dataset = RestorationDataset(root_dir=args.data_dir, is_train=True)
    # Accelerator auto-injects DistributedSampler if on multiple GPUs
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)

    model = PromptIR()
    model = torch.compile(model)

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.min_lr)
    criterion = CompositeLoss(fft_weight=0.1)

    psnr_metric = PeakSignalNoiseRatio(data_range=1.0).to(accelerator.device)
    ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(accelerator.device)

    model, optimizer, dataloader, scheduler, criterion = accelerator.prepare(
        model, optimizer, dataloader, scheduler, criterion
    )

    best_psnr = 0.0

    epoch_iterator = tqdm(range(args.epochs), desc="Overall Progress", disable=not accelerator.is_local_main_process,
                          dynamic_ncols=True)
    for epoch in epoch_iterator:
        model.train()
        psnr_metric.reset()
        ssim_metric.reset()

        epoch_losses = []
        pbar = tqdm(dataloader, desc=f"Epoch [{epoch + 1}/{args.epochs}]", leave=False,
                    disable=not accelerator.is_local_main_process)

        visual_sample = None

        for degraded, clean in pbar:
            # accelerator automatically places tensors on the correct device
            if visual_sample is None and accelerator.is_main_process:
                visual_sample = (degraded[:1].clone(), clean[:1].clone())

            if random.random() < 0.5:
                degraded, clean = apply_dense_mixup_cutmix(degraded, clean)

            optimizer.zero_grad()

            output = model(degraded)
            loss, loss_dict = criterion(output, clean)

            accelerator.backward(loss)

            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(model.parameters(), 0.01)

            optimizer.step()

            output_clamped = torch.clamp(output, 0.0, 1.0)
            psnr_metric.update(output_clamped, clean)

            epoch_losses.append(loss.item())
            if accelerator.is_local_main_process:
                pbar.set_postfix({'Loss': f"{loss.item():.4f}"})

        scheduler.step()

        # Wait for all GPUs to finish the epoch
        accelerator.wait_for_everyone()

        if accelerator.is_main_process:
            avg_loss = np.mean(epoch_losses)
            current_psnr = psnr_metric.compute().item()

            # Log visual progress
            model.eval()
            with torch.no_grad():
                sample_pred = torch.clamp(model(visual_sample[0]), 0.0, 1.0)
                stitched = torch.cat([visual_sample[0][0], sample_pred[0], visual_sample[1][0]], dim=2)
                accelerator.log({"Visuals/Restoration": wandb.Image(stitched, caption="In | Pred | GT")}, step=epoch)

            accelerator.log({"Train/Loss": avg_loss, "Train/PSNR": current_psnr}, step=epoch)

            unwrapped_model = accelerator.unwrap_model(model)

            if current_psnr > best_psnr:
                best_psnr = current_psnr
                torch.save(unwrapped_model.state_dict(), os.path.join(args.save_dir, "best_model.pth"))
                tqdm.write(f"New Best Model Saved (PSNR: {best_psnr:.2f})")

    accelerator.end_training()


if __name__ == "__main__":
    main()
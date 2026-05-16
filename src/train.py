import argparse
import os
import sys
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
    parser = argparse.ArgumentParser(description="Train PromptIR with Accelerate & Validation")
    parser.add_argument("--data_dir", type=str, default="dataset/train")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--min_lr", type=float, default=1e-6)
    parser.add_argument("--save_dir", type=str, default="checkpoints")
    args = parser.parse_args()

    accelerator = Accelerator(log_with="wandb", mixed_precision="fp16")
    set_seed(42 + accelerator.process_index)

    if accelerator.is_main_process:
        os.makedirs(args.save_dir, exist_ok=True)
        accelerator.init_trackers("PromptIR-Restoration", config=vars(args))

    # Initialize Train and Validation Datasets
    train_dataset = RestorationDataset(root_dir=args.data_dir, mode='train', val_split=0.1)
    val_dataset = RestorationDataset(root_dir=args.data_dir, mode='val', val_split=0.1)

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4,
                                  pin_memory=True)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    model = PromptIR()
    # model = torch.compile(model)

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.min_lr)
    criterion = CompositeLoss(fft_weight=0.1)

    # Separate metrics for Train and Val
    train_psnr_metric = PeakSignalNoiseRatio(data_range=1.0).to(accelerator.device)
    val_psnr_metric = PeakSignalNoiseRatio(data_range=1.0).to(accelerator.device)
    val_ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(accelerator.device)

    # Pass all components to accelerate
    model, optimizer, train_dataloader, val_dataloader, scheduler, criterion = accelerator.prepare(
        model, optimizer, train_dataloader, val_dataloader, scheduler, criterion
    )

    # --- W&B Visual Sample Setup ---
    fixed_deg, fixed_clean = None, None
    if accelerator.is_main_process:
        # Just grab the first rain and first snow image in the validation set
        rain_idx = next(i for i, name in enumerate(val_dataset.degraded_images) if 'rain' in name)
        snow_idx = next(i for i, name in enumerate(val_dataset.degraded_images) if 'snow' in name)

        rain_deg, rain_clean = val_dataset[rain_idx]
        snow_deg, snow_clean = val_dataset[snow_idx]

        fixed_deg = torch.stack([rain_deg, snow_deg]).to(accelerator.device)
        fixed_clean = torch.stack([rain_clean, snow_clean]).to(accelerator.device)

    best_val_psnr = 0.0

    # Flattened the loops: No outer tqdm wrapper, just cleanly printed epochs.
    for epoch in range(args.epochs):

        if accelerator.is_main_process:
            tqdm.write(f"\n--- Epoch [{epoch + 1}/{args.epochs}] ---")

        # --- TRAINING PHASE ---
        model.train()
        train_psnr_metric.reset()
        epoch_train_losses = []

        # dynamic_ncols fixes the terminal wrapping issue
        train_pbar = tqdm(train_dataloader, desc="Training", leave=False, disable=not accelerator.is_local_main_process,
                          file=sys.stdout, dynamic_ncols=True)

        for degraded, clean in train_pbar:
            if random.random() < 0.5:
                degraded, clean = apply_dense_mixup_cutmix(degraded, clean)

            optimizer.zero_grad()
            output = model(degraded)
            loss, _ = criterion(output, clean)

            accelerator.backward(loss)

            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(model.parameters(), 0.01)

            optimizer.step()

            output_clamped = torch.clamp(output, 0.0, 1.0)
            train_psnr_metric.update(output_clamped, clean)
            epoch_train_losses.append(loss.item())

            if accelerator.is_local_main_process:
                train_pbar.set_postfix({'Loss': f"{loss.item():.4f}"})

        scheduler.step()
        accelerator.wait_for_everyone()

        current_train_psnr = train_psnr_metric.compute().item()
        avg_train_loss = np.mean(epoch_train_losses)

        # --- VALIDATION PHASE ---
        model.eval()
        val_psnr_metric.reset()
        val_ssim_metric.reset()
        epoch_val_losses = []

        val_pbar = tqdm(val_dataloader, desc="Validation", leave=False, disable=not accelerator.is_local_main_process,
                        file=sys.stdout, dynamic_ncols=True)

        with torch.no_grad():
            for degraded, clean in val_pbar:
                output = model(degraded)
                loss, _ = criterion(output, clean)

                output_clamped = torch.clamp(output, 0.0, 1.0)
                val_psnr_metric.update(output_clamped, clean)
                val_ssim_metric.update(output_clamped, clean)
                epoch_val_losses.append(loss.item())

        accelerator.wait_for_everyone()

        current_val_psnr = val_psnr_metric.compute().item()
        current_val_ssim = val_ssim_metric.compute().item()
        avg_val_loss = np.mean(epoch_val_losses)

        # --- LOGGING & SAVING ---
        if accelerator.is_main_process:

            # W&B Visuals: Generate grid using the fixed rain/snow samples
            if fixed_deg is not None:
                with torch.no_grad():
                    fixed_pred = torch.clamp(model(fixed_deg), 0.0, 1.0)

                    # Top Row: Rain (In | Pred | Clean)
                    row_rain = torch.cat([fixed_deg[0], fixed_pred[0], fixed_clean[0]], dim=2)
                    # Bottom Row: Snow (In | Pred | Clean)
                    row_snow = torch.cat([fixed_deg[1], fixed_pred[1], fixed_clean[1]], dim=2)

                    stitched_grid = torch.cat([row_rain, row_snow], dim=1)  # Stack vertically

                accelerator.log({
                    "Train/Loss": avg_train_loss,
                    "Train/PSNR": current_train_psnr,
                    "Val/Loss": avg_val_loss,
                    "Val/PSNR": current_val_psnr,
                    "Val/SSIM": current_val_ssim,
                    "Visuals/Restoration": wandb.Image(stitched_grid,
                                                       caption="Top: Rain, Bottom: Snow | Left: Degraded, Mid: Restored, Right: Clean")
                }, step=epoch)
            else:
                accelerator.log({
                    "Train/Loss": avg_train_loss,
                    "Train/PSNR": current_train_psnr,
                    "Val/Loss": avg_val_loss,
                    "Val/PSNR": current_val_psnr,
                    "Val/SSIM": current_val_ssim,
                }, step=epoch)

            tqdm.write(
                f"Train Loss: {avg_train_loss:.4f} | Val PSNR: {current_val_psnr:.2f} | Val SSIM: {current_val_ssim:.4f}")

            # Save logic now strictly evaluates generalizability via Validation PSNR
            unwrapped_model = accelerator.unwrap_model(model)

            # Save latest model every epoch
            # latest_model_path = os.path.join(args.save_dir, "latest_model.pth")
            # torch.save(unwrapped_model.state_dict(), latest_model_path)

            # Save best model if PSNR improves
            if current_val_psnr > best_val_psnr:
                best_val_psnr = current_val_psnr
                best_model_path = os.path.join(args.save_dir, "best_model.pth")
                torch.save(unwrapped_model.state_dict(), best_model_path)
                tqdm.write(f"New Best Model Saved (Val PSNR: {best_val_psnr:.2f})")

    # --- FINAL W&B UPLOAD ---
    # We wait for everyone to finish the epoch loop, then upload the local files to the cloud.
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        tqdm.write("\nUploading final models to Weights & Biases...")

        # wandb.save() securely pushes the files from the local directory to the run's cloud storage
        if os.path.exists(os.path.join(args.save_dir, "best_model.pth")):
            wandb.save(os.path.join(args.save_dir, "best_model.pth"), base_path=args.save_dir)
            tqdm.write("best_model.pth uploaded.")

        # if os.path.exists(os.path.join(args.save_dir, "latest_model.pth")):
        #     wandb.save(os.path.join(args.save_dir, "latest_model.pth"), base_path=args.save_dir)
        #     tqdm.write("latest_model.pth uploaded.")

    accelerator.end_training()


if __name__ == "__main__":
    main()
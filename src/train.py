import argparse
import os
import sys
import random
import time
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


class TrainingState:
    """Helper class to track training progress for checkpointing."""
    def __init__(self):
        self.epoch = 0
        self.best_val_psnr = 0.0

    def state_dict(self):
        return {"epoch": self.epoch, "best_val_psnr": self.best_val_psnr}

    def load_state_dict(self, state_dict):
        self.epoch = state_dict.get("epoch", 0)
        self.best_val_psnr = state_dict.get("best_val_psnr", 0.0)


def main():
    parser = argparse.ArgumentParser(description="Train PromptIR with Accelerate & Validation")
    parser.add_argument("--data_dir", type=str, default="dataset/train")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--min_lr", type=float, default=1e-6)
    parser.add_argument("--save_dir", type=str, default="checkpoints")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None, help="Path to checkpoint directory to resume from")
    parser.add_argument("--checkpoint_interval", type=int, default=1, help="Save checkpoint every N epochs")
    parser.add_argument("--wandb_id", type=str, default=None, help="W&B run ID to resume")
    parser.add_argument("--max_hours", type=float, default=11.7, help="Hard stop training after N hours")
    args = parser.parse_args()

    start_time = time.time()
    accelerator = Accelerator(log_with="wandb", mixed_precision="fp16")
    set_seed(42 + accelerator.process_index)

    if args.resume_from_checkpoint:
        # If resuming, we want to stay in the same folder
        # Expected path: checkpoints/run_name/checkpoint_last
        checkpoint_path = os.path.normpath(args.resume_from_checkpoint)
        args.save_dir = os.path.dirname(checkpoint_path)

        if args.wandb_id is None:
            id_file = os.path.join(args.save_dir, "wandb_id.txt")
            if os.path.exists(id_file):
                with open(id_file, "r") as f:
                    args.wandb_id = f.read().strip()
                accelerator.print(f"Found W&B ID in {id_file}: {args.wandb_id}")
            else:
                print(f"Warning: No wandb_id provided and no wandb_id.txt found in {args.save_dir}. W&B logging may be inconsistent.")

    if accelerator.is_main_process:
        init_kwargs = {}
        if args.wandb_id:
            init_kwargs["wandb"] = {"id": args.wandb_id, "resume": "allow"}
        
        accelerator.init_trackers("PromptIR-Restoration", config=vars(args), init_kwargs=init_kwargs)
        
        # If not resuming, create a new run folder
        if not args.resume_from_checkpoint:
            run_name = wandb.run.name if wandb.run.name else "default_run"
            args.save_dir = os.path.join(args.save_dir, run_name)
            os.makedirs(args.save_dir, exist_ok=True)
            
            # Save W&B ID for future resumes
            with open(os.path.join(args.save_dir, "wandb_id.txt"), "w") as f:
                f.write(wandb.run.id)
    
    # Ensure all processes have the updated save_dir
    accelerator.wait_for_everyone()

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

    # Register objects for checkpointing
    training_state = TrainingState()
    accelerator.register_for_checkpointing(training_state)
    accelerator.register_for_checkpointing(scheduler)

    # Pass all components to accelerate
    model, optimizer, train_dataloader, val_dataloader, scheduler, criterion = accelerator.prepare(
        model, optimizer, train_dataloader, val_dataloader, scheduler, criterion
    )

    # Resume from checkpoint if provided
    if args.resume_from_checkpoint:
        accelerator.print(f"Resuming from checkpoint: {args.resume_from_checkpoint}")
        accelerator.load_state(args.resume_from_checkpoint)
        # training_state.epoch and training_state.best_val_psnr are now updated

    # --- W&B Visual Sample Setup ---
    fixed_deg, fixed_clean = None, None
    if accelerator.is_main_process:
        try:
            # Just grab the first rain and first snow image in the validation set
            rain_idx = next(i for i, name in enumerate(val_dataset.degraded_images) if 'rain' in name)
            snow_idx = next(i for i, name in enumerate(val_dataset.degraded_images) if 'snow' in name)

            rain_deg, rain_clean = val_dataset[rain_idx]
            snow_deg, snow_clean = val_dataset[snow_idx]

            fixed_deg = torch.stack([rain_deg, snow_deg]).to(accelerator.device)
            fixed_clean = torch.stack([rain_clean, snow_clean]).to(accelerator.device)
        except StopIteration:
            tqdm.write("Warning: Could not find both rain and snow images in validation set for visualization.")
            fixed_deg, fixed_clean = None, None

    try:
        # Flattened the loops: No outer tqdm wrapper, just cleanly printed epochs.
        for epoch in range(training_state.epoch, args.epochs):
            training_state.epoch = epoch

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
                loss, loss_dict = criterion(output, clean)

                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), 0.01)

                optimizer.step()

                output_clamped = torch.clamp(output, 0.0, 1.0)
                train_psnr_metric.update(output_clamped, clean)
                epoch_train_losses.append(loss.item())

                if accelerator.is_local_main_process:
                    train_pbar.set_postfix({
                        'Loss': f"{loss.item():.3f}",
                        'C': f"{loss_dict['loss_char']:.3f}",
                        'F': f"{loss_dict['loss_fft']:.3f}"
                    })

            scheduler.step()
            accelerator.wait_for_everyone()

            current_train_psnr = train_psnr_metric.compute().item()
            avg_train_loss = np.mean(epoch_train_losses)

            # --- VALIDATION PHASE ---
            model.eval()
            val_psnr_metric.reset()
            val_ssim_metric.reset()
            epoch_val_losses = []
            epoch_val_char = []
            epoch_val_fft = []

            val_pbar = tqdm(val_dataloader, desc="Validation", leave=False, disable=not accelerator.is_local_main_process,
                            file=sys.stdout, dynamic_ncols=True)

            with torch.no_grad():
                for degraded, clean in val_pbar:
                    output = model(degraded)
                    loss, loss_dict = criterion(output, clean)

                    output_clamped = torch.clamp(output, 0.0, 1.0)
                    val_psnr_metric.update(output_clamped, clean)
                    val_ssim_metric.update(output_clamped, clean)
                    epoch_val_losses.append(loss.item())
                    epoch_val_char.append(loss_dict['loss_char'])
                    epoch_val_fft.append(loss_dict['loss_fft'])

            accelerator.wait_for_everyone()

            current_val_psnr = val_psnr_metric.compute().item()
            current_val_ssim = val_ssim_metric.compute().item()
            avg_val_loss = np.mean(epoch_val_losses)
            avg_val_char = np.mean(epoch_val_char)
            avg_val_fft = np.mean(epoch_val_fft)

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
                        "Val/Char_Loss": avg_val_char,
                        "Val/FFT_Loss": avg_val_fft,
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
                        "Val/Char_Loss": avg_val_char,
                        "Val/FFT_Loss": avg_val_fft,
                        "Val/PSNR": current_val_psnr,
                        "Val/SSIM": current_val_ssim,
                    }, step=epoch)

                tqdm.write(
                    f"Train Loss: {avg_train_loss:.4f} | Val PSNR: {current_val_psnr:.2f} | Val SSIM: {current_val_ssim:.4f}")

                # Save logic now strictly evaluates generalizability via Validation PSNR
                unwrapped_model = accelerator.unwrap_model(model)

                # Save best model if PSNR improves
                if current_val_psnr > training_state.best_val_psnr:
                    training_state.best_val_psnr = current_val_psnr
                    best_model_path = os.path.join(args.save_dir, "best_model.pth")
                    torch.save(unwrapped_model.state_dict(), best_model_path)
                    tqdm.write(f"New Best Model Saved (Val PSNR: {training_state.best_val_psnr:.2f})")

            # Increment epoch for next possible resume
            training_state.epoch = epoch + 1
            
            # Save full training state for resuming
            if (epoch + 1) % args.checkpoint_interval == 0:
                checkpoint_dir = os.path.join(args.save_dir, "checkpoint_last")
                accelerator.save_state(checkpoint_dir)
                if accelerator.is_main_process:
                    tqdm.write(f"Checkpoint saved to {checkpoint_dir}")

            # Hard stop if time limit reached
            elapsed_hours = (time.time() - start_time) / 3600
            if elapsed_hours >= args.max_hours:
                accelerator.print(f"Time limit reached ({elapsed_hours:.2f} hours). Stopping training to allow upload.")
                break

    finally:
        # --- FINAL W&B UPLOAD ---
        # We wait for everyone to finish, then upload the local files to the cloud.
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            tqdm.write("\nUploading final results to Weights & Biases...")

            # Upload Best Model
            if os.path.exists(os.path.join(args.save_dir, "best_model.pth")):
                wandb.save(os.path.join(args.save_dir, "best_model.pth"), base_path=args.save_dir)
                tqdm.write("best_model.pth uploaded.")

            # Upload the final training state (checkpoint_last) for archiving
            checkpoint_dir = os.path.join(args.save_dir, "checkpoint_last")
            if os.path.exists(checkpoint_dir):
                # wandb.save(glob) will preserve the directory structure if base_path is set correctly
                wandb.save(os.path.join(checkpoint_dir, "*"), base_path=args.save_dir)
                tqdm.write("Final checkpoint_last uploaded to W&B.")

        accelerator.end_training()


if __name__ == "__main__":
    main()
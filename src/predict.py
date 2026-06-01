import argparse
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from model.promptir import PromptIR
from dataset import RestorationDataset


def apply_ensemble_transform(img_tensor, transform_idx):
    """Applies one of the 8 geometric transformations."""
    if transform_idx == 0:
        return img_tensor
    elif transform_idx == 1:
        return torch.rot90(img_tensor, 1, [2, 3])
    elif transform_idx == 2:
        return torch.rot90(img_tensor, 2, [2, 3])
    elif transform_idx == 3:
        return torch.rot90(img_tensor, 3, [2, 3])
    elif transform_idx == 4:
        return torch.flip(img_tensor, [3])
    elif transform_idx == 5:
        return torch.flip(torch.rot90(img_tensor, 1, [2, 3]), [3])
    elif transform_idx == 6:
        return torch.flip(torch.rot90(img_tensor, 2, [2, 3]), [3])
    elif transform_idx == 7:
        return torch.flip(torch.rot90(img_tensor, 3, [2, 3]), [3])
    return img_tensor


def invert_ensemble_transform(img_tensor, transform_idx):
    """Inverts the applied geometric transformation to realign predictions."""
    if transform_idx == 0:
        return img_tensor
    elif transform_idx == 1:
        return torch.rot90(img_tensor, -1, [2, 3])
    elif transform_idx == 2:
        return torch.rot90(img_tensor, -2, [2, 3])
    elif transform_idx == 3:
        return torch.rot90(img_tensor, -3, [2, 3])
    elif transform_idx == 4:
        return torch.flip(img_tensor, [3])
    elif transform_idx == 5:
        # Inverse of F(R1(I)) is R-1(F(I'))
        return torch.rot90(torch.flip(img_tensor, [3]), -1, [2, 3])
    elif transform_idx == 6:
        # Inverse of F(R2(I)) is R-2(F(I'))
        return torch.rot90(torch.flip(img_tensor, [3]), -2, [2, 3])
    elif transform_idx == 7:
        # Inverse of F(R3(I)) is R-3(F(I'))
        return torch.rot90(torch.flip(img_tensor, [3]), -3, [2, 3])
    return img_tensor


def predict_single(model, img):
    """Performs inference on a single image, ensuring it's padded to multiple of 8."""
    _, _, h, w = img.shape
    pad_h = (8 - h % 8) % 8
    pad_w = (8 - w % 8) % 8
    
    if pad_h > 0 or pad_w > 0:
        img = F.pad(img, (0, pad_w, 0, pad_h), mode='reflect')
    
    with torch.no_grad():
        out = model(img)
    
    return out[:, :, :h, :w]


def tiled_predict(model, img, tile_size=256, tile_overlap=32):
    """Performs inference using overlapped tiles with boundary-safe blending."""
    b, c, h, w = img.shape
    if h <= tile_size and w <= tile_size:
        return predict_single(model, img)

    stride = tile_size - tile_overlap
    
    # Pad image to be multiple of stride + overlap
    pad_h = (stride - (h - tile_overlap) % stride) % stride
    pad_w = (stride - (w - tile_overlap) % stride) % stride
    
    img_padded = F.pad(img, (0, pad_w, 0, pad_h), mode='reflect')
    _, _, hp, wp = img_padded.shape
    
    output = torch.zeros_like(img_padded)
    weight = torch.zeros_like(img_padded)
    
    # Base 1D window
    win_1d = torch.ones(tile_size, device=img.device)
    if tile_overlap > 0:
        ramp = torch.linspace(0, 1, tile_overlap, device=img.device)
        win_1d[:tile_overlap] = ramp
        win_1d[-tile_overlap:] = ramp.flip(0)
    
    base_window = win_1d.view(1, 1, tile_size, 1) * win_1d.view(1, 1, 1, tile_size)

    for y in range(0, hp - tile_overlap, stride):
        for x in range(0, wp - tile_overlap, stride):
            if y + tile_size > hp or x + tile_size > wp:
                continue
                
            tile = img_padded[:, :, y:y+tile_size, x:x+tile_size]
            with torch.no_grad():
                tile_pred = model(tile)
            
            # Adjust window for image boundaries to avoid zeroing out edges
            tile_window = base_window.clone()
            if y == 0: tile_window[:, :, :tile_overlap, :] = 1.0
            if x == 0: tile_window[:, :, :, :tile_overlap] = 1.0
            if y + tile_size >= hp: tile_window[:, :, -tile_overlap:, :] = 1.0
            if x + tile_size >= wp: tile_window[:, :, :, -tile_overlap:] = 1.0
            
            output[:, :, y:y+tile_size, x:x+tile_size] += tile_pred * tile_window
            weight[:, :, y:y+tile_size, x:x+tile_size] += tile_window
            
    output = output / (weight + 1e-8)
    return output[:, :, :h, :w]


def main():
    parser = argparse.ArgumentParser(description="Restoration Inference with 8-Fold Ensemble")
    parser.add_argument("--data_dir", type=str, default="dataset/test", help="Path to test data")
    parser.add_argument("--weights", type=str, required=True, help="Path to best model weights")
    parser.add_argument("--output", type=str, default="pred.npz", help="Output numpy archive")
    parser.add_argument("--use_tiling", action="store_true", help="Use tiled inference for large images")
    parser.add_argument("--tile_size", type=int, default=256, help="Tile size for inference")
    parser.add_argument("--tile_overlap", type=int, default=32, help="Overlap between tiles")
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    model = PromptIR().to(device)

    print(f"Loading weights from {args.weights}...")
    state_dict = torch.load(args.weights, map_location=device)

    # Clean state dict
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            new_state_dict[k[7:]] = v
        elif k.startswith('_orig_mod.'):
            new_state_dict[k[10:]] = v
        else:
            new_state_dict[k] = v
    
    model.load_state_dict(new_state_dict)
    model.eval()

    dataset = RestorationDataset(root_dir=args.data_dir, mode='test')
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    images_dict = {}

    print(f"Restoring {len(dataset)} images with 8-fold ensemble...")
    with torch.no_grad():
        for degraded, filename in tqdm(dataloader, desc="Inference", dynamic_ncols=True):
            degraded = degraded.to(device)
            
            ensemble_preds = []
            for i in range(8):
                transformed_input = apply_ensemble_transform(degraded, i)
                
                if args.use_tiling:
                    pred = tiled_predict(model, transformed_input, tile_size=args.tile_size, tile_overlap=args.tile_overlap)
                else:
                    pred = predict_single(model, transformed_input)
                
                realigned_pred = invert_ensemble_transform(pred, i)
                ensemble_preds.append(realigned_pred)

            # Average 8-fold predictions
            avg_pred = torch.stack(ensemble_preds).mean(dim=0)

            # Post-process to uint8
            avg_pred = torch.clamp(avg_pred, 0.0, 1.0)
            avg_pred_uint8 = (avg_pred.squeeze(0).cpu().numpy() * 255.0).round().astype(np.uint8)

            images_dict[filename[0]] = avg_pred_uint8

    print(f"Saving to {args.output}...")
    np.savez(args.output, **images_dict)
    print("Done!")


if __name__ == "__main__":
    main()

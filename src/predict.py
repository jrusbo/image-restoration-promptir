import argparse
import numpy as np
import torch
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
        # Original: flip(rot90(img, 1), [3])
        # Invert: rot90(flip(img, [3]), -1)
        return torch.rot90(torch.flip(img_tensor, [3]), -1, [2, 3])
    elif transform_idx == 6:
        # Original: flip(rot90(img, 2), [3])
        # Invert: rot90(flip(img, [3]), -2)
        return torch.rot90(torch.flip(img_tensor, [3]), -2, [2, 3])
    elif transform_idx == 7:
        # Original: flip(rot90(img, 3), [3])
        # Invert: rot90(flip(img, [3]), -3)
        return torch.rot90(torch.flip(img_tensor, [3]), -3, [2, 3])
    return img_tensor


def main():
    parser = argparse.ArgumentParser(description="Inference with Native 8-fold Self-Ensemble")
    parser.add_argument("--data_dir", type=str, default="dataset/test", help="Path to test data")
    parser.add_argument("--weights", type=str, required=True, help="Path to best model weights")
    parser.add_argument("--output", type=str, default="pred.npz", help="Output numpy archive")
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    model = PromptIR().to(device)

    print(f"Loading weights from {args.weights}...")
    state_dict = torch.load(args.weights, map_location=device)

    # Strip DDP 'module.' prefix if weights were saved directly from unwrapped DDP model
    if list(state_dict.keys())[0].startswith('module.'):
        state_dict = {k[7:]: v for k, v in state_dict.items()}

    model.load_state_dict(state_dict)
    model.eval()

    dataset = RestorationDataset(root_dir=args.data_dir, mode='test')
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    images_dict = {}

    print("Commencing Test-Time Inference with Native 8-Fold Ensemble...")
    with torch.no_grad():
        for degraded, filename in tqdm(dataloader, desc="Restoring Images", dynamic_ncols=True):
            degraded = degraded.to(device)
            ensemble_preds = []

            # Perform the 8-fold geometric self-ensemble manually
            for i in range(8):
                transformed_input = apply_ensemble_transform(degraded, i)
                pred = model(transformed_input)
                realigned_pred = invert_ensemble_transform(pred, i)
                ensemble_preds.append(realigned_pred)

            # Average the 8 predictions to suppress artifacts
            avg_pred = torch.stack(ensemble_preds).mean(dim=0)

            # Post-process to uint8 array
            avg_pred = torch.clamp(avg_pred, 0.0, 1.0)
            avg_pred_uint8 = (avg_pred.squeeze(0).cpu().numpy() * 255.0).astype(np.uint8)

            # filename is returned as a tuple, extract index 0
            images_dict[filename[0]] = avg_pred_uint8

    print(f"Packing {len(images_dict)} images into Numpy Archive...")
    np.savez(args.output, **images_dict)
    print(f"Successfully saved submission file to: {args.output}")


if __name__ == "__main__":
    main()
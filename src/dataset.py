import os
import random
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision.transforms import v2
from torchvision import tv_tensors


class RestorationDataset(Dataset):
    def __init__(self, root_dir, mode='train', val_split=0.1, seed=42):
        """
        mode: 'train', 'val', or 'test'
        val_split: Percentage of data to reserve for validation (default 10%)
        """
        self.root_dir = root_dir
        self.mode = mode

        if self.mode in ['train', 'val']:
            self.degraded_dir = os.path.join(root_dir, 'degraded')
            self.clean_dir = os.path.join(root_dir, 'clean')

            # Fetch all images
            all_images = sorted(os.listdir(self.degraded_dir))

            # Separate by degradation type to ensure a perfectly balanced 90/10 split
            rain_images = [f for f in all_images if 'rain' in f]
            snow_images = [f for f in all_images if 'snow' in f]

            # Shuffle using a fixed seed so the split is identical across epochs and GPUs
            rng = random.Random(seed)
            rng.shuffle(rain_images)
            rng.shuffle(snow_images)

            split_idx_rain = int(len(rain_images) * (1 - val_split))
            split_idx_snow = int(len(snow_images) * (1 - val_split))

            if self.mode == 'train':
                self.degraded_images = rain_images[:split_idx_rain] + snow_images[:split_idx_snow]
            else:  # val
                self.degraded_images = rain_images[split_idx_rain:] + snow_images[split_idx_snow:]

            # Geometric transforms only defined for training
            self.geom_transforms = v2.Compose([
                v2.RandomHorizontalFlip(p=0.5),
                v2.RandomVerticalFlip(p=0.5),
                v2.RandomChoice([
                    v2.RandomRotation((0, 0)),
                    v2.RandomRotation((90, 90)),
                    v2.RandomRotation((180, 180)),
                    v2.RandomRotation((270, 270))
                ]),
            ])

        else:  # test
            self.degraded_dir = os.path.join(root_dir, 'degraded')
            self.degraded_images = sorted(os.listdir(self.degraded_dir))

        # Standard tensor conversion
        self.to_tensor = v2.Compose([
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True)
        ])

    def __len__(self):
        return len(self.degraded_images)

    def __getitem__(self, idx):
        degraded_name = self.degraded_images[idx]
        degraded_path = os.path.join(self.degraded_dir, degraded_name)

        # Convert to RGB to ensure 3 channels
        degraded_img = Image.open(degraded_path).convert('RGB')

        if self.mode in ['train', 'val']:
            # Map degraded filename to clean filename
            if 'rain' in degraded_name:
                clean_name = degraded_name.replace('rain-', 'rain_clean-')
            elif 'snow' in degraded_name:
                clean_name = degraded_name.replace('snow-', 'snow_clean-')
            else:
                raise ValueError(f"Unknown degradation type in filename: {degraded_name}")

            clean_path = os.path.join(self.clean_dir, clean_name)
            clean_img = Image.open(clean_path).convert('RGB')

            degraded_tensor = tv_tensors.Image(degraded_img)
            clean_tensor = tv_tensors.Image(clean_img)

            # Only apply stochastic augmentations during 'train' mode
            if self.mode == 'train':
                degraded_tensor, clean_tensor = self.geom_transforms(degraded_tensor, clean_tensor)

            return self.to_tensor(degraded_tensor), self.to_tensor(clean_tensor)
        else:
            return self.to_tensor(degraded_img), degraded_name
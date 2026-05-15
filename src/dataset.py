import os
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision.transforms import v2
from torchvision import tv_tensors


class RestorationDataset(Dataset):
    def __init__(self, root_dir, is_train=True):
        self.root_dir = root_dir
        self.is_train = is_train
        self.degraded_dir = os.path.join(root_dir, 'degraded')

        # Ensure consistent ordering across GPUs
        self.degraded_images = sorted(os.listdir(self.degraded_dir))

        if self.is_train:
            self.clean_dir = os.path.join(root_dir, 'clean')

            # v2 Transforms: Handles (image, mask/target) pairs simultaneously
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
            self.to_tensor = v2.Compose([
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=True)
            ])
        else:
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

        if self.is_train:
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

            # Apply identical geometric transformations to both images
            degraded_tensor, clean_tensor = self.geom_transforms(degraded_tensor, clean_tensor)

            return self.to_tensor(degraded_tensor), self.to_tensor(clean_tensor)
        else:
            return self.to_tensor(degraded_img), degraded_name
import os
import random
from PIL import Image
from typing import Tuple, Union, List, Optional
import torch
from torch.utils.data import Dataset
from torchvision.transforms import v2
from torchvision import tv_tensors


class RestorationDataset(Dataset):
    """Dataset for image restoration tasks, supporting rain and snow degradations.

    Attributes:
        root_dir (str): Root directory of the dataset.
        mode (str): Dataset mode ('train', 'val', or 'test').
        degraded_dir (str): Directory containing degraded images.
        clean_dir (Optional[str]): Directory containing clean images.
        degraded_images (List[str]): List of filenames for degraded images.
        geom_transforms (Optional[v2.Compose]): Geometric augmentations for training.
        to_tensor (v2.Compose): Transform to convert images to tensors.
    """

    def __init__(
        self,
        root_dir: str,
        mode: str = 'train',
        val_split: float = 0.1,
        seed: int = 42
    ):
        """Initializes the RestorationDataset.

        Args:
            root_dir: Root directory of the dataset.
            mode: 'train', 'val', or 'test'.
            val_split: Percentage of data to reserve for validation.
            seed: Random seed for reproducible splits.
        """
        self.root_dir = root_dir
        self.mode = mode

        if self.mode in ['train', 'val']:
            self.degraded_dir = os.path.join(root_dir, 'degraded')
            self.clean_dir = os.path.join(root_dir, 'clean')

            all_images = sorted(os.listdir(self.degraded_dir))

            # Separate by degradation type to ensure a perfectly balanced split
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
            self.clean_dir = None

        self.to_tensor = v2.Compose([
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True)
        ])

    def __len__(self) -> int:
        """Returns the number of images in the dataset.

        Returns:
            The total number of samples.
        """
        return len(self.degraded_images)

    def __getitem__(self, idx: int) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, str]]:
        """Retrieves a sample from the dataset.

        Args:
            idx: Index of the sample to retrieve.

        Returns:
            If mode is 'train' or 'val': A tuple of (degraded_tensor, clean_tensor).
            If mode is 'test': A tuple of (degraded_tensor, filename).
        """
        degraded_name = self.degraded_images[idx]
        degraded_path = os.path.join(self.degraded_dir, degraded_name)

        degraded_img = Image.open(degraded_path).convert('RGB')

        if self.mode in ['train', 'val']:
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

            if self.mode == 'train':
                degraded_tensor, clean_tensor = self.geom_transforms(degraded_tensor, clean_tensor)

            return self.to_tensor(degraded_tensor), self.to_tensor(clean_tensor)
        else:
            return self.to_tensor(degraded_img), degraded_name

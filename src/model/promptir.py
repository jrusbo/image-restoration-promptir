import torch
import torch.nn as nn
from typing import List
from .blocks import TransformerBlock, PGM, DGPB


class Downsample(nn.Module):
    """Downsampling module using PixelUnshuffle.

    Attributes:
        body (nn.Sequential): Sequential layers for downsampling.
    """

    def __init__(self, dim: int):
        """Initializes the Downsample module.

        Args:
            dim: Number of input channels.
        """
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(dim, dim // 2, kernel_size=3, stride=1, padding=1, bias=False),
            nn.PixelUnshuffle(2)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Downsamples the input tensor.

        Args:
            x: Input tensor.

        Returns:
            Downsampled tensor.
        """
        return self.body(x)


class Upsample(nn.Module):
    """Upsampling module using PixelShuffle.

    Attributes:
        body (nn.Sequential): Sequential layers for upsampling.
    """

    def __init__(self, dim: int):
        """Initializes the Upsample module.

        Args:
            dim: Number of input channels.
        """
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(dim, dim * 2, kernel_size=3, stride=1, padding=1, bias=False),
            nn.PixelShuffle(2)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Upsamples the input tensor.

        Args:
            x: Input tensor.

        Returns:
            Upsampled tensor.
        """
        return self.body(x)


class PromptIR(nn.Module):
    """PromptIR model for image restoration.

    Attributes:
        patch_embed (nn.Conv2d): Initial patch embedding layer.
        encoder_level1 (nn.Sequential): First level of the encoder.
        down1_2 (Downsample): Downsampling from level 1 to 2.
        encoder_level2 (nn.Sequential): Second level of the encoder.
        down2_3 (Downsample): Downsampling from level 2 to 3.
        encoder_level3 (nn.Sequential): Third level of the encoder.
        down3_4 (Downsample): Downsampling from level 3 to 4.
        encoder_level4 (nn.Sequential): Fourth level (bottleneck) of the encoder.
        pgm (PGM): Prompt Generation Module.
        dgpb1 (DGPB): Skip connection modulation for level 1.
        dgpb2 (DGPB): Skip connection modulation for level 2.
        dgpb3 (DGPB): Skip connection modulation for level 3.
        up4_3 (Upsample): Upsampling from level 4 to 3.
        reduce_chan_level3 (nn.Conv2d): Channel reduction for level 3 concatenation.
        decoder_level3 (nn.Sequential): Third level of the decoder.
        up3_2 (Upsample): Upsampling from level 3 to 2.
        reduce_chan_level2 (nn.Conv2d): Channel reduction for level 2 concatenation.
        decoder_level2 (nn.Sequential): Second level of the decoder.
        up2_1 (Upsample): Upsampling from level 2 to 1.
        reduce_chan_level1 (nn.Conv2d): Channel reduction for level 1 concatenation.
        decoder_level1 (nn.Sequential): First level of the decoder.
        output (nn.Conv2d): Final output projection layer.
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        dim: int = 48,
        num_blocks: List[int] = [4, 6, 6, 8]
    ):
        """Initializes the PromptIR model.

        Args:
            in_channels: Number of input image channels.
            out_channels: Number of output image channels.
            dim: Base number of feature channels.
            num_blocks: List containing number of transformer blocks at each level.
        """
        super().__init__()

        self.patch_embed = nn.Conv2d(in_channels, dim, kernel_size=3, padding=1)

        self.encoder_level1 = nn.Sequential(*[TransformerBlock(dim) for _ in range(num_blocks[0])])
        self.down1_2 = Downsample(dim)

        self.encoder_level2 = nn.Sequential(*[TransformerBlock(dim * 2) for _ in range(num_blocks[1])])
        self.down2_3 = Downsample(dim * 2)

        self.encoder_level3 = nn.Sequential(*[TransformerBlock(dim * 4) for _ in range(num_blocks[2])])
        self.down3_4 = Downsample(dim * 4)

        self.encoder_level4 = nn.Sequential(*[TransformerBlock(dim * 8) for _ in range(num_blocks[3])])

        self.pgm = PGM(in_channels=dim * 8, num_prompts=5, prompt_dim=64)

        self.dgpb1 = DGPB(dim)
        self.dgpb2 = DGPB(dim * 2)
        self.dgpb3 = DGPB(dim * 4)

        self.up4_3 = Upsample(dim * 8)
        self.reduce_chan_level3 = nn.Conv2d(dim * 8, dim * 4, kernel_size=1, bias=False)
        self.decoder_level3 = nn.Sequential(*[TransformerBlock(dim * 4) for _ in range(num_blocks[2])])

        self.up3_2 = Upsample(dim * 4)
        self.reduce_chan_level2 = nn.Conv2d(dim * 4, dim * 2, kernel_size=1, bias=False)
        self.decoder_level2 = nn.Sequential(*[TransformerBlock(dim * 2) for _ in range(num_blocks[1])])

        self.up2_1 = Upsample(dim * 2)
        self.reduce_chan_level1 = nn.Conv2d(dim * 2, dim, kernel_size=1, bias=False)
        self.decoder_level1 = nn.Sequential(*[TransformerBlock(dim) for _ in range(num_blocks[0])])

        self.output = nn.Conv2d(dim, out_channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Restores the input image.

        Args:
            x: Input degraded image tensor of shape (B, C, H, W).

        Returns:
            Restored image tensor of shape (B, C, H, W).
        """
        inp = x

        x = self.patch_embed(x)

        out_enc1 = self.encoder_level1(x)
        x = self.down1_2(out_enc1)

        out_enc2 = self.encoder_level2(x)
        x = self.down2_3(out_enc2)

        out_enc3 = self.encoder_level3(x)
        x = self.down3_4(out_enc3)

        out_enc4 = self.encoder_level4(x)

        prompt = self.pgm(out_enc4)

        x = self.up4_3(out_enc4)
        skip3 = self.dgpb3(out_enc3, prompt)
        x = torch.cat([x, skip3], dim=1)
        x = self.reduce_chan_level3(x)
        x = self.decoder_level3(x)

        x = self.up3_2(x)
        skip2 = self.dgpb2(out_enc2, prompt)
        x = torch.cat([x, skip2], dim=1)
        x = self.reduce_chan_level2(x)
        x = self.decoder_level2(x)

        x = self.up2_1(x)
        skip1 = self.dgpb1(out_enc1, prompt)
        x = torch.cat([x, skip1], dim=1)
        x = self.reduce_chan_level1(x)
        x = self.decoder_level1(x)

        out = self.output(x) + inp
        return out

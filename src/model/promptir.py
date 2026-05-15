import torch
import torch.nn as nn
from .blocks import TransformerBlock, PGM, DGPB


class Downsample(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(dim, dim // 2, kernel_size=3, stride=1, padding=1, bias=False),
            nn.PixelUnshuffle(2)
        )

    def forward(self, x):
        return self.body(x)


class Upsample(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(dim, dim * 2, kernel_size=3, stride=1, padding=1, bias=False),
            nn.PixelShuffle(2)
        )

    def forward(self, x):
        return self.body(x)


class PromptIR(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, dim=48, num_blocks=[4, 6, 6, 8]):
        super().__init__()

        self.patch_embed = nn.Conv2d(in_channels, dim, kernel_size=3, padding=1)

        # Encoder (4 levels)
        self.encoder_level1 = nn.Sequential(*[TransformerBlock(dim) for _ in range(num_blocks[0])])
        self.down1_2 = Downsample(dim)

        self.encoder_level2 = nn.Sequential(*[TransformerBlock(dim * 2) for _ in range(num_blocks[1])])
        self.down2_3 = Downsample(dim * 2)

        self.encoder_level3 = nn.Sequential(*[TransformerBlock(dim * 4) for _ in range(num_blocks[2])])
        self.down3_4 = Downsample(dim * 4)

        self.encoder_level4 = nn.Sequential(*[TransformerBlock(dim * 8) for _ in range(num_blocks[3])])

        # Prompt Generation Module at the bottleneck
        self.pgm = PGM(in_channels=dim * 8, num_prompts=5, prompt_dim=64)

        # Degradation-Guided Perturbation Blocks in Skip Connections
        self.dgpb1 = DGPB(dim)
        self.dgpb2 = DGPB(dim * 2)
        self.dgpb3 = DGPB(dim * 4)

        # Decoder
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

    def forward(self, x):
        inp = x

        # Embed
        x = self.patch_embed(x)

        # Encoder
        out_enc1 = self.encoder_level1(x)
        x = self.down1_2(out_enc1)

        out_enc2 = self.encoder_level2(x)
        x = self.down2_3(out_enc2)

        out_enc3 = self.encoder_level3(x)
        x = self.down3_4(out_enc3)

        # Bottleneck
        out_enc4 = self.encoder_level4(x)

        # Generate Prompts
        prompt = self.pgm(out_enc4)

        # Decoder with DGPB modulated skip connections
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

        # Output + Global Residual
        out = self.output(x) + inp
        return out
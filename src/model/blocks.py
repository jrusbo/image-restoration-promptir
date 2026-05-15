import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class TLC(nn.Module):
    """Test-Time Local Converter for Global Average Pooling to mitigate train-test shift."""

    def __init__(self, kernel_size=7):
        super().__init__()
        self.kernel_size = kernel_size

    def forward(self, x):
        # Instead of standard GAP, we use a local sliding window pooling
        return F.avg_pool2d(x, kernel_size=self.kernel_size, stride=1, padding=self.kernel_size // 2)


class PGM(nn.Module):
    """Prompt Generation Module with exactly 5 learnable prompt components."""

    def __init__(self, in_channels, num_prompts=5, prompt_dim=64):
        super().__init__()
        self.num_prompts = num_prompts
        self.prompt_dim = prompt_dim

        self.tlc_pool = TLC(kernel_size=7)
        self.prompt_embeddings = nn.Parameter(torch.randn(1, num_prompts, prompt_dim))

        self.conv_reduce = nn.Sequential(
            nn.Conv2d(in_channels, prompt_dim, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(prompt_dim, num_prompts, kernel_size=1)
        )
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        b, c, h, w = x.shape
        # Extract degradation context using TLC
        context = self.tlc_pool(x)

        # Predict weights for the 5 prompt components
        weights = self.conv_reduce(context)  # [b, num_prompts, h, w]
        weights = F.adaptive_avg_pool2d(weights, (1, 1)).view(b, self.num_prompts, 1)
        weights = self.softmax(weights)

        # Generate dynamic prompt
        # prompt_embeddings: [1, num_prompts, prompt_dim] -> [b, num_prompts, prompt_dim]
        base_prompts = self.prompt_embeddings.expand(b, -1, -1)

        # Weighted sum of the 5 prompts: [b, prompt_dim]
        dynamic_prompt = torch.sum(base_prompts * weights, dim=1)
        return dynamic_prompt


class DGPB(nn.Module):
    """Degradation-Guided Perturbation Block injected into skip connections."""

    def __init__(self, dim, prompt_dim=64):
        super().__init__()
        self.prompt_proj = nn.Sequential(
            nn.Linear(prompt_dim, dim),
            nn.GELU(),
            nn.Linear(dim, dim)
        )
        self.feature_conv = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim)
        self.fusion = nn.Conv2d(dim, dim, kernel_size=1)

    def forward(self, skip_feature, prompt):
        b, c, h, w = skip_feature.shape

        # Project prompt to match feature dimension
        prompt_vec = self.prompt_proj(prompt).view(b, c, 1, 1)

        # Modulate skip feature with the prompt (Channel-wise multiplication)
        modulated = skip_feature * prompt_vec

        # Spatial refinement
        refined = self.feature_conv(modulated)
        return skip_feature + self.fusion(refined)


class MDTA(nn.Module):
    """Multi-Dconv Head Transposed Attention."""

    def __init__(self, dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=False)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=False)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=False)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)
        return self.project_out(out)


class GDFN(nn.Module):
    """Gated-Dconv Feed-Forward Network."""

    def __init__(self, dim, expansion_factor=2.66):
        super().__init__()
        hidden_dim = int(dim * expansion_factor)
        self.project_in = nn.Conv2d(dim, hidden_dim * 2, kernel_size=1, bias=False)
        self.dwconv = nn.Conv2d(hidden_dim * 2, hidden_dim * 2, kernel_size=3, stride=1, padding=1,
                                groups=hidden_dim * 2, bias=False)
        self.project_out = nn.Conv2d(hidden_dim, dim, kernel_size=1, bias=False)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        return self.project_out(x)


class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads=8):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = MDTA(dim, num_heads)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = GDFN(dim)

    def forward(self, x):
        b, c, h, w = x.shape
        # LayerNorm expects channel last
        x_norm = rearrange(x, 'b c h w -> b h w c')
        x_norm = self.norm1(x_norm)
        x_norm = rearrange(x_norm, 'b h w c -> b c h w')

        x = x + self.attn(x_norm)

        x_norm = rearrange(x, 'b c h w -> b h w c')
        x_norm = self.norm2(x_norm)
        x_norm = rearrange(x_norm, 'b h w c -> b c h w')

        x = x + self.ffn(x_norm)
        return x
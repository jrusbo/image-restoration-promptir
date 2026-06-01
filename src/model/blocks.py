import torch
import torch.nn as nn
import torch.nn.functional as F


class LayerNorm2d(nn.Module):
    """LayerNorm specifically for 4D Image Tensors (B, C, H, W).

    Attributes:
        weight (nn.Parameter): Learnable scaling factor.
        bias (nn.Parameter): Learnable shift factor.
        eps (float): Small constant for numerical stability.
    """

    def __init__(self, channels: int, eps: float = 1e-6):
        """Initializes the LayerNorm2d.

        Args:
            channels: Number of input channels.
            eps: Small constant for numerical stability.
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Calculates LayerNorm for 4D tensors.

        This implementation moves channels to the end, normalizes, and moves back,
        which is more efficient for the compiler than many small permutes.

        Args:
            x: Input tensor of shape (B, C, H, W).

        Returns:
            Normalized tensor of shape (B, C, H, W).
        """
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight.view(1, -1, 1, 1) * x + self.bias.view(1, -1, 1, 1)
        return x


class TLC(nn.Module):
    """Test-Time Local Converter for Global Average Pooling.

    Mitigates train-test shift by using local sliding window pooling instead of standard GAP.

    Attributes:
        kernel_size (int): Size of the local pooling window.
    """

    def __init__(self, kernel_size: int = 7):
        """Initializes the TLC module.

        Args:
            kernel_size: Size of the local pooling window.
        """
        super().__init__()
        self.kernel_size = kernel_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Applies local average pooling.

        Args:
            x: Input tensor.

        Returns:
            Pooled tensor.
        """
        return F.avg_pool2d(x, kernel_size=self.kernel_size, stride=1, padding=self.kernel_size // 2)


class PGM(nn.Module):
    """Prompt Generation Module.

    Generates dynamic prompts as a weighted sum of learnable prompt components.

    Attributes:
        num_prompts (int): Number of learnable prompt components.
        prompt_dim (int): Dimension of each prompt component.
        tlc_pool (TLC): Local pooling module for context extraction.
        prompt_embeddings (nn.Parameter): Learnable prompt components.
        conv_reduce (nn.Sequential): Network to predict weights for components.
        softmax (nn.Softmax): Softmax layer for weight normalization.
    """

    def __init__(self, in_channels: int, num_prompts: int = 5, prompt_dim: int = 64):
        """Initializes the PGM module.

        Args:
            in_channels: Number of input channels from the bottleneck.
            num_prompts: Number of learnable prompt components.
            prompt_dim: Dimension of each prompt component.
        """
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Generates a dynamic prompt based on the input context.

        Args:
            x: Input bottleneck features.

        Returns:
            Dynamic prompt tensor of shape (B, prompt_dim).
        """
        b, c, h, w = x.shape
        context = self.tlc_pool(x)

        weights = self.conv_reduce(context)
        weights = F.adaptive_avg_pool2d(weights, (1, 1)).view(b, self.num_prompts, 1)
        weights = self.softmax(weights)

        base_prompts = self.prompt_embeddings.expand(b, -1, -1)
        dynamic_prompt = torch.sum(base_prompts * weights, dim=1)
        return dynamic_prompt


class DGPB(nn.Module):
    """Degradation-Guided Perturbation Block.

    Injected into skip connections to modulate features based on the dynamic prompt.

    Attributes:
        prompt_proj (nn.Sequential): Network to project prompt to feature dimension.
        feature_conv (nn.Conv2d): Spatial refinement convolution.
        fusion (nn.Conv2d): Final fusion convolution.
    """

    def __init__(self, dim: int, prompt_dim: int = 64):
        """Initializes the DGPB module.

        Args:
            dim: Number of feature channels.
            prompt_dim: Dimension of the dynamic prompt.
        """
        super().__init__()
        self.prompt_proj = nn.Sequential(
            nn.Linear(prompt_dim, dim),
            nn.GELU(),
            nn.Linear(dim, dim)
        )
        self.feature_conv = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim)
        self.fusion = nn.Conv2d(dim, dim, kernel_size=1)

    def forward(self, skip_feature: torch.Tensor, prompt: torch.Tensor) -> torch.Tensor:
        """Modulates skip features using the dynamic prompt.

        Args:
            skip_feature: Feature tensor from the encoder skip connection.
            prompt: Dynamic prompt tensor.

        Returns:
            Modulated and refined feature tensor.
        """
        b, c, h, w = skip_feature.shape

        prompt_vec = self.prompt_proj(prompt).view(b, c, 1, 1)
        modulated = skip_feature * prompt_vec

        refined = self.feature_conv(modulated)
        return skip_feature + self.fusion(refined)


class MDTA(nn.Module):
    """Multi-Dconv Head Transposed Attention.

    Attributes:
        num_heads (int): Number of attention heads.
        temperature (nn.Parameter): Learnable scaling factor for attention scores.
        qkv (nn.Conv2d): Convolution to generate Q, K, V features.
        qkv_dwconv (nn.Conv2d): Depthwise convolution for spatial context.
        project_out (nn.Conv2d): Output projection convolution.
    """

    def __init__(self, dim: int, num_heads: int):
        """Initializes the MDTA module.

        Args:
            dim: Number of input channels.
            num_heads: Number of attention heads.
        """
        super().__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=False)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=False)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Applies multi-head transposed attention.

        Args:
            x: Input tensor of shape (B, C, H, W).

        Returns:
            Output tensor of shape (B, C, H, W).
        """
        b, c, h, w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)

        q = q.reshape(b, self.num_heads, -1, h * w)
        k = k.reshape(b, self.num_heads, -1, h * w)
        v = v.reshape(b, self.num_heads, -1, h * w)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)
        out = out.reshape(b, c, h, w)

        return self.project_out(out)


class GDFN(nn.Module):
    """Gated-Dconv Feed-Forward Network.

    Attributes:
        project_in (nn.Conv2d): Input projection convolution.
        dwconv (nn.Conv2d): Depthwise convolution.
        project_out (nn.Conv2d): Output projection convolution.
    """

    def __init__(self, dim: int, expansion_factor: float = 2.66):
        """Initializes the GDFN module.

        Args:
            dim: Number of input channels.
            expansion_factor: Factor to expand the hidden dimension.
        """
        super().__init__()
        hidden_dim = int(dim * expansion_factor)
        self.project_in = nn.Conv2d(dim, hidden_dim * 2, kernel_size=1, bias=False)
        self.dwconv = nn.Conv2d(hidden_dim * 2, hidden_dim * 2, kernel_size=3, stride=1, padding=1,
                                groups=hidden_dim * 2, bias=False)
        self.project_out = nn.Conv2d(hidden_dim, dim, kernel_size=1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Applies gated feed-forward network.

        Args:
            x: Input tensor.

        Returns:
            Output tensor.
        """
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        return self.project_out(x)


class TransformerBlock(nn.Module):
    """Transformer Block combining attention and feed-forward network.

    Attributes:
        norm1 (LayerNorm2d): First normalization layer.
        attn (MDTA): Attention module.
        norm2 (LayerNorm2d): Second normalization layer.
        ffn (GDFN): Feed-forward network module.
    """

    def __init__(self, dim: int, num_heads: int = 8):
        """Initializes the TransformerBlock.

        Args:
            dim: Number of input channels.
            num_heads: Number of attention heads.
        """
        super().__init__()
        self.norm1 = LayerNorm2d(dim)
        self.attn = MDTA(dim, num_heads)
        self.norm2 = LayerNorm2d(dim)
        self.ffn = GDFN(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Applies the transformer block.

        Args:
            x: Input tensor.

        Returns:
            Output tensor.
        """
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x

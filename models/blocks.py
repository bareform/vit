import torch
import torch.nn as nn
import torch.nn.functional as F

class LayerNorm(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.g = nn.Parameter(torch.ones(1, 1, dim))
        self.b = nn.Parameter(torch.zeros(1, 1, dim))

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        var = torch.var(input, dim=-1, unbiased=False, keepdim=True)
        mu = torch.mean(input, dim=-1, keepdim=True)
        out = (input - mu) * torch.rsqrt(var + 1e-5)
        out = out * self.g + self.b
        return out

class LinearAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        heads: int = 4,
        head_dim: int = 32,
    ) -> None:
        super().__init__()

        self.dim = dim
        self.heads = heads
        self.head_dim = head_dim
        self.scale = head_dim ** -0.5

        hidden_dim = heads * head_dim

        self.linear = nn.Linear(
            dim,
            hidden_dim * 3,
            bias=False,
        )

        self.proj = nn.Linear(
            hidden_dim,
            dim,
        )

        self.norm = LayerNorm(dim)

        self.attn_map = None

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        batch_size, num_tokens, _ = input.shape

        q, k, v = torch.chunk(
            self.linear(input),
            chunks=3,
            dim=-1,
        )

        q = q.view(
            batch_size,
            num_tokens,
            self.heads,
            self.head_dim,
        ).transpose(1, 2)

        k = k.view(
            batch_size,
            num_tokens,
            self.heads,
            self.head_dim,
        ).transpose(1, 2)

        v = v.view(
            batch_size,
            num_tokens,
            self.heads,
            self.head_dim,
        ).transpose(1, 2)

        q = q.softmax(dim=-1)
        k = k.softmax(dim=-2)

        q = q * self.scale
        if not torch.is_grad_enabled():
            self.attention_map = (
                q @ k.transpose(-2, -1)
            ).detach()

        context = torch.matmul(
            k.transpose(-2, -1),
            v,
        )

        out = torch.matmul(
            q,
            context,
        )

        out = out.transpose(1, 2).reshape(
            batch_size,
            num_tokens,
            self.heads * self.head_dim,
        )

        out = self.proj(out)
        out = self.norm(out)

        return out

class Block(nn.Module):
    def __init__(
        self,
        in_dim: int,
        attn_dim: int,
        heads: int = 4,
        head_dim: int = 32,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.attention = LinearAttention(
            dim=attn_dim,
            heads=heads,
            head_dim=head_dim,
        )
        self.ln1 = LayerNorm(in_dim)
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, in_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(in_dim * 4, in_dim),
            nn.Dropout(dropout),
        )
        self.ln2 = LayerNorm(in_dim)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        attention_output = self.attention(self.ln1(input))
        out = input + attention_output
        out = out + self.mlp(self.ln2(out))
        return out

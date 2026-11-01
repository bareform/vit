import torch
import torch.nn as nn

class PatchEmbeddings(nn.Module):
    def __init__(
        self,
        in_channels: int,
        embed_dim: int,
        patch_size: int
    ) -> None:
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
            padding=0,
            bias=True,
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        input = input.to(dtype=self.conv.weight.dtype)
        out = self.conv(input)
        out = out.flatten(2).transpose(1, 2)
        return out

class PositionalEmbeddings(nn.Module):
    def __init__(
        self,
        img_size: tuple[int, int, int],
        hidden_dim: int,
        embed_dim: int,
        patch_size: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.patch_embeddings = PatchEmbeddings(
            in_channels=img_size[0],
            embed_dim=embed_dim,
            patch_size=patch_size,
        )
        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_dim))
        num_patches = (img_size[1] // patch_size) * (img_size[2] // patch_size)
        self.position_embeddings = nn.Parameter(
            torch.randn(1, num_patches + 1, hidden_dim)
        )
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.position_embeddings, std=0.02)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        batch_size = input.size(0)
        out = self.patch_embeddings(input)
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        out = torch.cat((cls_tokens, out), dim=1)
        out = out + self.position_embeddings
        out = self.dropout(out)
        return out

import torch
import torch.nn as nn

from .blocks import Block
from .embeddings import PositionalEmbeddings

class Encoder(nn.Module):
    def __init__(
        self,
        num_blocks: int,
        in_dim: int,
        attn_dim: int,
        heads: int = 4,
        head_dim: int = 32,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.blocks = nn.ModuleList([])
        for _ in range(num_blocks):
            self.blocks.append(
                Block(
                    in_dim=in_dim,
                    attn_dim=attn_dim,
                    heads=heads,
                    head_dim=head_dim,
                    dropout=dropout,
                )
            )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        out = input
        for block in self.blocks:
            out = block(out)
        return out

class ViT(nn.Module):
    def __init__(
        self,
        img_size: tuple[int, int, int],
        num_classes: int,
        hidden_dim: int,
        patch_size: int,
        num_blocks: int,
        heads: int = 4,
        head_dim: int = 32,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.embeddings = PositionalEmbeddings(
            img_size=img_size,
            hidden_dim=hidden_dim,
            embed_dim=hidden_dim,
            patch_size=patch_size,
            dropout=dropout,
        )
        self.encoder = Encoder(
            num_blocks=num_blocks,
            in_dim=hidden_dim,
            attn_dim=hidden_dim,
            heads=heads,
            head_dim=head_dim,
            dropout=dropout,
        )
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        out = self.embeddings(input)
        out = self.encoder(out)
        out = out[:, 0, :]
        out = self.classifier(out)
        return out

"""
Defines the classes necessary for the model implementation
----------------------------------------------------------
"""

import torch
from torch import nn

class PatchEmbedding(nn.Module):
    def __init__(self, embed_dim, patch_size, num_patches, in_channels,
            dropout):
        super().__init__()
        self.embed_dim = embed_dim
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.in_channels = in_channels
        self.patcher = nn.Sequential(
                # We use conv for doing the patching
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=embed_dim * in_channels,
                    # if kernel_size = stride -> no overlap
                    kernel_size=patch_size,
                    stride=patch_size,
                    groups=in_channels,
                    ),
                # Linear projection of Flattened Patches. We keep the batch 
                # and the channels (b,c,h,w)
                nn.Flatten(start_dim=2)
                )
        self.position_embeddings = nn.Parameter(
                torch.randn(
                    size=(1, in_channels, num_patches, embed_dim),
                    requires_grad=True,
                    )
                )
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        # Create the patches
        x = self.patcher(x)
        x = x.reshape(1, self.in_channels, self.embed_dim, self.num_patches)
        x = x.permute(0, 1, 3, 2)
        # Unify the position with the patches
        # Patch + Position Embedding
        x = self.position_embeddings + x
        x = self.dropout(x)
        return x

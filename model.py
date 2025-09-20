"""
Defines the classes necessary for the model implementation
----------------------------------------------------------

Patch Embedding code from:
    https://medium.com/@fernandopalominocobo/demystifying-visual-transformersi\
            -with-pytorch-understanding-patch-embeddings-part-1-3-ba380f2aa37f
Perceiver code from:
    https://medium.com/@curttigges/the-annotated-perceiver-74752113eefb
Variational Autoencoder code from:
    https://avandekleut.github.io/vae/

(c) 2025 Bedartha Goswami <bedartha.goswami@iiserpune.ac.in>
"""

import torch
from torch import nn

class PatchEmbedding(nn.Module):
    def __init__(self, embed_dim, patch_size, num_patches, in_channels,
                 dropout, keep_channels=False):
        super().__init__()
        self.embed_dim = embed_dim
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.in_channels = in_channels
        self.keep_channels = keep_channels
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
        if self.keep_channels:
            self.position_embeddings = nn.Parameter(
                    torch.randn(
                        size=(1, in_channels, num_patches, embed_dim),
                        requires_grad=True,
                        )
                    )
        else:
            self.position_embeddings = nn.Parameter(
                    torch.randn(
                        size=(1, num_patches * in_channels, embed_dim),
                        requires_grad=True,
                        )
                    )
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        # Create the patches
        x = self.patcher(x)
        if self.keep_channels:
            # reshape the conv2d output so that it correponds to in_channels
            x = x.reshape(x.shape[0], self.in_channels,
                          self.embed_dim, self.num_patches)
            x = x.permute(0, 1, 3, 2)
        else:
            # Toy example
            # import numpy as np
            # import torch
            # num_patches, in_channels, embed_dim = 8, 2, 4 
            # a = np.arange(num_patches * in_channels * embed_dim)\
            #        .reshape(1, in_channels * embed_dim, num_patches
            # A = torch.from_numpy(a)
            # A.permute(0,2,1).reshape(1, in_channels * num_patches, embed_dim)
            x = x.permute(0, 2, 1)
            x = x.reshape(x.shape[0], self.in_channels*self.num_patches,
                          self.embed_dim)
        # Unify the position with the patches
        # Patch + Position Embedding
        x = self.position_embeddings + x
        x = self.dropout(x)
        return x


class PerceiverAttention(nn.Module):
    """Basic decoder block used both for cross-attention and the latent transformer
    """
    def __init__(self, embed_dim, mlp_dim, n_heads, dropout=0.0):
        super().__init__()

        self.lnorm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim=embed_dim,
                                          num_heads=n_heads,
                                          batch_first=True)

        self.lnorm2 = nn.LayerNorm(embed_dim)
        self.linear1 = nn.Linear(embed_dim, mlp_dim)
        self.act = nn.GELU()
        self.linear2 = nn.Linear(mlp_dim, embed_dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, q):
        # x will be of shape [BATCH_SIZE x TOKENS x EMBED_DIM]
        # q will be of shape [BATCH_SIZE x LATENT_DIM x EMBED_DIM] when this is
        # used for cross-attention; otherwise same as x

        # attention block
        out = self.lnorm1(x)
        out, _ = self.attn(query=q, key=x, value=x)
        # out will be of shape [BATCH_SIZEx LATENT_DIM x EMBED_DIM] after matmul
        # when used for cross-attention; otherwise same as x

        # first residual connection
        resid = out + q

        # dense block
        out = self.lnorm2(resid)
        out = self.linear1(out)
        out = self.act(out)
        out = self.linear2(out)
        out = self.drop(out)

        # second residual connection
        out = out + resid

        return out


class LatentTransformer(nn.Module):
    """Latent transformer module with n_layers count of decoders.
    """
    def __init__(self, embed_dim, mlp_dim, n_heads, dropout, n_layers):
        super().__init__()
        self.transformer = nn.ModuleList(
                [
                    PerceiverAttention(
                        embed_dim=embed_dim,
                        mlp_dim=mlp_dim,
                        n_heads=n_heads,
                        dropout=dropout
                        )
                    for l in range(n_layers)
                    ]
                )

    def forward(self, l):
        for trnfr in self.transformer:
            l = trnfr(l, l)
        return l


class PerceiverIO(nn.Module):
    """
    Implements the Perceiver IO block
    """
    def __init__(self, embed_dim,  mlp_dim, n_heads, dropout):
        """initialize perceiver io class"""
        super().__init__()
        self.embed_dim = embed_dim
        self.mlp_dim = mlp_dim
        self.n_heads = n_heads
        self.attention = PerceiverAttention(
                                      embed_dim=embed_dim,
                                      mlp_dim=mlp_dim,
                                      n_heads=n_heads,
                                      dropout=dropout
                                  )

    def forward(self, x, l):
        out = self.attention(x, l)
        return out


class PerceiverBlock(nn.Module):
    """
    Perceiver Block with the Perceiver IO module and the Latent Transformer
    """
    def __init__(self, embed_dim, latent_dim, mlp_dim, n_heads, n_layers,
                 dropout, batch_size):
        """initialize the Perceiver block"""
        super().__init__()
        self.latent = nn.Parameter(
                    torch.nn.init.trunc_normal_(
                        torch.zeros((
                            batch_size, latent_dim, embed_dim)),
                            mean=0, std=0.02, a=-2, b=2
                        )
                    )
        self.perceiver_io = PerceiverIO(embed_dim=embed_dim,
                                        mlp_dim=mlp_dim,
                                        n_heads=n_heads,
                                        dropout=dropout
                                        )
        self.latent_trnfr = LatentTransformer(embed_dim=embed_dim,
                                              mlp_dim=mlp_dim,
                                              n_heads=n_heads,
                                              n_layers=n_layers,
                                              dropout=dropout
                                              )

    def forward(self, x):
        """forward pass"""
        l = self.perceiver_io(x, self.latent)
        out = self.latent_trnfr(l)
        return out


class StackedPerceiver(nn.Module):
    """
    Stacked Perceiver stuff
    """
    def __init__(self, embed_dim, latent_dims, mlp_dims, n_heads,
                 n_trnfr_layers, dropouts, batch_size):
        """initialize the stacked perceiver"""
        super().__init__()
        len_stack = len(latent_dims)
        self.stacked_prcvr = nn.ModuleList(
                [
                    PerceiverBlock(
                        embed_dim=embed_dim,
                        latent_dim=latent_dims[i],
                        mlp_dim=mlp_dims[i],
                        n_heads=n_heads[i],
                        n_layers=n_trnfr_layers[i],
                        dropout=dropouts[i],
                        batch_size=batch_size
                        )
                    for i in range(len_stack)
                    ]
                )

    def forward(self, x):
        for prcvr_blk in self.stacked_prcvr:
            x = prcvr_blk(x)
        return x


class Encoder(nn.Module):
    """
    Encoder for the VAE
    """
    def __init__(self, vae_latent_dim, sp_latent_dims, sp_embed_dim,
                 sp_mlp_dims, sp_n_heads, sp_n_trnfr_layers, sp_dropouts,
                 batch_size):
        """initialize the VAE encoder"""
        super().__init__()
        self.stacked_perceiver = StackedPerceiver(
                                            embed_dim=sp_embed_dim,
                                            latent_dims=sp_latent_dims,
                                            mlp_dims=sp_mlp_dims,
                                            n_heads=sp_n_heads,
                                            n_trnfr_layers=sp_n_trnfr_layers,
                                            dropouts=sp_dropouts,
                                            batch_size=batch_size
                                            )
        conv1d_ks = int(sp_latent_dims[-1] / vae_latent_dim)
        self.conv1d_1 = nn.Conv1d(in_channels=sp_embed_dim,
                                  out_channels=1,
                                  kernel_size=conv1d_ks,
                                  stride=conv1d_ks)
        self.conv1d_2 = nn.Conv1d(in_channels=sp_embed_dim,
                                  out_channels=1,
                                  kernel_size=conv1d_ks,
                                  stride=conv1d_ks)

        self.N = torch.distributions.Normal(0, 1)
        self.N.loc = self.N.loc
        self.N.scale = self.N.scale
        self.kl = 0

    def forward(self, x):
        """forward pass"""
        l = self.stacked_perceiver(x)
        # permute the patch dimension and the embedding dimension so that
        # conv1d processes each embedding dimension as a channel
        mu = self.conv1d_1(l.permute(0, 2, 1))
        logvar = self.conv1d_2(l.permute(0, 2, 1))
        sig = torch.exp(0.5 * logvar)
        z = mu + sig * self.N.sample(mu.shape)
        return z


class Decoder(nn.Module):
    def __init__(self, vae_latent_dim, sp_latent_dims, sp_embed_dim,
                 sp_mlp_dims, sp_n_heads, sp_n_trnfr_layers, sp_dropouts,
                 batch_size):
        """initialize the VAE decoder"""
        super().__init__()
        conv1d_ks = int(sp_latent_dims[0] / vae_latent_dim)
        self.conv1d = nn.ConvTranspose1d(in_channels=1,
                                         out_channels=sp_embed_dim,
                                         kernel_size=conv1d_ks,
                                         stride=conv1d_ks,
                                         dilation=1)
        self.stacked_perceiver = StackedPerceiver(
                                            embed_dim=sp_embed_dim,
                                            latent_dims=sp_latent_dims[1:],
                                            mlp_dims=sp_mlp_dims,
                                            n_heads=sp_n_heads,
                                            n_trnfr_layers=sp_n_trnfr_layers,
                                            dropouts=sp_dropouts,
                                            batch_size=batch_size
                                            )

    def forward(self, z):
        l = self.conv1d(z)
        l = l.permute(0, 2, 1)
        x = self.stacked_perceiver(l)
        return x


class VariationalAutoencoder(nn.Module):
    """
    Set up the VAE class
    """
    def __init__(self, vae_latent_dim, sp_enc_latent_dims, sp_dec_latent_dims,
                 sp_embed_dim, sp_mlp_dims, sp_n_heads, sp_n_trnfr_layers,
                 sp_dropouts, batch_size):
        """initialize the VAE class"""
        super().__init__()
        self.encoder = Encoder(
                            vae_latent_dim=vae_latent_dim,
                            sp_latent_dims=sp_enc_latent_dims,
                            sp_embed_dim=sp_embed_dim,
                            sp_mlp_dims=sp_mlp_dims,
                            sp_n_heads=sp_n_heads,
                            sp_n_trnfr_layers=sp_n_trnfr_layers,
                            sp_dropouts=sp_dropouts,
                            batch_size=batch_size
                            )
        self.decoder = Decoder(
                            vae_latent_dim=vae_latent_dim,
                            sp_latent_dims=sp_dec_latent_dims,
                            sp_embed_dim=sp_embed_dim,
                            sp_mlp_dims=sp_mlp_dims,
                            sp_n_heads=sp_n_heads,
                            sp_n_trnfr_layers=sp_n_trnfr_layers,
                            sp_dropouts=sp_dropouts,
                            batch_size=batch_size
                            )

    def forward(self, x):
        """forward pass"""
        z = self.encoder(x)
        x = self.decoder(z)
        return (x)


class PatchDecoder(nn.Module):
    """
    remaps the tokens back to grid space
    """
    def __init__(self, embed_dim, data_channels, num_patches, patch_size,
                 input_size):
        """initialize the patch decoder"""
        super().__init__()
        self.embed_dim = embed_dim
        self.data_channels = data_channels
        self.num_patches = num_patches
        self.patch_size = patch_size
        self.input_size = input_size
        self.conv2d = nn.ConvTranspose2d(in_channels=embed_dim * data_channels,
                                         out_channels=data_channels,
                                         kernel_size=patch_size,
                                         stride=patch_size,
                                         dilation=1,
                                         #groups=data_channels
                                         )

    def forward(self, x):
        """forward pass"""
        x = x.reshape(x.shape[0],
                      self.data_channels * self.embed_dim,
                      int(self.input_size[0] / self.patch_size[0]),
                      int(self.input_size[1] / self.patch_size[1])
                      )
        x = self.conv2d(x)
        return x

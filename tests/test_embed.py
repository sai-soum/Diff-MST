import torch

from mst.modules import SpectrogramEncoder

if __name__ == "__main__":
    encoder = SpectrogramEncoder(embed_dim=1024, l2_norm=True)

    mix = torch.randn(4, 2, 262144)
    z = encoder(mix)
    print(z.shape)

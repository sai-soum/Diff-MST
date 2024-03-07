import torch
from mst.panns import TCN
from mst.modules import WaveformTransformerEncoder


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# encoder = TCN(n_inputs=2)
encoder = WaveformTransformerEncoder(n_inputs=2)

print(count_parameters(encoder) / 1e6)

x = torch.randn(4, 2, 262144)


y = encoder(x)
print(y.shape)

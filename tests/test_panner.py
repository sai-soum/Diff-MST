import torch
from dasp_pytorch.functional import stereo_panner

tracks = torch.ones(1, 4, 1)
print(tracks)
print(tracks.shape)

param_dict = {"stereo_panner": {"pan": torch.tensor([0.0, 0.5, 1.0, 0.0])}}

tracks = stereo_panner(tracks, **param_dict["stereo_panner"])

print(tracks)
print(tracks.shape)
tracks.sum(dim=2)

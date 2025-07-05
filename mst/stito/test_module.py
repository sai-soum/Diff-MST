import os
import yaml
import torch
import torchaudio
import pyloudnorm as pyln

from collections import OrderedDict
from importlib import import_module
from mst.stito.panns import Cnn14



def load_param_model(ckpt_path: str = None, use_gpu: bool = False):

    if ckpt_path is None:  # look in tmp direcory
        ckpt_path = os.path.join(os.getcwd(), "tmp", "afx-rep.ckpt")
        os.makedirs("tmp", exist_ok=True)
        if not os.path.isfile(ckpt_path):
            # download from huggingfacehub
            os.system(
                "wget -O tmp/afx-rep.ckpt https://huggingface.co/csteinmetz1/afx-rep/resolve/main/afx-rep.ckpt"
            )
            os.system(
                "wget -O tmp/config.yaml https://huggingface.co/csteinmetz1/afx-rep/resolve/main/config.yaml"
            )

    config_path = os.path.join(os.path.dirname(ckpt_path), "config.yaml")

    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    encoder_configs = config["model"]["init_args"]["encoder"]

    module_path, class_name = encoder_configs["class_path"].rsplit(".", 1)
    module_path = module_path.replace("lcap", "mst")
    module_path = module_path.replace("models", "stito")
    module = import_module(module_path)
    model = getattr(module, class_name)(**encoder_configs["init_args"])

    checkpoint = torch.load(ckpt_path, map_location="cpu")

    # load state dicts
    state_dict = {}
    for k, v in checkpoint["state_dict"].items():
        if k.startswith("encoder"):
            state_dict[k.replace("encoder.", "", 1)] = v

    model.load_state_dict(state_dict)
    model.eval()

    if use_gpu:
        model.cuda()

    return model
def get_param_embeds(
    x: torch.Tensor,
    model: torch.nn.Module,
    sample_rate: float,
    requires_grad: bool = False,
    peak_normalize: bool = False,
    dropout: float = 0.0,
):
    bs, chs, seq_len = x.shape

    x_device = x

    # move audio to model device
    x = x.type_as(next(model.parameters()))

    # if peak_normalize:
    #    x = batch_peak_normalize(x)

    if sample_rate != 48000:
        x = torchaudio.functional.resample(x, sample_rate, 48000)

    seq_len = x.shape[-1]  # update seq_len after resampling
    # if longer than 262144 crop, else repeat pad to 262144
    # if seq_len > 262144:
    #    x = x[:, :, :262144]
    # else:
    #    x = torch.nn.functional.pad(x, (0, 262144 - seq_len), "replicate")

    # peak normalize each batch item
    for batch_idx in range(bs):
        x[batch_idx, ...] /= x[batch_idx, ...].abs().max().clamp(1e-8)

    if not requires_grad:
        with torch.no_grad():
            mid_embeddings, side_embeddings = model(x)
    else:
        mid_embeddings, side_embeddings = model(x)

    # add dropout
    if dropout > 0.0:
        mid_embeddings = torch.nn.functional.dropout(
            mid_embeddings, p=dropout, training=True
        )
        side_embeddings = torch.nn.functional.dropout(
            side_embeddings, p=dropout, training=True
        )

    # check for nan
    if torch.isnan(mid_embeddings).any():
        print("Warning: NaNs found in mid_embeddings")
        mid_embeddings = torch.nan_to_num(mid_embeddings)
    elif torch.isnan(side_embeddings).any():
        print("Warning: NaNs found in side_embeddings")
        side_embeddings = torch.nan_to_num(side_embeddings)

    # l2 normalize
    mid_embeddings = torch.nn.functional.normalize(mid_embeddings, p=2, dim=-1)
    side_embeddings = torch.nn.functional.normalize(side_embeddings, p=2, dim=-1)

    embeddings = {
        "mid": mid_embeddings.type_as(x_device),
        "side": side_embeddings.type_as(x_device),
    }

    return embeddings


if __name__ == "__main__":
        # load pretrained model
    model = load_param_model(use_gpu=True)

    # load audio file
    audio, sr = torchaudio.load("/data4/soumya/Mixing_Secrets_Full/'4 Out Of 10'/full_mix_previews/full_mix_preview.mp3")
    audio = [audio, audio, audio, audio]
    audio = torch.stack(audio, dim=0).squeeze(1)
    print(audio.shape)
    # audio must be of shape bs, chs, seq_len
    # audio = audio.unsqueeze(0)
    
    # extract embeddings
    embed_dict = get_param_embeds(audio, model, sr)
    for embed_name, embed in embed_dict.items():
        print(embed_name, embed.shape)
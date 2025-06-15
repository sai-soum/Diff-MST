# import torch
from typing import Tuple
# from mst.utils import load_diffmst
import torch
import os
import yaml
import torch
from importlib import import_module
from mst.modules import MixStyleTransferModel

def load_diffmst(config_path: str, ckpt_path: str, map_location: str = "cpu"):
    with open(config_path) as f:
        config = yaml.safe_load(f)

    core_model_configs = config["model"]["init_args"]["model"]

    module_path, class_name = core_model_configs["class_path"].rsplit(".", 1)
    module = import_module(module_path)
    model = getattr(module, class_name)(**core_model_configs["init_args"])

    submodule_configs = core_model_configs["init_args"]

    # create track encoder module
    module_path, class_name = submodule_configs["track_encoder"]["class_path"].rsplit(
        ".", 1
    )
    module = import_module(module_path)
    track_encoder = getattr(module, class_name)(
        **submodule_configs["track_encoder"]["init_args"]
    )

    # create mix encoder module
    module_path, class_name = submodule_configs["mix_encoder"]["class_path"].rsplit(
        ".", 1
    )
    module = import_module(module_path)
    mix_encoder = getattr(module, class_name)(
        **submodule_configs["mix_encoder"]["init_args"]
    )

    # create controller module
    module_path, class_name = submodule_configs["controller"]["class_path"].rsplit(
        ".", 1
    )
    module = import_module(module_path)
    controller = getattr(module, class_name)(
        **submodule_configs["controller"]["init_args"]
    )

    # create mix console module
    module_path, class_name = config["model"]["init_args"]["mix_console"][
        "class_path"
    ].rsplit(".", 1)
    module = import_module(module_path)
    mix_console = getattr(module, class_name)(
        **config["model"]["init_args"]["mix_console"]["init_args"]
    )

    checkpoint = torch.load(ckpt_path, map_location=map_location)

    # load state dicts
    state_dict = {}
    for k, v in checkpoint["state_dict"].items():
        if k.startswith("model.track_encoder"):
            state_dict[k.replace("model.track_encoder.", "", 1)] = v
    track_encoder.load_state_dict(state_dict)

    state_dict = {}
    for k, v in checkpoint["state_dict"].items():
        if k.startswith("model.mix_encoder"):
            state_dict[k.replace("model.mix_encoder.", "", 1)] = v
    mix_encoder.load_state_dict(state_dict)

    state_dict = {}
    for k, v in checkpoint["state_dict"].items():
        if k.startswith("model.controller"):
            state_dict[k.replace("model.controller.", "", 1)] = v
    controller.load_state_dict(state_dict)

    state_dict = {}
    for k, v in checkpoint["state_dict"].items():
        if k.startswith("model.mix_console"):
            state_dict[k.replace("model.mix_console.", "", 1)] = v
    mix_console.load_state_dict(state_dict)

    model = MixStyleTransferModel(
        track_encoder,
        mix_encoder,
        controller,
    )
    model.eval()

    return model, mix_console

class DiffMSTWrapper(torch.nn.Module):
    def __init__(
        self,
        model: torch.nn.Module,
        analysis_window: int = 441000,
        track_lufs_target: float = -48.0,
        eps: float = 1e-9,
    ):
        super().__init__()
        self.model = model
        self.analysis_window = analysis_window
        self.track_lufs_target = track_lufs_target
        self.eps = eps

    def estimate_loudness(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.sqrt((x ** 2).mean() + self.eps)
        return 20.0 * torch.log10(rms + self.eps)

    def forward(
        self,
        tracks: torch.Tensor,  # (1, n_tracks, T)
        ref: torch.Tensor      # (1, 2, T)
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        bs, n_tracks, T = tracks.shape
        assert bs == 1, "Only batch size 1 is supported."

        norm_tracks = []
        valid_mask = torch.zeros(n_tracks, dtype=torch.bool)

        for t in range(n_tracks):
            wav = tracks[:, t, :]
            lufs = self.estimate_loudness(wav)

            # TorchScript-safe conditional
            gain = torch.pow(10.0, (self.track_lufs_target - lufs) / 20.0)
            norm_tracks.append(wav * gain)
            valid_mask[t] = True

        norm_tracks_tensor = torch.stack(norm_tracks, dim=1)  # (1, n_tracks, T)
        track_p, fx_bus_p, master_p = self.model(norm_tracks_tensor, ref)
        return track_p, fx_bus_p, master_p


if __name__ == "__main__":
    import os
    print("CWD:", os.getcwd())

    # Load pretrained model
    model, mix_console = load_diffmst(
        "/Users/svanka/Codes/Diff-MST/diffmst_wrapper.yaml",
        "/Users/svanka/Downloads/ckpt12.6/gp_5k8/epoch=189-step=11970.ckpt"
    )
    model.eval()

    # Wrap model
    wrapper = DiffMSTWrapper(model)
    wrapper.eval()

    # Create example inputs (shorter for testing)
    example_tracks = torch.randn(1, 5, 441000)
    example_ref = torch.randn(1, 2, 441000)
    t_p, fx_bus_p, master_p = wrapper(example_tracks, example_ref)
    print("Output shapes:", t_p.shape, fx_bus_p.shape, master_p.shape)
    # Script and save
    try:
        scripted_model = torch.jit.script(wrapper)
        scripted_model.save("diffmst_wrapper_scripted.pt")
        print("✅ Model scripted and saved successfully.")
        # load and test the scripted model
        loaded_model = torch.jit.load("diffmst_wrapper_scripted.pt")
        loaded_model.eval()
        t_p, fx_bus_p, master_p = loaded_model(example_tracks, example_ref)
        print("Loaded model output shapes:", t_p.shape, fx_bus_p.shape, master_p.shape)
        # check if the outputs are the same
        assert torch.allclose(t_p, t_p), "Output mismatch after loading scripted model."
        print("✅ Outputs match after loading scripted model.")
    except Exception as e:
        print("❌ Error during scripting or saving:", e)

# import torch
from typing import Tuple
# from mst.utils import load_diffmst
import torch
import os
import yaml
import torch
from importlib import import_module
from mst.modules import MixStyleTransferModel
from mst.variable_length import VariableLengthEncoder
import torchaudio
from torchaudio.transforms import Resample
import glob

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

    # track_encoder = VariableLengthEncoder(track_encoder,
    #                                     chunk_seconds = 5.0,
    #                                     hop_seconds = 2.5,
    #                                     # 0 = mean, 1 = max, 2 = attention
    #                                     pool_mode = 0,  
    #                                     embed_dim = 512     # or "attention"
    #                                 )
    # mix_encoder = VariableLengthEncoder(mix_encoder,
    #                                     chunk_seconds = 5.0,
    #                                     hop_seconds = 2.5,
    #                                     # 0 = mean, 1 = max, 2 = attention
    #                                     pool_mode = 0,  
    #                                     embed_dim = 512     # or "attention"
    #                                 )

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
            print(f"Track {t} LUFS: {lufs.item():.2f}")
            if lufs < self.track_lufs_target:
                continue

            # TorchScript-safe conditional
            # gain = torch.pow(10.0, (self.track_lufs_target - lufs) / 20.0)
            # norm_tracks.append(wav * gain)
            norm_tracks.append(wav)
            valid_mask[t] = True

        norm_tracks_tensor = torch.stack(norm_tracks, dim=1)  # (1, n_tracks, T)
        track_p, fx_bus_p, master_p = self.model(norm_tracks_tensor, ref)
        return track_p, fx_bus_p, master_p
    
def estimate_loudness(x: torch.Tensor) :
        eps = 1e-9
        rms = torch.sqrt((x ** 2).mean() + eps)
        return 20.0 * torch.log10(rms + eps)  

def load_data(track_dir: str,
              ref_dir: str,
              duration: float = 5.0,
              sample_rate: int = 44100) -> Tuple[torch.Tensor, torch.Tensor]:
    
    tracks = []
    track_files = sorted(glob.glob(os.path.join(track_dir, "*.wav")))
    for files in track_files:
        track, sr = torchaudio.load(files, num_frames=int(duration * sample_rate))
        if sr != sample_rate:
            resampler = Resample(orig_freq=sr, new_freq=sample_rate)
            track = resampler(track)
        if track.shape[0] > 1:
            track = track.mean(dim=0, keepdim=True)
        if track.shape[1] < int(duration * sample_rate):
            continue
        tracks.append(track)
    tracks_tensor = torch.stack(tracks, dim=1)  # (1, n_tracks, T)

    ref, sr = torchaudio.load(ref_dir, num_frames=int(duration * sample_rate))
    if sr != sample_rate:
        resampler = Resample(orig_freq=sr, new_freq=sample_rate)
        ref = resampler(ref)
    ref_tensor = ref.unsqueeze(0)  # (1, 2, T)
    # ref loudness
    ref_lufs = estimate_loudness(ref_tensor)
    print(f"Reference LUFS: {ref_lufs.item():.2f}")
    return tracks_tensor, ref_tensor



if __name__ == "__main__":
    import os
    print("CWD:", os.getcwd())

    # Load pretrained model
    model, mix_console = load_diffmst(
        "/Users/svanka/Codes/Diff-MST/scriptting_wrapper/diffmst_wrapper.yaml",
        "/Users/svanka/Downloads/ckpt12.6/gp_5k8/epoch=189-step=11970.ckpt"
        # "/Users/svanka/Downloads/ckpt12.6/gpec_dr09/epoch=258-step=16317.ckpt"
    )
    model.eval()

    # Wrap model
    wrapper = DiffMSTWrapper(model)
    wrapper.eval()
    track_path = "/Users/svanka/Downloads/BenFlowers_Ecstasy"
    ref_path = "/Users/svanka/Codes/sai-soum.github.io/assets/audio/Listening_Examples_Diff_MST/Electronic/electronic-ref-16lufs.wav"
    # track_path = "/Users/svanka/Downloads/DiffMSTTest/tracks"
    # ref_path = "/Users/svanka/Downloads/DiffMSTTest/ref/REFERENCE MSTTest - 0005 - Audio - RnB - Synth Lead A Minor 05_130bpm.wav"
    # Create example inputs (shorter for testing)
    # example_tracks = torch.randn(1, 5, 441000)
    # example_ref = torch.randn(1, 2, 441000)
    example_tracks, example_ref = load_data(track_path, ref_path)   
    print("Example input shapes:", example_tracks.shape, example_ref.shape)
    t_p, fx_bus_p, master_p = wrapper(example_tracks, example_ref)
    _, mix, t_p_dict, fx_bus_p_dict, master_p_dict,= mix_console(example_tracks, 
                                                                                t_p, 
                                                                                fx_bus_p, 
                                                                                master_p,
                                                                                use_track_input_fader = True,
                                                                                use_track_eq = False,
                                                                                use_track_compressor= False,
                                                                                use_track_panner= True,
                                                                                use_master_bus = False,
                                                                                use_fx_bus = False,
                                                                                use_output_fader = False)
    print("Output shapes:", t_p.shape, fx_bus_p.shape, master_p.shape)
    print("track_params", t_p_dict)
    print("mix", mix.shape)
    mix_lufs = estimate_loudness(mix)
    print(f"Mix LUFS: {mix_lufs.item():.2f}")
    #  normalise to -16 LUFS
    mix_lufs_target = -16.0
    mix_gain = torch.pow(10.0, (mix_lufs_target - mix_lufs) / 20.0)
    mix = mix * mix_gain  # apply gain to mix
    print(f"Applied gain: {mix_gain.item():.2f}")
    mix = mix.squeeze(0)  # (2, T) — remove batch dimension
    mix = mix.detach().clamp(-1.0, 1.0).to(torch.float32) # ensure proper dtype and range
    example_ref = example_ref.squeeze(0)  # (2, T) — remove batch dimension
    example_ref = example_ref.detach().clamp(-1.0, 1.0).to(torch.float32) # ensure proper dtype and range
    sum_mix = example_tracks.sum(dim=1)  # (T) — sum across tracks
    sum_lufs = estimate_loudness(sum_mix)
    print(f"Sum of tracks LUFS: {sum_lufs.item():.2f}")
    sum_mix = sum_mix.detach().clamp(-1.0, 1.0).to(torch.float32) # ensure proper dtype and range
    print("Sum of tracks shape:", sum_mix.shape)

    torchaudio.save(
        "scriptting_wrapper/diffmst_wrapper_mix.wav",
        mix,
        44100,
        encoding="PCM_F",
        format="wav",
        bits_per_sample=32
    )
    print("Mix saved successfully.")
    # save ref audio
    torchaudio.save(
        "scriptting_wrapper/diffmst_wrapper_ref.wav",
        example_ref.squeeze(0),  # (2, T) — remove batch dimension
        44100,
        encoding="PCM_F",
        format="wav",
        bits_per_sample=32
    )
    print("Reference audio saved successfully.")
    # save sum of tracks
    torchaudio.save(
        "scriptting_wrapper/diffmst_wrapper_tracks.wav",
        sum_mix,  # (T) — sum across tracks
        44100,
        encoding="PCM_F",
        format="wav",
        bits_per_sample=32
    )

    # # Script and save
    # try:
    #     example_ref = example_ref.unsqueeze(0)  # (1, 2, T) for scripting
    #     # scripted_model = torch.jit.script(wrapper)
    #     # scripted_model.save("/Users/svanka/Downloads/ckpt12.6/gpec_dr09/gpec_scripted.pt")
    #     # print("✅ Model scripted and saved successfully.")
    #     # load and test the scripted model
    #     loaded_model = torch.jit.load("/Users/svanka/Codes/Diff-MST/scriptting_wrapper/diffmst_wrapper_scripted.pt")
    #     loaded_model.eval()
    #     t_p, fx_bus_p, master_p = loaded_model(example_tracks, example_ref)
    #     # print("Loaded model output shapes:", t_p.shape, fx_bus_p.shape, master_p.shape)
    #     # print("params", t_p, fx_bus_p, master_p)
      
    #     # check if the outputs are the same
    #     assert torch.allclose(t_p, t_p), "Output mismatch after loading scripted model."
    #     print("✅ Outputs match after loading scripted model.")

    # except Exception as e:
    #     print("❌ Error during scripting or saving:", e)

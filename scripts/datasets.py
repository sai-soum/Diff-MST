import os
import sys
import glob
import torchaudio
from tqdm import tqdm


def process_mixing_secrets():
    out_dir = "/import/c4dm-datasets-ext/mixing-secrets-mono/"
    # process LibriSpeech dataset to 48 kHz sample rate
    root_dir = "/import/c4dm-datasets-ext/mixing-secrets/"

    song_dirs = glob.glob(os.path.join(root_dir, "*"))

    for song_dir in song_dirs:
        song_out_dir = os.path.join(out_dir, os.path.basename(song_dir))
        if os.path.exists(song_out_dir):
            continue
        os.makedirs(song_out_dir, exist_ok=True)
        song_track_dir = os.path.join(song_dir, "tracks")
        song_out_track_dir = os.path.join(song_out_dir, "tracks")
        os.makedirs(song_out_track_dir, exist_ok=True)
        print(song_track_dir)

        song_track_sub_dirs = glob.glob(os.path.join(song_track_dir, "*"))

        track_load_dir = None
        for song_track_sub_dir in song_track_sub_dirs:
            if "_Full" in song_track_sub_dir:
                track_load_dir = song_track_sub_dir

        if track_load_dir is None:
            track_load_dir = song_track_sub_dirs[0]

        track_filepaths = glob.glob(os.path.join(track_load_dir, "*.wav"))

        for filepath in tqdm(track_filepaths):
            out_filepath = os.path.join(song_out_track_dir, os.path.basename(filepath))
            # convert the sample rate to 48 kHz using ffmpeg
            try:
                x, sr = torchaudio.load(filepath)
            except Exception as e:
                print(e)
                continue

            if sr != 48000:
                x = torchaudio.functional.resample(x, sr, 48000)

            if x.shape[0] == 2:
                torchaudio.save(
                    out_filepath.replace(".wav", "_L.wav"),
                    x[0:1, :],
                    48000,
                    encoding="PCM_S",
                    bits_per_sample=16,
                )
                torchaudio.save(
                    out_filepath.replace(".wav", "_R.wav"),
                    x[1:2, :],
                    48000,
                    encoding="PCM_S",
                    bits_per_sample=16,
                )
            else:
                torchaudio.save(
                    out_filepath,
                    x,
                    48000,
                    encoding="PCM_S",
                    bits_per_sample=16,
                )


def process_medleydb():
    out_dir = "/import/c4dm-datasets-ext/medleydb-mono/"
    # process LibriSpeech dataset to 48 kHz sample rate
    root_dirs = [
        "/import/c4dm-datasets/MedleyDB_V1/V1",
        "/import/c4dm-datasets/MedleyDB_V2/V2",
    ]

    for root_dir in root_dirs:
        song_dirs = glob.glob(os.path.join(root_dir, "*"))

        for song_dir in song_dirs:
            print(song_dir)
            song_out_dir = os.path.join(out_dir, os.path.basename(song_dir))
            # if os.path.exists(song_out_dir):
            #    continue
            os.makedirs(song_out_dir, exist_ok=True)
            song_track_dir = os.path.join(song_dir, f"{os.path.basename(song_dir)}_RAW")
            song_out_track_dir = os.path.join(song_out_dir, f"tracks")
            os.makedirs(song_out_track_dir, exist_ok=True)
            print(song_track_dir)

            track_filepaths = glob.glob(os.path.join(song_track_dir, "*.wav"))

            for filepath in tqdm(track_filepaths):
                out_filepath = os.path.join(
                    song_out_track_dir, os.path.basename(filepath)
                )
                # convert the sample rate to 48 kHz using ffmpeg
                try:
                    x, sr = torchaudio.load(filepath)
                except Exception as e:
                    print(e)
                    continue

                if sr != 48000:
                    x = torchaudio.functional.resample(x, sr, 48000)

                if x.shape[0] == 2:
                    torchaudio.save(
                        out_filepath.replace(".wav", "_L.wav"),
                        x[0:1, :],
                        48000,
                        encoding="PCM_S",
                        bits_per_sample=16,
                    )
                    torchaudio.save(
                        out_filepath.replace(".wav", "_R.wav"),
                        x[1:2, :],
                        48000,
                        encoding="PCM_S",
                        bits_per_sample=16,
                    )
                else:
                    torchaudio.save(
                        out_filepath,
                        x,
                        48000,
                        encoding="PCM_S",
                        bits_per_sample=16,
                    )


if __name__ == "__main__":
    dataset = "medleydb"

    if dataset == "mixing-secrets":
        process_mixing_secrets()
    elif dataset == "medleydb":
        process_medleydb()
    else:
        raise NotImplementedError

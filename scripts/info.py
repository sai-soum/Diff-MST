import os
import glob
import torchaudio
from tqdm import tqdm

root_dir = "/import/c4dm-datasets-ext/mixing-secrets/"

# find all directories containing tracks
song_dirs = glob.glob(os.path.join(root_dir, "*"))

files = {"stereo": 0, "mono": 0}

for song_dir in tqdm(song_dirs):
    # get all tracks in song dir
    track_filepaths = glob.glob(os.path.join(song_dir, "tracks", "**", "*.wav"))

    for track_filepath in track_filepaths:
        # get into
        md = torchaudio.info(track_filepath)

        if md.num_channels == 2:
            files["stereo"] += 1
        elif md.num_channels == 1:
            files["mono"] += 1

    print(files)

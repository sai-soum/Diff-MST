import os
import multiprocessing
from pydub import AudioSegment
from functools import partial

def convert_single_file(input_root, output_root, file_path):
    rel_path = os.path.relpath(os.path.dirname(file_path), input_root)
    output_dir = os.path.join(output_root, rel_path)
    os.makedirs(output_dir, exist_ok=True)

    output_filename = os.path.splitext(os.path.basename(file_path))[0] + ".wav"
    output_path = os.path.join(output_dir, output_filename)

    try:
        audio = AudioSegment.from_mp3(file_path)
        audio = audio.set_frame_rate(44100)
        audio.export(output_path, format="wav")
        print(f"[✓] {file_path} -> {output_path}")
    except Exception as e:
        print(f"[!] Error converting {file_path}: {e}")

def get_all_mp3_files(root):
    return [
        os.path.join(dirpath, filename)
        for dirpath, _, filenames in os.walk(root)
        for filename in filenames if filename.lower().endswith(".mp3")
    ]

def convert_all(input_root, output_root, max_cpu_usage=0.5):
    all_files = get_all_mp3_files(input_root)
    num_cores = max(1, int(multiprocessing.cpu_count() * max_cpu_usage))

    print(f"Found {len(all_files)} MP3 files. Using {num_cores} workers.")
    
    with multiprocessing.Pool(processes=num_cores) as pool:
        pool.map(partial(convert_single_file, input_root, output_root), all_files)

# Example usage
if __name__ == "__main__":
    input_dataset_folder = "/import/c4dm-datasets-ext/mtg-jamendo"    # Replace with your input path
    output_dataset_folder = "/import/c4dm-scratch-02/MTG-jamendo-wav"  # Replace with your output path
    convert_all(input_dataset_folder, output_dataset_folder)

import os
import re
import yaml
import soundfile as sf
import librosa
import numpy as np
import pickle
import shutil
from glob import glob
from tqdm import tqdm
import audalign as ad

def opj(*args):
    return os.path.join(*args)

def add_space_between_cases(text):
    return re.sub(r'([a-z])([A-Z])', r'\1 \2', text)

def categorize_tracks(multitrack_dir):
    correspondance = {
        'kick': [], 'snare': [], 'aux_perc': [], 'percussion': [], 'drum': [], 'bass': [], 'synth': [], 'keys': [], 
        'room': [], 'organ': [], 'brass': [], 'woodwind': [], 'vocal': [], 'string': [], 'fx': [], 'guitar': [], 'other': []
    }
    track_names = glob(opj(multitrack_dir, "*.wav"))
    if len(track_names) == 0:
        return {}
    names = [os.path.basename(name) for name in track_names]
    length = len(names)
    
    for name in names:
        process_name = re.sub(r'\d+', '', name.replace(".wav", "").replace("_", " ").replace("-", " "))
        process_name = add_space_between_cases(process_name).lower()
        
        categories = {
            "kick": ["kick"],
            "snare": ["snare"],
            "aux_perc": ["hihat", "tom", "cymbal", "ride", "crash", "splash", "hi", "hat", "ride"],
            "percussion": ["perc", "shaker", "kalimba", "clap", "snap", "djembe", "cowbell", "conga", "glockenspiel", "timpani", "congo", "beat", "overhead", "beatbox", "tambourine", "triangle", "maraca", "bongo", "cabasa", "guiro", "woodblock", "clave", "castanet", "agogo", "whistle", "bell", "chime", "gong", "tambourine"],
            "drum": ["drum"],
            "bass": ["bass"],
            "synth": ["synth"],
            "keys": ["piano", "keys", "clavinet", "rhodes", "accordion"],
            "room": ["room"],
            "organ": ["organ"],
            "brass": ["brass", "trumpet", "trombone", "horn", "bagpipe", "tuba", "euphonium"],
            "woodwind": ["woodwind", "flute", "clarinet", "oboe", "saxophone", "sax", "bassoon", "alto", "soprano"],
            "vocal": ["vocal", "choir", "vox", "backing"],
            "string": ["string", "violin", "cello", "viola"],
            "fx": ["fx"],
            "guitar": ["guitar", "gtr", "ukulele", "ukelele", "banjo", "fiddle", "mandolin"]
        }
        
        found = False
        for category, keywords in categories.items():
            if any(keyword in process_name for keyword in keywords):
                correspondance[category].append(name)
                found = True
                break
        
        if not found:
            correspondance['other'].append(name)
    
    return {k: v for k, v in correspondance.items() if v}

def get_first_subdir(parent_dir):
    """Returns the first subdirectory inside a given directory, or None if empty."""
    if os.path.exists(parent_dir) and os.path.isdir(parent_dir):
        subdirs = [d for d in os.listdir(parent_dir) if os.path.isdir(opj(parent_dir, d))]
        
        # if there is a subdir called _MACOSX, delete it
        if "__MACOSX" in subdirs:
            # remove it from the directory
            shutil.rmtree(opj(parent_dir, "__MACOSX"))
            subdirs.remove("__MACOSX")
        if len(subdirs) == 0:
            print(f"Empty directory: {parent_dir}")
        if subdirs:
            return opj(parent_dir, subdirs[0])
    return None

def save_correspondance(dataset_dir):
    song_dirs = glob(opj(dataset_dir, "*"))
    print(f"Found {len(song_dirs)} songs")  
    failed_dir = {}
    failed_dir["excerpt"] = []
    failed_dir["full"] = []
    
    for song_dir in song_dirs:
        song_name = os.path.basename(song_dir)
        # check if the correspondance files already exist
            # check if they contain {}

        # excerpt_multitrack_dir = get_first_subdir(opj(song_dir, "excerpt_multitrack"))
        excerpt_multitrack_dir = glob(opj(song_dir, "excerpt_multitrack", "*", "*.wav"))
        if excerpt_multitrack_dir:
            excerpt_multitrack_dir = os.path.dirname(excerpt_multitrack_dir[0])
        else:
            excerpt_multitrack_dir = None
        full_multitrack_dir = glob(opj(song_dir, "full_multitrack", "*", "*.wav"))
        if full_multitrack_dir:
            full_multitrack_dir = os.path.dirname(full_multitrack_dir[0])
        else:
            full_multitrack_dir = None
        # full_multitrack_dir = get_first_subdir(opj(song_dir, "full_multitrack"))
        
        correspondance_excerpt = categorize_tracks(excerpt_multitrack_dir) if excerpt_multitrack_dir else {}
        correspondance_full = categorize_tracks(full_multitrack_dir) if full_multitrack_dir else {}
        
        aligned_dir = opj(song_dir, "aligned")
        os.makedirs(aligned_dir, exist_ok=True)
        if correspondance_excerpt!= {}:
            print("success")
            with open(opj(aligned_dir, "correspondance_excerpt.yaml"), "w") as f:
                yaml.dump(correspondance_excerpt, f)
        else: 
            # check if the excerpt multitrack directory is empty
           
            failed_dir["excerpt"].append(os.path.basename(song_dir))
        if correspondance_full != {}:
            print("success")
            with open(opj(aligned_dir, "correspondance_full.yaml"), "w") as f:
                yaml.dump(correspondance_full, f)
        else: 
            # check if the full multitrack directory is empty
            failed_dir["full"].append(os.path.basename(song_dir))
    print(len(failed_dir["excerpt"]), "excerpt directories failed")
    print(len(failed_dir["full"]), "full directories failed ")  
    # save failed directories as yaml in data/
    with open(opj("data/failed_dir.yaml"), "w") as f:
        yaml.dump(failed_dir, f)
    print(f"Failed directories saved in data/failed_dir.yaml")
        

if __name__ == "__main__":
    dataset_folder = "/data4/soumya/Mixing_Secrets_Full"
    save_correspondance(dataset_folder)
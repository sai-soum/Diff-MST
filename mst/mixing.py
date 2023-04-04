# Store mixing functions here (e.g. knowledge engineering)
import torch
import json
import random
import mst.modules
import mst.dataloaders.medley 
from mst.dataloaders.medley import MedleyDBDataset



def instrument_metadata(instrument_id: list):
    """Convert the metadata info into istrument names
    """
    instrument_number_file = json.load(open("/homes/ssv02/Diff-MST/inst_id.txt"))
    #iid = instrument_id.cpu().tolist()
    bs,num_tracks = instrument_id.size()
    mdata = []
    for i in range(bs):
        iid = instrument_id[i,:]
        print(iid)
        iid = iid.cpu().tolist()
        metadata ={}
        for j, id in enumerate(iid): 
            instrument = [instrument for instrument, number in instrument_number_file.items() if number == id]
            metadata[j] = instrument[0]
            
        
        #metadata = dict(zip(range(len(metadata)), metadata))
        #print("metadata:", metadata)
        mdata.append(metadata)
    #print("mdata:", mdata)
    return mdata


def naive_random_mix(tracks: torch.Tensor, mix_console: torch.nn.Module):
    """Generate a random mix by sampling parameters uniformly on the parameter ranges.

    Args:
        tracks (torch.Tensor):
        mix_console (torch.nn.Module):

    Returns:
        mix (torch.Tensor)
        param_dict (dict):
    """
    bs, num_tracks, seq_len = tracks.size()

    # generate random parameter tensor
    mix_params = torch.rand(bs, num_tracks, mix_console.num_control_params)
    mix_params = mix_params.type_as(tracks)

    # generate a mix of the tracks
    mix, param_dict = mix_console(tracks, mix_params)

    return mix, param_dict




def knowledge_engineering_mix(
    tracks: torch.Tensor, mix_console: torch.nn.Module, instrument_id: list, genre: str
):
    """Generate a mix using knowledge engineering
    """

    bs, num_tracks, seq_len = tracks.size()
    mdata = instrument_metadata(instrument_id)
    #print("mdata:", mdata)

    if mix_console.num_control_params != 2:
        raise ValueError(
            "Knowledge engineering mixing only works with mix consoles with 2 control parameters"
        )
    
    mix_params = torch.full((bs, num_tracks, mix_console.num_control_params), -18.0)
    mix_params[:,:,1] = 0.5
    

    for j in range(bs):
        metadata = mdata[j]
        for i in range(len(metadata)):
            
            if metadata[i]=="silence":
                mix_params[j, i, 0]= -88.0
                pass
            elif metadata[i] in ["female singer", "male singer", "male rapper", "Rap", "male speaker"]:
                #print("vocal")
                #centre pan
                mix_params[j, i, 0] = -7.0
            elif metadata[i] == "vocalist":
                #wide pan
                mix_params[j, i, 0] = -9.0
                mix_params[j, i, 1] = random.choice([0.1,0.2,0.8,0.9])
            elif metadata[i] in ["electric bass", "double bass"]:
                #print("bass")
                #centre pan
                mix_params[j, i, 0] = -10.0
            elif metadata[i] in ["bass drum", "kick drum"]:
                #print("kick")
                #centre pan
                mix_params[j, i, 0] = -12.0
            elif metadata[i] == ["snare drum","tabla"]:
                #mid pan
                mix_params[j, i, 0] = -8.0
                mix_params[j, i, 1] = random.choice([0.4,0.6])
            elif metadata[i] in ["drum set", "drum machine"]:
                #mid pan
                mix_params[j, i, 0] = -10.0
                mix_params[j, i, 1] = random.choice([0.3,0.4,0.6,0.7])
            elif metadata[i] in ["high hat", "toms", "timpani", "bongo", "doumbek","darbuka"]:
                #mid pan
                mix_params[j, i, 0] = -10.0
                mix_params[j, i, 1] = random.choice([0.3,0.4,0.6,0.7])
            elif metadata[i] in ["cymbal", "glockenspiel", "claps", "vibraphone", "auxiliary percussion", "chimes", "gong"]:
                #wide pan
                mix_params[j, i, 0] = -12.0
                mix_params[j, i, 1] = random.choice([0.1,0.2,0.8,0.9])
            elif metadata[i] in ["guiro", "cowbell"]:
                #extreme pan
                mix_params[j, i, 0] = -14.0
                mix_params[j, i, 1] = random.choice([0.0,1.0])
            elif metadata[i] in ["sleigh bells", "shaker", "tambourine", "cabasa", "castanet", "scratches","gu"]:
                #extreme pan
                mix_params[j, i, 0] = -14.0
                mix_params[j, i, 1] = random.choice([0.0,1.0])
            elif metadata[i] in ["fx/processed sound", "sampler", "synthesizer", "harmonica"]:
                #mid pan
                mix_params[j, i, 0] = -12.0
                mix_params[j, i, 1] = random.choice([0.3,0.4,0.6,0.7])
            elif metadata[i] in ["electric piano", "accordion", "harpsichord", "organ", "piano", "tack piano", "melodica"]:
                #print("piano")
                #mid pan
                mix_params[j, i, 0] = -8.0
                mix_params[j, i, 1] = random.choice([0.3,0.4,0.6,0.7])
            elif metadata[i] in ["cello", "double bass"]:
                #print("cello")
                #mid pan
                mix_params[j, i, 0] = -10.0
                mix_params[j, i, 1] = random.choice([0.3,0.4,0.6,0.7])
            elif metadata[i] in ["violin", "viola", "string section", "violin section", "cello section", "viola section", "harp","erhu"]:
                #print("strings")
                #wide pan
                mix_params[j, i, 0] = -12.0
                mix_params[j, i, 1] =  random.choice([0.1,0.2,0.8,0.9])
            elif metadata[i] in ["tenor saxophone", "bassoon", "bass clarinet"]:
                #mid pan
                mix_params[j, i, 0] = -10.0
                mix_params[j, i, 1] = random.choice([0.3,0.4,0.6,0.7])
            elif metadata[i] in ["flute", "clarinet", "saxophone", "oboe", "soprano saxophone", "alto saxophone", "piccolo","dizi", "bamboo flute"]:
                #wide pan
                mix_params[j, i, 0] = -10.0
                mix_params[j, i, 1] =  random.choice([0.1,0.2,0.8,0.9])
            elif metadata[i] in ["brass section", "flute section", "clarinet section", "saxophone section", "oboe section", "piccolo section"]:
                #wide pan
                mix_params[j, i, 0] = -12.0
                mix_params[j, i, 1] =  random.choice([0.1,0.2,0.8,0.9])
            elif metadata[i] in ["trombone section", "trombone", "tuba", "baritone saxophone", "euphonium"]:
                #mid pan
                mix_params[j, i, 0] = -10.0
                mix_params[j, i, 1] = random.choice([0.3,0.4,0.6,0.7])
            elif metadata[i] in ["trumpet", "french horn", "horn section", "cornet", "trumpet section"]:
                #wide pan
                mix_params[j, i, 0] = -12.0
                mix_params[j, i, 1] =  random.choice([0.1,0.2,0.8,0.9])
            elif metadata[i] in ["clean electric guitar", "distorted electric guitar", "acoustic guitar", "lap steel guitar", "banjo","oud","guzheng","mandolin","zhongruan","yangqin","liuqin"]:
                #mid pan
                #print("guitar")
                mix_params[j, i, 0] = -10.0
                mix_params[j, i, 1] = random.choice([0.3,0.4,0.6,0.7])
            elif metadata[i] in ["Main System", "Musical Theatre"]:
                #mid pan
                mix_params[j, i, 0] = -14.0
                mix_params[j, i, 1] = random.choice([0.3,0.4,0.6,0.7])
    mix_params = mix_params.type_as(tracks)

    # generate a mix of the tracks
    mix, param_dict = mix_console(tracks, mix_params)

    return mix, param_dict
    


if __name__ == "__main__":
    
    
    dataset = MedleyDBDataset(root_dirs=["/import/c4dm-datasets/MedleyDB_V1/V1"])
    print(len(dataset))

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=2, shuffle=False)
    mix_console = mst.modules.BasicMixConsole(sample_rate=44100.0)
    for i, (track, instrument_id, genre) in enumerate(dataloader):
        print("\n\n mixing")
        print("track", track.size())
        batch_size, num_tracks, seq_len = track.size()
        
        print(instrument_id)
        
        print("genre", genre)
        mix, param_dict = knowledge_engineering_mix(track, mix_console, instrument_id, genre)
        if i==0:
            break

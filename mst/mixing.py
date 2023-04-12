# Store mixing functions here (e.g. knowledge engineering)
import torch
import json
import random
import mst.modules
from mst.modules import BasicMixConsole 

import mst.dataloaders.medley 
from mst.dataloaders.cambridge import CambridgeDataset

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
    tracks: torch.Tensor, mix_console: torch.nn.Module, instrument_id: list
):
    """Generate a mix using knowledge engineering
    """

    bs, num_tracks, seq_len = tracks.size()
    mdata = instrument_metadata(instrument_id)
    #print("mdata:", mdata)

    #BasicMixConsole
   
    mix_params = torch.full((bs, num_tracks, mix_console.num_control_params), -18.0)
    pan_params = torch.full((bs, num_tracks, 1), 0.5)
    
    for j in range(bs):
        metadata = mdata[j]
        for i in range(len(metadata)):
            
            if metadata[i]=="silence":
                mix_params[j, i, 0]= -88.0
            elif metadata[i] in ["female singer", "male singer", "male rapper", "Rap", "male speaker"]:
                #print("vocal")
                #centre pan
                mix_params[j, i, 0] = round(random.uniform(-8.5, -6.5), 1)
            elif metadata[i] == "vocalist":
                #wide pan
                mix_params[j, i, 0] = round(random.uniform(-10.0, -8.0), 1)
                pan_params[j, i, 0] = random.choice([0.1,0.2,0.8,0.9])
            elif metadata[i] in ["electric bass", "double bass"]:
                #print("bass")
                #centre pan
                mix_params[j, i, 0] = round(random.uniform(-11.5, -8.5), 1)
            elif metadata[i] in ["bass drum", "kick drum"]:
                #print("kick")
                #centre pan
                mix_params[j, i, 0] = round(random.uniform(-13.0, -11.0), 1)
            elif metadata[i] == ["snare drum","tabla"]:
                #mid pan
                mix_params[j, i, 0] = round(random.uniform(-8.0, -9.5), 1)
                pan_params[j, i, 0] = random.choice([0.4,0.6])
            elif metadata[i] in ["drum set", "drum machine"]:
                #mid pan
                mix_params[j, i, 0] = round(random.uniform(-12.0, -9.0), 1)
                pan_params[j, i, 0] = random.choice([0.3,0.4,0.6,0.7])
            elif metadata[i] in ["high hat", "toms", "timpani", "bongo", "doumbek","darbuka"]:
                #mid pan
                mix_params[j, i, 0] = round(random.uniform(-12.0, -9.0), 1)
                pan_params[j, i, 0] = random.choice([0.3,0.4,0.6,0.7])
            elif metadata[i] in ["cymbal", "glockenspiel", "claps", "vibraphone", "auxiliary percussion", "chimes", "gong"]:
                #wide pan
                mix_params[j, i, 0] = round(random.uniform(-13.0, -10.0), 1)
                pan_params[j, i, 0] = random.choice([0.1,0.2,0.8,0.9])
            elif metadata[i] in ["guiro", "cowbell"]:
                #extreme pan
                mix_params[j, i, 0] = round(random.uniform(-15.0, -13.0), 1)
                pan_params[j, i, 0] = random.choice([0.0,1.0])
            elif metadata[i] in ["sleigh bells", "shaker", "tambourine", "cabasa", "castanet", "scratches","gu"]:
                #extreme pan
                mix_params[j, i, 0] = round(random.uniform(-15.0, -13.0), 1)
                pan_params[j, i, 0] = random.choice([0.0,1.0])
            elif metadata[i] in ["fx/processed sound", "sampler", "synthesizer", "harmonica"]:
                #mid pan
                mix_params[j, i, 0] = round(random.uniform(-13.0, -10.0), 1)
                pan_params[j, i, 0] = random.choice([0.3,0.4,0.6,0.7])
            elif metadata[i] in ["electric piano", "accordion", "harpsichord", "organ", "piano", "tack piano", "melodica"]:
                #print("piano")
                #mid pan
                mix_params[j, i, 0] = round(random.uniform(-9.0, -7.0), 1)
                pan_params[j, i, 0] = random.choice([0.3,0.4,0.6,0.7])
            elif metadata[i] in ["cello", "double bass"]:
                #print("cello")
                #mid pan
                mix_params[j, i, 0] = round(random.uniform(-12.0, -9.0), 1)
                pan_params[j, i, 0] = random.choice([0.3,0.4,0.6,0.7])
            elif metadata[i] in ["violin", "viola", "string section", "violin section", "cello section", "viola section", "harp","erhu"]:
                #print("strings")
                #wide pan
                mix_params[j, i, 0] = round(random.uniform(-13.0, -11.0), 1)
                pan_params[j, i, 0] =  random.choice([0.1,0.2,0.8,0.9])
            elif metadata[i] in ["tenor saxophone", "bassoon", "bass clarinet"]:
                #mid pan
                mix_params[j, i, 0] = round(random.uniform(-11.0, -9.0), 1)
                pan_params[j, i, 0] = random.choice([0.3,0.4,0.6,0.7])
            elif metadata[i] in ["flute", "clarinet", "saxophone", "oboe", "soprano saxophone", "alto saxophone", "piccolo","dizi", "bamboo flute"]:
                #wide pan
                mix_params[j, i, 0] = round(random.uniform(-11.0, -9.0), 1)
                pan_params[j, i, 0]=  random.choice([0.1,0.2,0.8,0.9])
            elif metadata[i] in ["brass section", "flute section", "clarinet section", "saxophone section", "oboe section", "piccolo section"]:
                #wide pan
                mix_params[j, i, 0] = round(random.uniform(-13.0, -11.0), 1)
                pan_params[j, i, 0] =  random.choice([0.1,0.2,0.8,0.9])
            elif metadata[i] in ["trombone section", "trombone", "tuba", "baritone saxophone", "euphonium"]:
                #mid pan
                mix_params[j, i, 0] = round(random.uniform(-11.0, -9.0), 1)
                pan_params[j, i, 0] = random.choice([0.3,0.4,0.6,0.7])
            elif metadata[i] in ["trumpet", "french horn", "horn section", "cornet", "trumpet section"]:
                #wide pan
                mix_params[j, i, 0] = round(random.uniform(-13.0, -11.0), 1)
                pan_params[j, i, 0] =  random.choice([0.1,0.2,0.8,0.9])
            elif metadata[i] in ["clean electric guitar", "distorted electric guitar", "acoustic guitar", "lap steel guitar", "banjo","oud","guzheng","mandolin","zhongruan","yangqin","liuqin"]:
                #mid pan
                #print("guitar")
                mix_params[j, i, 0] = round(random.uniform(-11.0, -9.0), 1)
                pan_params[j, i, 0] = random.choice([0.3,0.4,0.6,0.7])
            elif metadata[i] in ["Main System", "Musical Theatre"]:
                #mid pan
                mix_params[j, i, 0] = round(random.uniform(-15.0, -13.0), 1)
                pan_params[j, i, 0] = random.choice([0.3,0.4,0.6,0.7])

    if mix_console.num_control_params == 2:
        mix_params[:,:,1]= pan_params[:, :, 0]
        param_dict = {
                "input_gain": {
                    "gain_db": mix_params[:,:, 0],  # bs, num_tracks, 1
                },
                "stereo_panner": {
                    "pan": mix_params[:,:, 1],  # bs, num_tracks, 1
                },
            }
            
    #applying knowledge engineered compressor and EQ
    elif mix_console.num_control_params!= 2:
        mix_params[:,:,25] = pan_params[:, :, 0]

        # for j in range(bs):
        #     metadata = mdata[j]
        #     for i in range(len(metadata)):
                
        #         if metadata[i]==["silence","noise"]:
        #             mix_params[j, i, 0]= -88.0



                    
        #         elif metadata[i] in ["female singer", "male singer", "male rapper", "Rap", "male speaker","vox","vocals"]:
        #             #print("vocal")
        #             #centre pan
        #             mix_params[j, i, 0] = round(random.uniform(-8.5, -6.5), 1)
        #         elif metadata[i] == ["vocalist", "backingvox","chorus","vocoder","choir"]:
        #             #wide pan
        #             mix_params[j, i, 0] = round(random.uniform(-10.0, -8.0), 1)
        #             mix_params[j, i, 25] = random.choice([0.1,0.2,0.8,0.9])
        #         elif metadata[i] in ["electric bass", "double bass"]:
        #             #print("bass")
        #             #centre pan
        #             mix_params[j, i, 0] = round(random.uniform(-11.5, -8.5), 1)
        #         elif metadata[i] in ["bass drum", "kick drum","kick"]:
        #             #print("kick")
        #             #centre pan
        #             mix_params[j, i, 0] = round(random.uniform(-13.0, -11.0), 1)
        #         elif metadata[i] == ["snare drum","tabla","snare"]:
        #             #mid pan
        #             mix_params[j, i, 0] = round(random.uniform(-8.0, -9.5), 1)
        #             mix_params[j, i, 25] = random.choice([0.4,0.6])
        #         elif metadata[i] in ["drum set", "drum machine","overhead","drum","percussion","loop"]:
        #             #mid pan
        #             mix_params[j, i, 0] = round(random.uniform(-12.0, -9.0), 1)
        #             mix_params[j, i, 25] = random.choice([0.3,0.4,0.6,0.7])
        #         elif metadata[i] in ["high hat","hi hat", "toms", "timpani", "bongo", "tom","doumbek","darbuka","bongos","cajon","conga"]:
        #             #mid pan
        #             mix_params[j, i, 0] = round(random.uniform(-12.0, -9.0), 1)
        #             mix_params[j, i, 1] = random.choice([0.3,0.4,0.6,0.7])
        #         elif metadata[i] in ["cymbal", "glockenspiel", "claps","clap", "vibraphone", "auxiliary percussion", "chimes", "gong","marimba"]:
        #             #wide pan
        #             mix_params[j, i, 0] = round(random.uniform(-13.0, -10.0), 1)
        #             mix_params[j, i, 25] = random.choice([0.1,0.2,0.8,0.9])
        #         elif metadata[i] in ["guiro", "cowbell","triangle","hit","crash","stick","bell"]:
        #             #extreme pan
        #             mix_params[j, i, 0] = round(random.uniform(-15.0, -13.0), 1)
        #             mix_params[j, i, 25] = random.choice([0.0,1.0])
        #         elif metadata[i] in ["sleigh bells", "shaker", "tambourine", "cabasa", "castanet", "scratches","gu"]:
        #             #extreme pan
        #             mix_params[j, i, 0] = round(random.uniform(-15.0, -13.0), 1)
        #             mix_params[j, i, 25] = random.choice([0.0,1.0])
        #         elif metadata[i] in ["fx/processed sound", "sampler", "synthesizer", "harmonica","synth"]:
        #             #mid pan
        #             mix_params[j, i, 0] = round(random.uniform(-13.0, -10.0), 1)
        #             mix_params[j, i, 25] = random.choice([0.3,0.4,0.6,0.7])
        #         elif metadata[i] in ["electric piano","xylophone", "accordion", "harpsichord","keys", "organ", "rhodes","piano", "tack piano", "melodica"]:
        #             #print("piano")
        #             #mid pan
        #             mix_params[j, i, 0] = round(random.uniform(-9.0, -7.0), 1)
        #             mix_params[j, i, 25] = random.choice([0.3,0.4,0.6,0.7])
        #         elif metadata[i] in ["cello", "double bass","bass"]:
        #             #print("cello")
        #             #mid pan
        #             mix_params[j, i, 0] = round(random.uniform(-12.0, -9.0), 1)
        #             mix_params[j, i, 25] = random.choice([0.3,0.4,0.6,0.7])
        #         elif metadata[i] in ["violin", "viola", "string section", "quartet","violin section", "string","cello section", "viola section", "harp","erhu"]:
        #             #print("strings")
        #             #wide pan
        #             mix_params[j, i, 0] = round(random.uniform(-13.0, -11.0), 1)
        #             mix_params[j, i, 25] =  random.choice([0.1,0.2,0.8,0.9])
        #         elif metadata[i] in ["tenor saxophone", "bassoon", "bass clarinet","tenor","alto","soprano","saxophone","bagpipe"]:
        #             #mid pan
        #             mix_params[j, i, 0] = round(random.uniform(-11.0, -9.0), 1)
        #             mix_params[j, i, 25] = random.choice([0.3,0.4,0.6,0.7])
        #         elif metadata[i] in ["flute", "clarinet","woodwind","saxophone", "oboe", "soprano saxophone", "alto saxophone", "piccolo","dizi", "bamboo flute"]:
        #             #wide pan
        #             mix_params[j, i, 0] = round(random.uniform(-11.0, -9.0), 1)
        #             mix_params[j, i, 25] =  random.choice([0.1,0.2,0.8,0.9])
        #         elif metadata[i] in ["brass section", "flute section", "clarinet section", "saxophone section", "oboe section", "piccolo section"]:
        #             #wide pan
        #             mix_params[j, i, 0] = round(random.uniform(-13.0, -11.0), 1)
        #             mix_params[j, i, 25] =  random.choice([0.1,0.2,0.8,0.9])
        #         elif metadata[i] in ["trombone section", "trombone", "tuba", "baritone saxophone", "euphonium","brass"]:
        #             #mid pan
        #             mix_params[j, i, 0] = round(random.uniform(-11.0, -9.0), 1)
        #             mix_params[j, i, 1] = random.choice([0.3,0.4,0.6,0.7])
        #         elif metadata[i] in ["trumpet", "french horn", "horn section", "cornet", "trumpet section","horn"]:
        #             #wide pan
        #             mix_params[j, i, 0] = round(random.uniform(-13.0, -11.0), 1)
        #             mix_params[j, i, 25] =  random.choice([0.1,0.2,0.8,0.9])
        #         elif metadata[i] in ["clean electric guitar", "distorted electric guitar", "ukulele","acoustic guitar", "fiddle","lap steel guitar", "banjo","oud","guzheng","mandolin","zhongruan","yangqin","liuqin","sitar","ukelele","guitar"]:
        #             #mid pan
        #             #print("guitar")
        #             mix_params[j, i, 0] = round(random.uniform(-11.0, -9.0), 1)
        #             mix_params[j, i, 25] = random.choice([0.3,0.4,0.6,0.7])
        #         elif metadata[i] in ["Main System", "Musical Theatre","roommic","misc","sfx"]:
        #             #mid pan
        #             mix_params[j, i, 0] = round(random.uniform(-15.0, -13.0), 1)
        #             mix_params[j, i, 25] = random.choice([0.3,0.4,0.6,0.7])


   
    mix = mix_console.forward_mix_console(tracks, param_dict)
    # peak normalize the mix
    mix /= mix.abs().max().clamp(min=1e-8)

    return mix, param_dict
    


# if __name__ == "__main__":
    
    
#     dataset = CambridgeDataset(root_dirs=["/import/c4dm-multitrack-private/C4DM Multitrack Collection/mixing-secrets"])
#     print(len(dataset))

#     dataloader = torch.utils.data.DataLoader(dataset, batch_size=2, shuffle=False)
#     mix_console = mst.modules.BasicMixConsole(sample_rate=44100.0)
#     for i, (track, instrument_id) in enumerate(dataloader):
#         print("\n\n mixing")
#         print("track", track.size())
#         batch_size, num_tracks, seq_len = track.size()
        
#         print(instrument_id)
        
        
#         mix, param_dict = knowledge_engineering_mix(track, mix_console, instrument_id)
#         print("mix", mix.size())
#         if i==0:
#             break

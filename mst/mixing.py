# Store mixing functions here (e.g. knowledge engineering)
import torch
import json
import random
import numpy as np
import mst.modules
from mst.modules import BasicMixConsole, AdvancedMixConsole

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
        #print(iid)
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
    tracks: torch.Tensor, mix_console: torch.nn.Module, instrument_id: list, stereo_id: list
):
    """Generate a mix using knowledge engineering
    """

    bs, num_tracks, seq_len = tracks.size()
    mdata = instrument_metadata(instrument_id)
    #print("mdata:", mdata)

    #BasicMixConsole
    
    
    mix_params = torch.full((bs, num_tracks, mix_console.num_control_params), -18.0)
    pan_params = torch.full((bs, num_tracks, 1), 0.5)

    if mix_console.num_control_params > 2:
        advance_console = True
        eq = torch.tensor([10,100,1])
        compressor = torch.tensor([10,1,10, 10, 1, 1])

        eq_lowpass_params = torch.full((bs, num_tracks, 3),0)
        eq_lowpass_params[:,:,1] = 50
        eq_lowpass_params[:,:,2] = 0
        
        eq_bandpass1_params = torch.full((bs, num_tracks, 3),0)
        eq_bandpass1_params[:,:,1] = 200
        eq_bandpass1_params[:,:,2] = 0
        
        eq_bandpass2_params = torch.full((bs, num_tracks, 3),0)
        eq_bandpass2_params[:,:,1] = 500
        eq_bandpass2_params[:,:,2] = 0
        
        eq_bandpass3_params = torch.full((bs, num_tracks, 3),0)
        eq_bandpass3_params[:,:,1] = 1000
        eq_bandpass3_params[:,:,2] = 0
    
        eq_bandpass4_params = torch.full((bs, num_tracks, 3),0)
        eq_bandpass4_params[:,:,1] = 5000
        eq_bandpass4_params[:,:,2] = 0
        
        eq_highpass_params = torch.full((bs, num_tracks, 3),0)
        eq_highpass_params[:,:,1] = 10000
        eq_highpass_params[:,:,2] = 0
    
        comp_params = torch.full((bs, num_tracks, 6),-5)
        comp_params[:,:,1] = 1
        comp_params[:,:,2] = 0
        comp_params[:,:,3] = 0
        comp_params[:,:,4] = 0
        comp_params[:,:,5] = 0

        # dist_params = torch.full((bs, num_tracks, 2),(0,0))
        # reverb_params = torch.full((bs, num_tracks, 2),(0,0))

    skip = False
    for j in range(bs):
        metadata = mdata[j]
        stereo_info = stereo_id[j,:]
        
        for i in range(len(metadata)):
            #print(stereo_info[i])
            if stereo_info[i] == 1:
                if i == num_tracks-1:
                    #print("last track")
                    skip =False
                else:
                    #print("stereo")
                    skip = True
            
            if metadata[i]=="silence":
                mix_params[j, i, 0]= -88.0

            elif metadata[i] in ["female singer", "male singer", "male rapper", "Rap", "male speaker"]:
                #print("vocal")
                #centre pan
                mix_params[j, i, 0] = round(random.uniform(-8.5, -6.5), 1)
                if advance_console:
                    eq_lowpass_params[j, i, :] = torch.tensor([-3, 100, 0.5]) + random.uniform(-1, 1) * eq
                    eq_bandpass1_params[j, i, :] = torch.tensor([0, 250, 1]) + random.uniform(-1, 1) * eq
                    eq_bandpass2_params[j, i, :] = torch.tensor([1, 500, 1]) + random.uniform(-1, 1) * eq
                    eq_bandpass3_params[j, i, :] = torch.tensor([1, 1000, 1]) + random.uniform(-1, 1) * eq
                    eq_bandpass4_params[j, i, :] = torch.tensor([0, 5000, 1]) + random.uniform(-1, 1) * eq
                    eq_highpass_params[j, i, :] = torch.tensor([-3, 10000, 0.5]) + random.uniform(-1, 1) * eq
                    comp_params[j, i, :] = torch.tensor([-16, 2, 10, 10, 5, 3])+ random.uniform(-1, 1) * compressor
                
            elif metadata[i] == "vocalist":
                #wide pan
                mix_params[j, i, 0] = round(random.uniform(-10.0, -8.0), 1)
                pan_params[j, i, 0] = random.choice([0.1,0.2,0.8,0.9])
                if advance_console:
                    eq_lowpass_params[j, i, :] = torch.tensor([-3, 100, 0.5]) + random.uniform(-1, 1) * eq
                    eq_bandpass1_params[j, i, :] = torch.tensor([1, 250, 1]) + random.uniform(-1, 1) * eq
                    eq_bandpass2_params[j, i, :] = torch.tensor([0, 500, 1]) + random.uniform(-1, 1) * eq
                    eq_bandpass3_params[j, i, :] = torch.tensor([0, 1000, 1]) + random.uniform(-1, 1) * eq
                    eq_bandpass4_params[j, i, :] = torch.tensor([0, 5000, 1]) + random.uniform(-1, 1) * eq
                    eq_highpass_params[j, i, :] = torch.tensor([-3, 10000, 0.5]) + random.uniform(-1, 1) * eq
                    comp_params[j, i, :] = torch.tensor([-16, 2, 10, 10, 5, 3])+ random.uniform(-1, 1) * compressor

            elif metadata[i] in ["electric bass", "double bass"]:
                #print("bass")
                #centre pan
                mix_params[j, i, 0] = round(random.uniform(-11.5, -8.5), 1)
                if advance_console:
                    eq_lowpass_params[j, i, :] = torch.tensor([-2, 50, 1.5]) + random.uniform(-1, 1) * eq
                    eq_bandpass1_params[j, i, :] = torch.tensor([3, 100, 2]) + random.uniform(-1, 1) * eq
                    eq_bandpass2_params[j, i, :] = torch.tensor([0, 500, 1]) + random.uniform(-1, 1) * eq
                    eq_bandpass3_params[j, i, :] = torch.tensor([3, 1000, 1]) + random.uniform(-1, 1) * eq
                    comp_params[j, i, :] = torch.tensor([-25, 3, 10, 10, 10, 4]) + random.uniform(-1, 1) * compressor
                    if random.uniform(0,1) > 0.5:
                        eq_bandpass4_params[j, i, :] = torch.tensor([-3, 3000, 1]) + random.uniform(-1, 1) * eq
                        eq_highpass_params[j, i, :] = torch.tensor([0, 10000, 0.5]) + random.uniform(-1, 1) * eq
                
                
            elif metadata[i] in ["bass drum", "kick drum"]:
                #print("kick")
                #centre pan
                mix_params[j, i, 0] = round(random.uniform(-13.0, -11.0), 1)
                if advance_console:
                    eq_lowpass_params[j, i, :] = torch.tensor([2, 50, 2]) + random.uniform(-1, 1) * eq
                    eq_bandpass1_params[j, i, :] = [1, 100, 1] + random.uniform(-1, 1) * eq
                    eq_bandpass2_params[j, i, :] = [-3, 500, 1] + random.uniform(-1, 1) * eq
                    comp_params[j, i, :] = [-20, 4, 10, 10, 5, 4] + random.uniform(-1, 1) * compressor
                    if random.uniform(0,1) > 0.5:
                        eq_bandpass3_params[j, i, :] = [0, 2000, 1] + random.uniform(-1, 1) * eq
                        eq_bandpass4_params[j, i, :] = [-6, 5000, 1] + random.uniform(-1, 1) * eq
                        eq_highpass_params[j, i, :] = [0, 10000, 0.5] + random.uniform(-1, 1) * eq
                

            elif metadata[i] == ["snare drum","tabla"]:
                #mid pan
                mix_params[j, i, 0] = round(random.uniform(-8.0, -9.5), 1)
                pan_params[j, i, 0] = random.choice([0.4,0.6])
                if advance_console:
                    eq_lowpass_params[j, i, :] = torch.tensor([-1, 100, 0.5])+ random.uniform(-1, 1) * eq
                    eq_bandpass1_params[j, i, :] = torch.tensor([3, 250, 1])+ random.uniform(-1, 1) * eq
                    eq_bandpass2_params[j, i, :] = torch.tensor([1, 500, 0.5])+ random.uniform(-1, 1) * eq
                    comp_params[j, i, :] = torch.tensor([-18, 3, 10, 10, 5, 3])+ random.uniform(-1, 1) * compressor
                    if random.uniform(0,1) > 0.5:
                        eq_bandpass3_params[j, i, :] = torch.tensor([-3, 3000, 1])+ random.uniform(-1, 1) * eq
                        eq_bandpass4_params[j, i, :] = torch.tensor([-6, 5000, 1])+ random.uniform(-1, 1) * eq
                        eq_highpass_params[j, i, :] = torch.tensor([0, 10000, 0.5])+ random.uniform(-1, 1) * eq
                

            elif metadata[i] in ["drum set", "drum machine"]:
                #mid pan
                mix_params[j, i, 0] = round(random.uniform(-12.0, -9.0), 1)
                pan_params[j, i, 0] = random.choice([0.3,0.4,0.6,0.7])
                if advance_console:
                    if random.uniform(0,1) > 0.5:
                        eq_lowpass_params[j, i, :] = torch.tensor([-6, 80, 1]) + random.uniform(-1, 1) * eq
                        eq_bandpass1_params[j, i, :] = torch.tensor([0, 250, 1])+ random.uniform(-1, 1) * eq
                        eq_bandpass2_params[j, i, :] = torch.tensor([3, 500, 1])+ random.uniform(-1, 1) * eq
                        eq_bandpass3_params[j, i, :] = torch.tensor([-3, 1000, 1]) + random.uniform(-1, 1) * eq
                        eq_bandpass4_params[j, i, :] = torch.tensor([-3, 4000, 1]) + random.uniform(-1, 1) * eq
                        eq_highpass_params[j, i, :] =torch.tensor([-6, 8000, 1])+ random.uniform(-1, 1) * eq
                        comp_params[j, i, :] = torch.tensor([-18, 4, 15, 15, 5, 2]) + random.uniform(-1, 1) * compressor

            elif metadata[i] in ["high hat", "toms", "timpani", "bongo", "doumbek","darbuka"]:
                #mid pan
                mix_params[j, i, 0] = round(random.uniform(-12.0, -9.0), 1)
                pan_params[j, i, 0] = random.choice([0.3,0.4,0.6,0.7])
                if advance_console:
                    eq_lowpass_params[j, i, :] = torch.tensor([-6, 300, 2]) + random.uniform(-1, 1) * eq
                    if random.uniform(0,1) > 0.5:
                        eq_bandpass1_params[j, i, :] = torch.tensor([0, 1000, 1]) + random.uniform(-1, 1) * eq
                        eq_bandpass2_params[j, i, :] = torch.tensor([0, 2000, 1])+ random.uniform(-1, 1) * eq
                        eq_bandpass3_params[j, i, :] = torch.tensor([0, 4000, 1]) + random.uniform(-1, 1) * eq
                        eq_bandpass4_params[j, i, :] = torch.tensor([0, 8000, 1]) + random.uniform(-1, 1) * eq
                        eq_highpass_params[j, i, :] = torch.tensor([0, 10000, 1]) + random.uniform(-1, 1) * eq
                        comp_params[j, i, :] = torch.tensor([-16, 2, 15, 15, 5, 2]) + random.uniform(-1, 1) * compressor

            elif metadata[i] in ["cymbal", "glockenspiel", "claps", "vibraphone", "auxiliary percussion", "chimes", "gong"]:
                #wide pan
                mix_params[j, i, 0] = round(random.uniform(-13.0, -10.0), 1)
                pan_params[j, i, 0] = random.choice([0.1,0.2,0.8,0.9])
                if advance_console:
                    eq_lowpass_params[j, i, :] = torch.tensor([-6, 250, 2]) + random.uniform(-1, 1) * eq
                    eq_bandpass1_params[j, i, :] = torch.tensor([-3, 500, 1]) + random.uniform(-1, 1) * eq
                    eq_bandpass2_params[j, i, :] = torch.tensor([-2, 1000, 1]) + random.uniform(-1, 1) * eq
                    eq_bandpass3_params[j, i, :] = torch.tensor([-1, 2000, 1]) + random.uniform(-1, 1) * eq
                    eq_bandpass4_params[j, i, :] = torch.tensor([-1, 4000, 1]) + random.uniform(-1, 1) * eq
                    eq_highpass_params[j, i, :] = torch.tensor([-1, 8000, 1]) + random.uniform(-1, 1) * eq
                    comp_params[j, i, :] = torch.tensor([-20, 2, 10, 10, 5, 4]) + random.uniform(-1, 1) * compressor


            elif metadata[i] in ["guiro", "cowbell"]:
                #extreme pan
                mix_params[j, i, 0] = round(random.uniform(-15.0, -13.0), 1)
                pan_params[j, i, 0] = random.choice([0.0,1.0])
                if advance_console:
                    eq_lowpass_params[j, i, :] = torch.tensor([-6, 250, 1])+ random.uniform(-1, 1) * eq
                    eq_bandpass1_params[j, i, :] = torch.tensor([-3, 500, 1]) + random.uniform(-1, 1) * eq
                    eq_bandpass2_params[j, i, :] = torch.tensor([-2, 1000, 1]) + random.uniform(-1, 1) * eq
                    eq_bandpass3_params[j, i, :] = torch.tensor([-1, 2000, 1]) + random.uniform(-1, 1) * eq
                    eq_bandpass4_params[j, i, :] = torch.tensor([-1, 4000, 1]) + random.uniform(-1, 1) * eq
                    eq_highpass_params[j, i, :] = torch.tensor([-1, 8000, 1]) + random.uniform(-1, 1) * eq
                    comp_params[j, i, :] = torch.tensor([-5, 1, 0, 0, 0, 0]) + random.uniform(-1, 1) * compressor

            elif metadata[i] in ["sleigh bells", "shaker", "tambourine", "cabasa", "castanet", "scratches","gu"]:
                #extreme pan
                mix_params[j, i, 0] = round(random.uniform(-15.0, -13.0), 1)
                pan_params[j, i, 0] = random.choice([0.0,1.0])
                if advance_console:
                    eq_lowpass_params[j, i, :] = torch.tensor([-3, 200, 0.5]) + random.uniform(-1, 1) * eq
                    eq_bandpass1_params[j, i, :] = torch.tensor([-3, 500, 1])+ random.uniform(-1, 1) * eq
                    eq_bandpass2_params[j, i, :] = torch.tensor([0, 1000, 1])+ random.uniform(-1, 1) * eq
                    eq_bandpass3_params[j, i, :] = torch.tensor([0, 2000, 1])+ random.uniform(-1, 1) * eq
                    eq_bandpass4_params[j, i, :] = torch.tensor([0, 4000, 1])+ random.uniform(-1, 1) * eq
                    eq_highpass_params[j, i, :] = torch.tensor([0, 8000, 0.5])+ random.uniform(-1, 1) * eq
                    comp_params[j, i, :] = torch.tensor([-16, 3, 15, 15, 5, 3]) + random.uniform(-1, 1) * compressor

            elif metadata[i] in ["fx/processed sound", "sampler", "synthesizer", "harmonica"]:
                #mid pan
                mix_params[j, i, 0] = round(random.uniform(-13.0, -10.0), 1)
                pan_params[j, i, 0] = random.choice([0.3,0.4,0.6,0.7])
                if advance_console:
                    eq_lowpass_params[j, i, :] = torch.tensor([-6, 80, 1])+ random.uniform(-1, 1) * eq
                    eq_bandpass1_params[j, i, :] = torch.tensor([-3, 200, 1]) + random.uniform(-1, 1) * eq
                    eq_bandpass2_params[j, i, :] = torch.tensor([3, 1000, 1]) + random.uniform(-1, 1) * eq
                    eq_bandpass3_params[j, i, :] = torch.tensor([3, 2000, 1]) + random.uniform(-1, 1) * eq
                    eq_bandpass4_params[j, i, :] = torch.tensor([0, 4000, 1])+ random.uniform(-1, 1) * eq
                    eq_highpass_params[j, i, :] = torch.tensor([-1, 8000, 1])+ random.uniform(-1, 1) * eq
                    comp_params[j, i, :] = torch.tensor([-22, 3, 20, 20, 5, 3] )+ random.uniform(-1, 1) * compressor

            elif metadata[i] in ["electric piano", "accordion", "harpsichord", "organ", "piano", "tack piano", "melodica"]:
                #print("piano")
                #mid pan
                mix_params[j, i, 0] = round(random.uniform(-9.0, -7.0), 1)
                pan_params[j, i, 0] = random.choice([0.3,0.4,0.6,0.7])
                if advance_console:
                    eq_lowpass_params[j, i, :] = torch.tensor([-3, 100, 0.5]) + random.uniform(-1, 1) * eq
                    eq_bandpass1_params[j, i, :] = torch.tensor([0, 200, 1]) + random.uniform(-1, 1) * eq
                    eq_bandpass2_params[j, i, :] = torch.tensor([0, 1000, 1]) + random.uniform(-1, 1) * eq
                    eq_bandpass3_params[j, i, :] = torch.tensor([3, 3000, 1]) + random.uniform(-1, 1) * eq
                    eq_bandpass4_params[j, i, :] = torch.tensor([-3, 8000, 1])+ random.uniform(-1, 1) * eq
                    eq_highpass_params[j, i, :] = torch.tensor([-3, 10000, 0.5])+ random.uniform(-1, 1) * eq
                    comp_params[j, i, :] = torch.tensor([-20, 2, 15, 15, 5, 3] )+ random.uniform(-1, 1) * compressor

            elif metadata[i] in ["cello", "double bass"]:
                #print("cello")
                #mid pan
                mix_params[j, i, 0] = round(random.uniform(-12.0, -9.0), 1)
                pan_params[j, i, 0] = random.choice([0.3,0.4,0.6,0.7])
                if advance_console:
                    eq_lowpass_params[j, i, :] = torch.tensor([-3, 100, 0.5]) + random.uniform(-1, 1) * eq
                    eq_bandpass1_params[j, i, :] = torch.tensor([1, 250, 1])+ random.uniform(-1, 1) * eq
                    eq_bandpass2_params[j, i, :] = torch.tensor([0, 500, 1])+ random.uniform(-1, 1) * eq
                    eq_bandpass3_params[j, i, :] = torch.tensor([-2, 1000, 1])+ random.uniform(-1, 1) * eq
                    eq_bandpass4_params[j, i, :] = torch.tensor([-1, 4000, 1])+ random.uniform(-1, 1) * eq
                    eq_highpass_params[j, i, :] = torch.tensor([-3, 10000, 0.5])+ random.uniform(-1, 1) * eq
                    comp_params[j, i, :] = torch.tensor([-22, 3, 15, 15, 5, 3]) + random.uniform(-1, 1) * compressor

            elif metadata[i] in ["violin", "viola", "string section", "violin section", "cello section", "viola section", "harp","erhu"]:
                #print("strings")
                #wide pan
                mix_params[j, i, 0] = round(random.uniform(-13.0, -11.0), 1)
                pan_params[j, i, 0] =  random.choice([0.1,0.2,0.8,0.9])
                if advance_console:
                    eq_lowpass_params[j, i, :] = torch.tensor([0, 100, 0.5]) + random.uniform(-1, 1) * eq
                    eq_bandpass1_params[j, i, :] = torch.tensor([3, 500, 1])+ random.uniform(-1, 1) * eq
                    eq_bandpass2_params[j, i, :] = torch.tensor([0, 1000, 1])+ random.uniform(-1, 1) * eq
                    eq_bandpass3_params[j, i, :] = torch.tensor([-1, 2000, 1])+ random.uniform(-1, 1) * eq
                    eq_bandpass4_params[j, i, :] = torch.tensor([0, 3000, 1])+ random.uniform(-1, 1) * eq
                    eq_highpass_params[j, i, :] = torch.tensor([0, 10000, 0.5])+ random.uniform(-1, 1) * eq
                    comp_params[j, i, :] = torch.tensor([-22, 3, 25, 25, 5, 3]) + random.uniform(-1, 1) * compressor

            elif metadata[i] in ["tenor saxophone", "bassoon", "bass clarinet"]:
                #mid pan
                mix_params[j, i, 0] = round(random.uniform(-11.0, -9.0), 1)
                pan_params[j, i, 0] = random.choice([0.3,0.4,0.6,0.7])
                if advance_console:
                    eq_lowpass_params[j, i, :] = torch.tensor([-3, 100, 0.5]) + random.uniform(-1, 1) * eq
                    eq_bandpass1_params[j, i, :] = torch.tensor([1, 250, 1])+ random.uniform(-1, 1) * eq
                    eq_bandpass2_params[j, i, :] = torch.tensor([1, 500, 1])+ random.uniform(-1, 1) * eq
                    eq_bandpass3_params[j, i, :] = torch.tensor([-3, 2000, 1])+ random.uniform(-1, 1) * eq
                    eq_bandpass4_params[j, i, :] = torch.tensor([-6, 3000, 1])+ random.uniform(-1, 1) * eq
                    eq_highpass_params[j, i, :] = torch.tensor([-3, 10000, 0.5])+ random.uniform(-1, 1) * eq
                    comp_params[j, i, :] =torch.tensor([-16, 3, 20, 20, 5, 3]) + random.uniform(-1, 1) * compressor

            elif metadata[i] in ["flute", "clarinet", "saxophone", "oboe", "soprano saxophone", "alto saxophone", "piccolo","dizi", "bamboo flute"]:
                #wide pan
                mix_params[j, i, 0] = round(random.uniform(-11.0, -9.0), 1)
                pan_params[j, i, 0]=  random.choice([0.1,0.2,0.8,0.9])
                if advance_console:
                    eq_lowpass_params[j, i, :] = torch.tensor([-3, 100, 0.5]) + random.uniform(-1, 1) * eq
                    eq_bandpass1_params[j, i, :] = torch.tensor([3, 500, 1])+ random.uniform(-1, 1) * eq
                    eq_bandpass2_params[j, i, :] = torch.tensor([-1, 1000, 1])+ random.uniform(-1, 1) * eq
                    eq_bandpass3_params[j, i, :] = torch.tensor([-1, 2000, 1])+ random.uniform(-1, 1) * eq
                    eq_bandpass4_params[j, i, :] = torch.tensor([-2, 3000, 1])+ random.uniform(-1, 1) * eq
                    eq_highpass_params[j, i, :] = torch.tensor([-3, 10000, 0.5])+ random.uniform(-1, 1) * eq
                    if random.uniform(0, 1) > 0.6:
                        comp_params[j, i, :] = torch.tensor([-16, 2, 10, 10, 5, 4]) + random.uniform(-1, 1) * compressor

            elif metadata[i] in ["flute section", "clarinet section", "saxophone section", "oboe section", "piccolo section"]:
                #wide pan
                mix_params[j, i, 0] = round(random.uniform(-13.0, -11.0), 1)
                pan_params[j, i, 0] =  random.choice([0.1,0.2,0.8,0.9])
                if advance_console:
                    eq_lowpass_params[j, i, :] = torch.tensor([-3, 100, 0.5]) + random.uniform(-1, 1) * eq
                    eq_bandpass1_params[j, i, :] = torch.tensor([3, 500, 1])+ random.uniform(-1, 1) * eq
                    eq_bandpass2_params[j, i, :] = torch.tensor([-1, 1000, 1])+ random.uniform(-1, 1) * eq
                    eq_bandpass3_params[j, i, :] = torch.tensor([-3, 2000, 1])+ random.uniform(-1, 1) * eq
                    eq_bandpass4_params[j, i, :] = torch.tensor([0, 3000, 1])+ random.uniform(-1, 1) * eq
                    eq_highpass_params[j, i, :] = torch.tensor([-3, 10000, 0.5])+ random.uniform(-1, 1) * eq
                    comp_params[j, i, :] = torch.tensor([-5, 1, 0, 0, 0, 0]) + random.uniform(-1, 1) * compressor

            elif metadata[i] in ["trombone section", "trombone", "tuba", "baritone saxophone", "euphonium"]:
                #mid pan
                mix_params[j, i, 0] = round(random.uniform(-11.0, -9.0), 1)
                pan_params[j, i, 0] = random.choice([0.3,0.4,0.6,0.7])
                if advance_console:
                    eq_lowpass_params[j, i, :] = torch.tensor([-3, 100, 0.5]) + random.uniform(-1, 1) * eq
                    eq_bandpass1_params[j, i, :] = torch.tensor([2, 250, 1])+ random.uniform(-1, 1) * eq
                    eq_bandpass2_params[j, i, :] = torch.tensor([3, 500, 1])+ random.uniform(-1, 1) * eq
                    eq_bandpass3_params[j, i, :] = torch.tensor([-2, 2000, 1])+ random.uniform(-1, 1) * eq
                    eq_bandpass4_params[j, i, :] = torch.tensor([-3, 3000, 1])+ random.uniform(-1, 1) * eq
                    eq_highpass_params[j, i, :] = torch.tensor([-3, 10000, 0.5])+ random.uniform(-1, 1) * eq
                    if random.uniform(0, 1) > 0.6:
                        comp_params[j, i, :] = torch.tensor([-18, 3, 25, 25, 5, 3]) + random.uniform(-1, 1) * compressor
                
            elif metadata[i] in ["trumpet", "french horn", "horn section", "cornet", "trumpet section","brass section"]:
                #wide pan
                mix_params[j, i, 0] = round(random.uniform(-13.0, -11.0), 1)
                pan_params[j, i, 0] =  random.choice([0.1,0.2,0.8,0.9])
                if advance_console:
                    eq_lowpass_params[j, i, :] = torch.tensor([-3,100,0.5]) + random.uniform(-1, 1) * eq
                    eq_bandpass1_params[j, i, :] = torch.tensor([3, 500, 1])+ random.uniform(-1, 1) * eq
                    eq_bandpass2_params[j, i, :] = torch.tensor([0, 1000, 1])+ random.uniform(-1, 1) * eq
                    eq_bandpass3_params[j, i, :] = torch.tensor([-1, 2000, 1])+ random.uniform(-1, 1) * eq
                    eq_bandpass4_params[j, i, :] = torch.tensor([-2, 3000, 1])+ random.uniform(-1, 1) * eq
                    eq_highpass_params[j, i, :] = torch.tensor([0, 10000, 0.5])+ random.uniform(-1, 1) * eq
                    comp_params[j, i, :] = torch.tensor([-5, 1, 0, 0, 0, 0]) + random.uniform(-1, 1) * compressor

            elif metadata[i] in ["clean electric guitar", "distorted electric guitar", "acoustic guitar", "lap steel guitar", "banjo","oud","guzheng","mandolin","zhongruan","yangqin","liuqin"]:
                #mid pan
                #print("guitar")
                mix_params[j, i, 0] = round(random.uniform(-11.0, -9.0), 1)
                pan_params[j, i, 0] = random.choice([0.3,0.4,0.6,0.7])
                if advance_console:
                    eq_lowpass_params[j, i, :] = torch.tensor([-3, 100, 0.5]) + random.uniform(-1, 1) * eq
                    eq_bandpass1_params[j, i, :] = torch.tensor([0, 200, 1])+ random.uniform(-1, 1) * eq
                    eq_bandpass2_params[j, i, :] = torch.tensor([-1, 1000, 1])+ random.uniform(-1, 1) * eq
                    eq_bandpass3_params[j, i, :] = torch.tensor([3, 3000, 1])+ random.uniform(-1, 1) * eq
                    eq_bandpass4_params[j, i, :] = torch.tensor([-3, 8000, 1])+ random.uniform(-1, 1) * eq
                    eq_highpass_params[j, i, :] = torch.tensor([-3, 10000, 0.5])+ random.uniform(-1, 1) * eq
                    comp_params[j, i, :] = torch.tensor([-20, 2, 10, 10, 5, 3])+ random.uniform(-1, 1) * compressor

            elif metadata[i] in ["Main System", "Musical Theatre"]:
                #mid pan
                mix_params[j, i, 0] = round(random.uniform(-15.0, -13.0), 1) 
                pan_params[j, i, 0] = random.choice([0.3,0.4,0.6,0.7])
                if advance_console:
                    if random.uniform(0,1)>0.8:
                        eq_lowpass_params [j, i, :] +=  random.uniform(-1, 1) * eq
                        eq_bandpass1_params[j, i, :] +=  random.uniform(-1, 1) * eq
                        eq_bandpass2_params[j, i, :] +=  random.uniform(-1, 1) * eq
                        eq_bandpass3_params[j, i, :] +=  random.uniform(-1, 1) * eq
                        eq_bandpass4_params[j, i, :] +=  random.uniform(-1, 1) * eq
                        eq_highpass_params[j, i, :] +=  random.uniform(-1, 1) * eq
                        comp_params[j, i, :] +=  random.uniform(-1, 1) * compressor

            if skip:                
                mix_params[j, i+1, :] = mix_params[j, i, :]
                pan_params[j, i+1, :] = pan_params[j, i, :]
                eq_lowpass_params[j, i+1, :] = eq_lowpass_params[j, i, :]
                eq_bandpass1_params[j, i+1, :] = eq_bandpass1_params[j, i, :]
                eq_bandpass2_params[j, i+1, :] = eq_bandpass2_params[j, i, :]
                eq_bandpass3_params[j, i+1, :] = eq_bandpass3_params[j, i, :]
                eq_bandpass4_params[j, i+1, :] = eq_bandpass4_params[j, i, :]
                eq_highpass_params[j, i+1, :] = eq_highpass_params[j, i, :]
                comp_params[j, i+1, :] = comp_params[j, i, :]
                
                i = i+1
                skip = False

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
        # print(eq_lowpass_params.shape)
        # print(mix_params[:,:,1:4].shape)
        mix_params[:,:,25] = pan_params[:, :, 0]
        mix_params[:,:,1:4]=eq_lowpass_params
        mix_params[:,:,4:7]=eq_bandpass1_params
        mix_params[:,:,7:10]=eq_bandpass2_params
        mix_params[:,:,10:13]=eq_bandpass3_params
        mix_params[:,:,13:16]=eq_bandpass4_params
        mix_params[:,:,16:19]=eq_highpass_params
        mix_params[:,:,19:25]=comp_params

        param_dict = param_dict = {
            "input_gain": {
                "gain_db": mix_params[..., 0],
            },
            "parametric_eq": {
                "low_shelf_gain_db": mix_params[..., 1],
                "low_shelf_cutoff_freq": mix_params[..., 2],
                "low_shelf_q_factor": mix_params[..., 3],
                "first_band_gain_db": mix_params[..., 4],
                "first_band_cutoff_freq": mix_params[..., 5],
                "first_band_q_factor": mix_params[..., 6],
                "second_band_gain_db": mix_params[..., 7],
                "second_band_cutoff_freq": mix_params[..., 8],
                "second_band_q_factor": mix_params[..., 9],
                "third_band_gain_db": mix_params[..., 10],
                "third_band_cutoff_freq": mix_params[..., 11],
                "third_band_q_factor": mix_params[..., 12],
                "fourth_band_gain_db": mix_params[..., 13],
                "fourth_band_cutoff_freq": mix_params[..., 14],
                "fourth_band_q_factor": mix_params[..., 15],
                "high_shelf_gain_db": mix_params[..., 16],
                "high_shelf_cutoff_freq": mix_params[..., 17],
                "high_shelf_q_factor": mix_params[..., 18],
            },
            #release and attack time must be the same
            "compressor": {
                "threshold_db": mix_params[..., 19],
                "ratio": mix_params[..., 20],
                "attack_ms": mix_params[..., 21], 
                "release_ms": mix_params[..., 22],
                "knee_db": mix_params[..., 23],
                "makeup_gain_db": mix_params[..., 24],
            },
            "stereo_panner": {
                "pan": mix_params[..., 25],
            },
        }

        
    mix = mix_console.forward_mix_console(tracks, param_dict)
    # peak normalize the mix
    mix /= mix.abs().max().clamp(min=1e-8)

    return mix, param_dict
    


if __name__ == "__main__":
    
    
    #dataset = CambridgeDataset(root_dirs=["/import/c4dm-multitrack-private/C4DM Multitrack Collection/mixing-secrets"])
    dataset = MedleyDBDataset(root_dirs=["/import/c4dm-datasets/MedleyDB_V1/V1"])
    print(len(dataset))

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=2, shuffle=False)
    mix_console = mst.modules.BasicMixConsole(sample_rate=44100.0)
    mix_console = mst.modules.AdvancedMixConsole()
    for i, (track, instrument_id, stereo) in enumerate(dataloader):
        print("\n\n mixing")
        print("track", track.size())
        batch_size, num_tracks, seq_len = track.size()
        
        print(instrument_id)
        print(stereo)
        
        
        mix, param_dict = knowledge_engineering_mix(track, mix_console, instrument_id, stereo)
        print("mix", mix.size())
        if i==0:
            break

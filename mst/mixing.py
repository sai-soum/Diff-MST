# Store mixing functions here (e.g. knowledge engineering)
import torch
import json
import mst.modules


def instrument_metadata(instrument_id: list):
    """Convert the metadata info into istrument names
    """
    instrument_number_file = json.load(open("/homes/ssv02/Diff-MST/inst_id.txt"))
    iid = instrument_id.cpu().tolist()
    
    metadata =[]
    for id in iid: 
        instrument = [instrument for instrument, number in instrument_number_file.items() if number == id]
        
        metadata.append(instrument[0])
        
    mdata = dict(zip(range(len(metadata)), metadata))
    
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
    print("bs, num_tracks, seq_len", bs, num_tracks, seq_len)
    metadata = instrument_metadata(instrument_id)
    print(metadata)
    mix_params = torch.rand(bs, num_tracks, mix_console.num_control_params)
    print("mix_params", mix_params)
    print("mix_params.size()", mix_params.size())
    

    
    for i , track in enumerate(tracks):
        print(i)
        if 'silence' in metadata.values():
            track_idx = [idx for idx, instrument in metadata.items() if instrument == 'silence']
            mix_params[i, track_idx, :] = 0.0
        if 'bass' in metadata.values():
            track_idx = [idx for idx, instrument in metadata.items() if instrument == 'bass']
            mix_params[i, track_idx, 1] = 0.5
        if 'bass drum' or 'kick drum' in metadata.values():
            track_idx = [idx for idx, instrument in metadata.items() if instrument == 'kick drum' or instrument == 'bass drum']
            # print("kick", track_idx)
            mix_params[i, track_idx, 1] = 0.5
        if 'electric piano' or 'accordion'  in metadata.values():
            track_idx = [idx for idx, instrument in metadata.items() if instrument == 'electric piano' or instrument == 'accordion']
            # print("piano", track_idx)
            mix_params[i, track_idx, 1] = torch.rand(0.3,0.8)
        

    print("mix_params", mix_params)    
           
           
           


    mix = mix_console(tracks, param_dict)

    return mix, param_dict

if __name__ == "__main__":
    tracks = torch.rand(1, 4, 2400)
    instrument_id = torch.tensor([52, 33,  26 ,0])
    mix_console = mst.modules.BasicMixConsole(sample_rate=44100.0)
    mix, param_dict = knowledge_engineering_mix(tracks, mix_console, instrument_id)


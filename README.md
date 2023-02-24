# Diff-MST
<div align="center">

# Differentiable Mixing Style Transfer

<img src="./Assets/mst_wbg.png">

</div>

Mixing style transfer using reference mix. 
Given the tracks of a song and the correcsponding reference mix, the model can predict mixing console parameter values for each of the tracks.

There are two mixing console configurations (in `modules.py`)
1. `BasicMixConsole`: Gain + Pan
2. `AdvancedMixConsole`: Gain + Pan + Diff EQ + Diff Compressor

Mixes for training can be created using either `naive_random_mix` (assigns random parameter values for mixing console to create a mix) or `knowledge_engineering_mix` (uses knowledge engineering to assign parameter values for mixing console to create a mix). Both of these modules can be found in `mixing.py`



# Usage

Clone the repository and install the `mst` package.
```
git clone https://github.com/sai-soum/mix_style_transfer.git
cd mix_style_transfer
python -m venv env
source env/bin/activate
pip install -e .
```

[dasp-pytorch](https://github.com/csteinmetz1/dasp-pytorch) is required for differentiable audio effects.
Clone the repo into the top-level of the project directory.
```
git clone https://github.com/csteinmetz1/dasp-pytorch.git
cd dasp-pytorch
pip install -e .
```

Since `dasp` is currently under development you need to pull changes periodically. 
To do so change to the directory and pull.
```
cd dasp-pytorch
git pull
```

## Train

To train the model: 
```
python scripts/train.py fit --config=configs/medleydb_resnet.yaml
```
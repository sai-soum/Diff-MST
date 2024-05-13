
<div align="center">

# Differentiable Mixing Style Transfer
[Paper](https://sai-soum.github.io/assets/pdf/diffmst.pdf) | [Website](https://sai-soum.github.io/projects/diffmst/)


<img src="./Assets/diffmst-main_modified.jpg">

</div>

<!-- Mixing style transfer using reference mix. 
There are two mixing console configurations (in `modules.py`)
1. `BasicMixConsole`: Gain + Pan
2. `AdvancedMixConsole`: Gain + Pan + Diff EQ + Diff Compressor

Mixes for training can be created using either `naive_random_mix` (assigns random parameter values for mixing console to create a mix) or `knowledge_engineering_mix` (uses knowledge engineering to assign parameter values for mixing console to create a mix). Both of these modules can be found in `mixing.py`

 -->
# Repository Structure
1. `configs` - Contains configuration files for training and inference.
2. `mst` - Contains the main codebase for the project.
    - `dataloaders` - Contains dataloaders for the project.
    - `modules` - Contains the modules for different components of the system.
    - `mixing` - Contains the mixing modules for creating mixes.
    - `loss` - Contains the loss functions for the project.
    - `panns` - contains the most basic components like cnn14, resnet, etc.
    - `utils` - Contains utility functions for the project.
3. `scripts` - Contains scripts for running inference.  

# Usage
Clone the repository and install the `mst` package.
```
git clone https://github.com/sai-soum/Diff-MST.git
cd Diff-MST
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

## Train
We use [LightningCLI](https://lightning.ai/docs/pytorch/stable/) for training and [Wandb](https://wandb.ai/site) for logging.
First update the paths in the configuration file for both the logger, loss function, and the dataset root directory.
Then call the `main.py` script passing in the configuration file. 

### Method 1: Training with random mixes of the same song as reference using MRSTFT loss.
```
CUDA_VISIBLE_DEVICES=0 python main.py fit \
-c configs/config.yaml \
-c configs/optimizer.yaml \
-c configs/data/medley+cambridge-8.yaml \
-c configs/models/naive.yaml
```
You can change the number of tracks, the size of training data for an epoch, and the batch size in the data configuration file located at `configs/data/`

### Method 2: Training with real unpaired songs as reference using AFloss.
```
CUDA_VISIBLE_DEVICES=0 python main.py fit \
-c configs/config.yaml \
-c configs/optimizer.yaml \
-c configs/data/medley+cambridge+jamendo-8.yaml \
-c configs/models/naive+feat.yaml
```

## Inference
To evaluate the model on real world data, run the ` scripts/eval_all_combo.py` script. 

Update the model checkpoints and the inference examples directory in the script. 

`Python 3.10` was used for training. 

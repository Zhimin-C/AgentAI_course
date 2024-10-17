# Resource aalocation
srun -t 10:00:00 -c8 --mem=150gb --gres=gpu:v100:1 --pty /bin/bash

# Load modules
module load anaconda3/2023.09-0 cuda/11.8.0 git-lfs/3.3.0 openmpi/4.1.6 parallel/20220522 

# Create python environment
conda create -n agentAI python=3.11

# Activate python environment
source activate agentAI

# Install required packages
pip install numpy scipy scikit-learn matplotlib pandas seaborn
pip install roboflow faiss-gpu-cu11
conda install pytorch torchvision pytorch-cuda=12.1 -c pytorch -c nvidia


### conda cheatsheet: https://docs.conda.io/projects/conda/en/latest/_downloads/843d9e0198f2a193a3484886fa28163c/conda-cheatsheet.pdf

#### exporting conda environment: conda env export | grep -v "^prefix: " > environment.yml
#### importing conda environment: conda env create -f environment.yml
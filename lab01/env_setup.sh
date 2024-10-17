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
conda install pytorch torchvision pytorch-cuda=12.1 -c pytorch -c nvidia
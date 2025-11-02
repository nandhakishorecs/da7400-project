#!/bin/bash
# create and activate environment
conda create --name rl_test python=3.11 -y
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate rl_test

# install dependencies
pip install --upgrade --force-reinstall "gymnasium[classic-control,box2d]==0.28.1"

pip install wandb tqdm 

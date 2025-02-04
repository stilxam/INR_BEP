#!/bin/bash

# Ensure CUDA is visible to JAX
export XLA_FLAGS=--xla_gpu_cuda_data_dir=/usr/local/cuda

# Add the project root to PYTHONPATH
export PYTHONPATH=$PYTHONPATH:$(pwd)

# Activate conda environment (without running conda init)
eval "$(conda shell.bash hook)"
conda activate inr_edu_24

# Login to wandb (assuming you have wandb.login file)
if [ -f "./wandb.login" ]; then
    wandblogin="$(< ./wandb.login)"
    wandb login "$wandblogin"
else
    echo "Warning: wandb.login file not found. Make sure you're logged into wandb."
fi

echo 'Starting local test experiment!'

# Check if audio data exists
if [ ! -f "./example_data/data_gt_bach.npy" ]; then
    echo "Error: Audio data file not found at ./example_data/data_gt_bach.npy"
    exit 1
fi

# Run test configuration
python run_parallel.py --config=./configs/test_config.yaml 
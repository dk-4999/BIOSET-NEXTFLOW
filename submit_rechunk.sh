#!/bin/bash
#SBATCH --job-name=bioset_rechunk
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --time=4:00:00
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --output=/scratch/$USER/BIOSET-NEXTFLOW/logs/rechunk_%j.out
#SBATCH --error=/scratch/$USER/BIOSET-NEXTFLOW/logs/rechunk_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=YOUR_EMAIL@nyu.edu

module purge
module load anaconda3/2025.06
source /share/apps/anaconda3/2025.06/etc/profile.d/conda.sh
conda activate /scratch/$USER/envs/bioset

export PYTHONPATH=/scratch/$USER/envs/bioset/lib/python3.11/site-packages:$PYTHONPATH
export PATH=/scratch/$USER/envs/bioset/bin:$PATH

mkdir -p /scratch/$USER/BIOSET-NEXTFLOW/data
mkdir -p /scratch/$USER/BIOSET-NEXTFLOW/logs

/scratch/$USER/envs/bioset/bin/bioset convert \
    --source "https://lsp-public-data.s3.amazonaws.com/biomedvis-challenge-2025/Dataset1-LSP13626-melanoma-in-situ/0" \
    --target "/scratch/$USER/BIOSET-NEXTFLOW/data/rechunked.zarr" \
    --tile 128 \
    --overwrite
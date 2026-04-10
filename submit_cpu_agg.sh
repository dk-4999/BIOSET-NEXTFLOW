#!/bin/bash
# CPU Aggregation Workaround Script
#
# Use this if Nextflow's CPU_AGGREGATION stage fails because it cannot find
# the GPU stage checkpoints (it looks in its own work directory instead of
# the GPU process work directory).
#
# Before running:
#   1. Find the GPU stage work directory containing the checkpoints:
#      find /scratch/$USER/BIOSET-NEXTFLOW/nextflow_work -name "*.pkl" | head -1
#   2. Update --checkpoint-dir below to point to that directory
#      (the directory that CONTAINS the melanoma_in_situ/ subfolder)
#
#SBATCH --job-name=bioset_cpu_agg
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128GB
#SBATCH --time=4:00:00
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --output=/scratch/$USER/BIOSET-NEXTFLOW/BioSET_Preprocessing-master/logs/cpu_%j.out
#SBATCH --error=/scratch/$USER/BIOSET-NEXTFLOW/BioSET_Preprocessing-master/logs/cpu_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=YOUR_EMAIL@nyu.edu

module purge
module load anaconda3/2025.06
source /share/apps/anaconda3/2025.06/etc/profile.d/conda.sh
conda activate /scratch/$USER/envs/bioset

export PATH=/scratch/$USER/envs/bioset/bin:$PATH
export PYTHONPATH=/scratch/$USER/envs/bioset/lib/python3.11/site-packages:$PYTHONPATH

# UPDATE THIS: path to the GPU stage work directory that contains checkpoints/
GPU_WORK_DIR="/scratch/$USER/BIOSET-NEXTFLOW/nextflow_work/REPLACE_WITH_GPU_WORK_DIR"

/scratch/$USER/envs/bioset/bin/bioset run \
    --zarr-path "/scratch/$USER/BIOSET-NEXTFLOW/data/rechunked.zarr" \
    --zarr-url "https://lsp-public-data.s3.amazonaws.com/biomedvis-challenge-2025/Dataset1-LSP13626-melanoma-in-situ/0" \
    --meta "https://lsp-public-data.s3.amazonaws.com/biomedvis-challenge-2025/Dataset1-LSP13626-melanoma-in-situ/OME/METADATA.ome.xml" \
    --channels "0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69" \
    --tile 128 \
    --batch 70 \
    --alpha 0.4 \
    --trim-q 0.98 \
    --vox 0.14,0.14,0.28 \
    --min-vol-um3 1.0 \
    --dilate-um 0,0.5,1.0,1.5,2.0 \
    --max-set-size 4 \
    --min-marker-vox 100 \
    --min-support-pair 50 \
    --min-support-set 10 \
    --hierarchy-levels 4 \
    --output-dir "/scratch/$USER/BIOSET-NEXTFLOW/BioSET_Preprocessing-master/results" \
    --output-name "melanoma_in_situ" \
    --checkpoint-dir "${GPU_WORK_DIR}/checkpoints" \
    --stage cpu
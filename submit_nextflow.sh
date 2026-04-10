#!/bin/bash
#SBATCH --job-name=bioset_nf
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16GB
#SBATCH --time=48:00:00
#SBATCH --account=torch_pr_608_general
#SBATCH --output=/scratch/$USER/BIOSET-NEXTFLOW/BioSET_Preprocessing-master/logs/nextflow_%j.out
#SBATCH --error=/scratch/$USER/BIOSET-NEXTFLOW/BioSET_Preprocessing-master/logs/nextflow_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=YOUR_EMAIL@nyu.edu

module purge
module load openjdk/21.0.3
module load nextflow/24.04.6
module load anaconda3/2025.06
source /share/apps/anaconda3/2025.06/etc/profile.d/conda.sh
conda activate /scratch/$USER/envs/bioset

export NXF_HOME=/scratch/$USER/.nextflow
export NXF_TEMP=/scratch/$USER/.nextflow/tmp
export NXF_WORK=/scratch/$USER/BIOSET-NEXTFLOW/nextflow_work
export JAVA_TOOL_OPTIONS="-Djava.io.tmpdir=/scratch/$USER/.nextflow/tmp"
export PYTHONPATH=/scratch/$USER/envs/bioset/lib/python3.11/site-packages:$PYTHONPATH
export NXF_CONDA_CACHEDIR=/scratch/$USER/.nextflow/conda
export PATH=/scratch/$USER/envs/bioset/bin:$PATH
export NXF_PYTHON=/scratch/$USER/envs/bioset/bin/python

mkdir -p /scratch/$USER/.nextflow/tmp
mkdir -p /scratch/$USER/BIOSET-NEXTFLOW/nextflow_work

cd /scratch/$USER/BIOSET-NEXTFLOW/BioSET_Preprocessing-master

nextflow run main.nf \
    -profile nyu_torch \
    -w /scratch/$USER/BIOSET-NEXTFLOW/nextflow_work \
    --mode legacy \
    --zarr_path "/scratch/$USER/BIOSET-NEXTFLOW/data/rechunked.zarr" \
    --zarr_url "https://lsp-public-data.s3.amazonaws.com/biomedvis-challenge-2025/Dataset1-LSP13626-melanoma-in-situ/0" \
    --metadata_url "https://lsp-public-data.s3.amazonaws.com/biomedvis-challenge-2025/Dataset1-LSP13626-melanoma-in-situ/OME/METADATA.ome.xml" \
    --channels "0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69" \
    --channel_batch 70 \
    --output_dir "/scratch/$USER/BIOSET-NEXTFLOW/BioSET_Preprocessing-master/results" \
    --output_name "melanoma_in_situ" \
    --checkpoint_dir "/scratch/$USER/BIOSET-NEXTFLOW/BioSET_Preprocessing-master/checkpoints" \
    --outdir "/scratch/$USER/BIOSET-NEXTFLOW/BioSET_Preprocessing-master/bioset_results" \
    -resume
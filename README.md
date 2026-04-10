# BioSET Preprocessing

<p align="center">
  <strong>GPU-accelerated spatial overlap analysis pipeline for multi-channel biomedical imaging</strong>
</p>

<p align="center">
  Processes large-scale 3D tissue volumes to identify and quantify co-localization patterns between biomarkers.
  Orchestrated via <strong>Nextflow</strong> for reproducible, resumable execution on HPC clusters.
</p>

---

## Overview

BioSET Preprocessing takes multiplexed immunofluorescence imaging data (OME-Zarr format) and computes pairwise and multi-channel co-localization statistics across the entire tissue volume. It supports two input modes:

- **Legacy mode** — raw OME-Zarr → adaptive thresholding → overlap analysis
- **MCMICRO mode** — MCMICRO segmentation + quantification outputs → overlap analysis

Output is a compressed SQLite database (`.bioset` format) ready for visualization in the BioSET Viewer.

---

## Features

- **GPU-accelerated** processing using CuPy (CUDA 12.x)
- **Nextflow orchestration** — reproducible, resumable, HPC-ready
- **Resumable checkpointing** — interrupt and resume at any tile
- **Two-stage pipeline** — GPU tile processing + CPU aggregation run separately for efficient resource use
- **Hierarchical aggregation** — multi-scale spatial analysis (4 levels)
- **Portable output** — compressed SQLite database (`.bioset` format)
- **NYU Torch HPC** ready — includes SLURM job scripts and Nextflow profile

---

## Requirements

- Python 3.10+
- NVIDIA GPU with CUDA 12.x support
- Java 17+ (for Nextflow)
- Nextflow 24.04+
- 64GB+ RAM recommended for large datasets

---

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/dk-4999/BIOSET-NEXTFLOW.git
cd BIOSET-NEXTFLOW
```

### 2. Create Conda Environment

```bash
conda create -p /path/to/envs/bioset python=3.11 -y
conda activate /path/to/envs/bioset
```

### 3. Install Dependencies

```bash
# GPU acceleration (match your CUDA version)
pip install cupy-cuda12x

# All other dependencies
pip install numpy scipy "dask[array]" zarr==2.18.7 ome-zarr ome-types \
            requests tifffile pyyaml matplotlib rechunker==0.5.4

# Install the project
pip install --no-deps -e .
```

### 4. Verify Installation

```bash
python -c "import cupy, zarr, dask, ome_types; print('All OK')"
bioset --help
```

---

## Quick Start

### Step 1: Rechunk Data Locally (Strongly Recommended)

Remote data access is IO-bound and ~100x slower. Rechunk once to local storage:

```bash
bioset convert \
    --source "https://your-data-url/zarr/0" \
    --target "/path/to/local/rechunked.zarr" \
    --tile 128 \
    --overwrite
```

### Step 2: Run the Pipeline

```bash
nextflow run main.nf \
    -profile nyu_torch \
    --mode legacy \
    --zarr_path "/path/to/local/rechunked.zarr" \
    --zarr_url "https://your-data-url/zarr/0" \
    --metadata_url "https://your-data-url/OME/METADATA.ome.xml" \
    --channels "0,1,2,...,69" \
    --channel_batch 70 \
    --output_name "my_analysis" \
    -resume
```

### Step 3: Check Output

```bash
ls results/my_analysis.bioset
```

---

## Running on NYU Torch HPC

### Prerequisites

- Active NYU HPC account with an allocation on Torch
- Know your account name (run `sacctmgr show associations user=$USER format=account%50`)

### Setup

```bash
# Load modules
module load anaconda3/2025.06
module load openjdk/21.0.3
module load nextflow/24.04.6

# Create conda environment in scratch (avoids home directory inode limits)
conda create -p /scratch/$USER/envs/bioset python=3.11 -y
conda activate /scratch/$USER/envs/bioset

# Install dependencies
pip install cupy-cuda12x numpy scipy "dask[array]" zarr==2.18.7 \
            ome-zarr ome-types requests tifffile pyyaml matplotlib rechunker==0.5.4
pip install --no-deps -e .
```

### Step 1: Rechunk the Data (One-Time Setup)

Remote data is ~100x slower than local. Before running the pipeline, rechunk the dataset to scratch using `submit_rechunk.sh`:

```bash
sbatch submit_rechunk.sh
```

This submits a CPU job that downloads and rechunks the source zarr to `data/rechunked.zarr`. Expect ~3-4 hours for a 150GB dataset. You'll get an email when it finishes.

Verify the output afterward:

```bash
python -c "
import zarr
z = zarr.open('/scratch/$USER/BIOSET-NEXTFLOW/data/rechunked.zarr', mode='r')
print('Shape:', z.shape, 'Chunks:', z.chunks)
"
```

### Step 2: Run the Pipeline

```bash
sbatch submit_nextflow.sh
```

### Step 3 (Workaround): Run CPU Aggregation Manually

If Nextflow's `CPU_AGGREGATION` stage fails due to the checkpoint path issue (it looks for checkpoints in its own work directory instead of the GPU stage's work directory), use `submit_cpu_agg.sh` as a direct workaround:

```bash
# First find the GPU stage work directory that has the checkpoints
find /scratch/$USER/BIOSET-NEXTFLOW/nextflow_work -name "*.pkl" | head -1

# Edit submit_cpu_agg.sh to point --checkpoint-dir to that work directory
nano submit_cpu_agg.sh

# Then submit
sbatch submit_cpu_agg.sh
```

---

### Edit the Submit Scripts

Open `submit_nextflow.sh` and update:

```bash
# Update your account name
#SBATCH --account=YOUR_ACCOUNT_NAME

# Update your email
#SBATCH --mail-user=YOUR_NETID@nyu.edu

# Update paths to match your scratch directory
--zarr_path "/scratch/$USER/BIOSET-NEXTFLOW/data/rechunked.zarr"
--output_dir "/scratch/$USER/BIOSET-NEXTFLOW/results"
--checkpoint_dir "/scratch/$USER/BIOSET-NEXTFLOW/checkpoints"
--outdir "/scratch/$USER/BIOSET-NEXTFLOW/bioset_results"
```

### Submit the Job

```bash
sbatch submit_nextflow.sh
```

### Monitor Progress

```bash
# Check job status
squeue -u $USER

# Watch live log
tail -f logs/nextflow_JOBID.out
```

### Resume After Interruption

The pipeline saves a checkpoint after every tile. Just resubmit:

```bash
sbatch submit_nextflow.sh   # -resume flag is already in the script
```

---

## Pipeline Architecture

```
Input (OME-Zarr)
       │
       ▼
┌──────────────────┐
│ GLOBAL_THRESHOLDS│  CPU  — Computes per-channel intensity thresholds
│  (~5 min)        │        from low-resolution volume
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ GPU_TILE_PROCESS │  GPU  — Processes 3,784 tiles (128×128 pixels each)
│  (~1-2 hours)    │        Thresholding → CC filter → Dilation → Overlap mining
│                  │        Saves checkpoint per tile (fully resumable)
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ CPU_AGGREGATION  │  CPU  — Loads checkpoints, aggregates across tiles
│  (~30 min)       │        Computes 4-level spatial hierarchy
│                  │        Writes final .bioset database
└────────┬─────────┘
         │
         ▼
   results/*.bioset
```

### How Co-localization is Computed

For each spatial tile and each channel:

1. **Alpha Thresholding** — Adaptive threshold using median + MAD from low-resolution volume. Final threshold = max(global, local) to prevent under-thresholding.

2. **Connected Components Filtering** — Removes small noise objects below `min_obj_vol_um3`.

3. **Multi-radius Dilation** — Euclidean Distance Transform swept at multiple radii (e.g., 0, 0.5, 1.0, 1.5, 2.0 µm) to capture proximity-based co-localization.

4. **Set Overlap Mining** — Apriori-style frequent itemset mining for all channel pairs and higher-order sets (up to `max_set_size`). Computes IoU and Overlap Coefficient.

---

## Configuration Reference

### Pipeline Parameters

| Parameter            | Default             | Description                                                          |
| -------------------- | ------------------- | -------------------------------------------------------------------- |
| `--mode`             | `legacy`            | `legacy` or `mcmicro`                                                |
| `--zarr_path`        | `null`              | Local OME-Zarr path (preferred for speed)                            |
| `--zarr_url`         | `null`              | Remote OME-Zarr URL                                                  |
| `--metadata_url`     | `''`                | OME-XML metadata URL                                                 |
| `--channels`         | `''`                | Comma-separated channel indices                                      |
| `--tile_size`        | `128`               | Tile size in pixels                                                  |
| `--channel_batch`    | `70`                | Channels per GPU batch (set to all channels for max GPU utilization) |
| `--alpha`            | `0.4`               | Threshold aggressiveness (higher = more selective)                   |
| `--trim_q`           | `0.98`              | Background trimming quantile                                         |
| `--voxel_size`       | `0.14,0.14,0.28`    | Physical voxel size (x,y,z) in µm                                    |
| `--dilate_um`        | `0,0.5,1.0,1.5,2.0` | Dilation radii in microns                                            |
| `--max_set_size`     | `4`                 | Maximum combination size                                             |
| `--min_marker_vox`   | `100`               | Min voxels per marker to be active                                   |
| `--min_support_pair` | `50`                | Min intersection voxels for pairs                                    |
| `--min_support_set`  | `10`                | Min intersection voxels for sets                                     |
| `--hierarchy_levels` | `4`                 | Number of spatial aggregation levels                                 |
| `--output_name`      | `analysis`          | Output filename (without .bioset)                                    |

### Nextflow Profiles

| Profile       | Description                           |
| ------------- | ------------------------------------- |
| `standard`    | Local execution (development/testing) |
| `nyu_torch`   | NYU Torch HPC cluster (SLURM + Conda) |
| `docker`      | Docker container execution            |
| `singularity` | Singularity container execution       |
| `test`        | Minimal resources for quick testing   |

---

## Output Format

The pipeline produces a `.bioset` file (gzip-compressed SQLite) with four tables:

| Table           | Description                                                           |
| --------------- | --------------------------------------------------------------------- |
| `metadata`      | Channels, dilations, hierarchy levels, volume bounds                  |
| `channel_stats` | Per-channel per-tile voxel counts and intensities                     |
| `combinations`  | Pairwise and multi-channel overlap metrics (IoU, overlap coefficient) |
| `tiles`         | Spatial tile coordinates for each combination                         |

### Reading Output

```python
import gzip, sqlite3, json, tempfile

with gzip.open("analysis.bioset", "rb") as f:
    tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    tmp.write(f.read()); tmp.close()

conn = sqlite3.connect(tmp.name)
conn.row_factory = sqlite3.Row

# Top 20 co-localizing channel pairs by IoU
cursor = conn.execute("""
    SELECT channels,
           CAST(SUM(total_count) AS REAL) / SUM(total_union) as global_iou
    FROM combinations
    WHERE dilation = 2.0 AND hierarchy_level = 0 AND channel_count = 2
    GROUP BY channels HAVING SUM(total_union) > 0
    ORDER BY global_iou DESC LIMIT 20
""")
for row in cursor:
    print(f"{row['channels']}: IoU={row['global_iou']:.4f}")
```

---

## Important Notes

### Big-endian Zarr Data

Some OME-Zarr datasets (e.g., BiomedVis Challenge 2025) use big-endian (`>u2`) encoding. The pipeline automatically converts to `float32` before GPU processing to handle this correctly.

### GPU Utilization

Set `--channel_batch` equal to your total number of channels to maximize GPU utilization. Loading all channels in one batch minimizes I/O and keeps GPU usage high (required threshold on Torch: >50%).

### Local Data is Critical

The pipeline is IO-bound when reading from remote URLs. Always rechunk data locally first using `bioset convert`. Expect ~100x speedup with local data (1+ tiles/sec vs 0.01 tiles/sec remotely).

### Scratch Purge Policy (NYU Torch)

Files in `/scratch` are deleted after 60 days of no access. Archive your `.bioset` output:

```bash
cp /scratch/$USER/results/analysis.bioset /archive/$USER/
```

---

## Troubleshooting

| Issue                                    | Solution                                                                           |
| ---------------------------------------- | ---------------------------------------------------------------------------------- |
| `No module named 'bioset_preprocessing'` | Conda env not activating — add `beforeScript` to Nextflow profile                  |
| `No readable zarr components`            | Using `path` instead of `val` for zarr input in Nextflow process                   |
| `No checkpoints found`                   | CPU aggregation looking in wrong directory — pass absolute `--checkpoint-dir`      |
| `0 active channels`                      | Big-endian zarr dtype — ensure `.astype(np.float32)` is applied when reading tiles |
| `Low GPU utilization (job cancelled)`    | Increase `--channel_batch` to total number of channels                             |
| `OutOfMemoryError`                       | Reduce `--channel_batch` or `--tile_size`                                          |
| Slow processing                          | Use local zarr — run `bioset convert` first                                        |
| `cudaErrorMpsConnectionFailed`           | Remove `preemption=yes` from `#SBATCH --comment`                                   |

---

## Project Structure

```
BIOSET-NEXTFLOW/
├── main.nf                          # Nextflow pipeline definition
├── nextflow.config                  # Profiles (nyu_torch, docker, etc.)
├── submit_nextflow.sh               # SLURM job submission script (NYU Torch)
├── submit_rechunk.sh                # SLURM job to rechunk remote zarr → local scratch (one-time)
├── submit_cpu_agg.sh                # SLURM job to run CPU aggregation directly (workaround)
├── pyproject.toml                   # Python package definition
├── configs/
│   └── mcmicro/
│       ├── marker_thresholds.yml    # Per-marker positivity thresholds (MCMICRO mode)
│       ├── markers.csv              # Channel/marker definitions
│       └── params.yml               # MCMICRO workflow parameters
├── containers/
│   ├── Dockerfile.gpu               # GPU container (CUDA 12.2)
│   └── Dockerfile.cpu               # CPU-only container
├── scripts/
│   ├── submit_full_pipeline.sh      # Alternative pipeline submission script
│   └── make_mcmicro_project.py      # MCMICRO project layout generator
└── src/bioset_preprocessing/
    ├── pipeline.py                  # Main Pipeline class
    ├── config.py                    # PipelineConfig, VoxelSizeUM
    ├── cli.py                       # bioset CLI entry point
    ├── io.py                        # ZarrPyramid I/O
    ├── tiling.py                    # Tile iteration utilities
    ├── checkpoint.py                # Save/load tile checkpoints
    ├── aggregation.py               # Hierarchical aggregation
    ├── writer.py                    # .bioset database writer
    ├── filtering.py                 # Post-processing filters
    ├── validation.py                # Compare two .bioset outputs
    ├── adapters/
    │   └── mcmicro_adapter.py       # MCMICRO → BioSET adapter
    ├── converters/
    │   └── tiff_to_zarr.py          # TIFF/Zarr rechunking utilities
    └── stages/
        ├── threshold.py             # AlphaThreshold
        ├── cc_filter.py             # ConnectedComponentsFilter
        ├── dilation.py              # EDTSweepDilation
        └── overlaps.py              # OverlapMiner
```

---

## License

MIT License — see [LICENSE](LICENSE) for details.

---

## Authors

Divij Kapur

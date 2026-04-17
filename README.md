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

## Running on NYU Torch HPC

### Prerequisites

- Active NYU HPC account with an allocation on Torch
- Know your account name (run `sacctmgr show associations user=$USER format=account%50`)

### Step 1: Clone the Repository

```bash
# Work in scratch to avoid home directory inode limits
cd /scratch/$USER
mkdir -p BIOSET-NEXTFLOW && cd BIOSET-NEXTFLOW

git clone https://github.com/dk-4999/BIOSET-NEXTFLOW.git BioSET_Preprocessing-master
cd BioSET_Preprocessing-master
```

### Step 2: Create Conda Environment

```bash
# Load required modules
module load anaconda3/2025.06
module load openjdk/21.0.3
module load nextflow/24.04.6

# Create conda environment in scratch
conda create -p /scratch/$USER/envs/bioset python=3.11 -y
source /share/apps/anaconda3/2025.06/etc/profile.d/conda.sh
conda activate /scratch/$USER/envs/bioset

# Install dependencies
pip install cupy-cuda12x numpy scipy "dask[array]" zarr==2.18.7 \
            ome-zarr ome-types requests tifffile pyyaml matplotlib rechunker==0.5.4

# Install the project (editable mode)
pip install --no-deps -e .
```

### Step 3: Verify Installation

```bash
python -c "import cupy, zarr, dask, ome_types; print('All OK')"
bioset --help
```

### Step 4: Update Configuration Files

#### 4a. Update `nextflow.config`

The `nyu_torch` profile needs your SLURM account name. Edit `nextflow.config`:

```bash
nano nextflow.config
```

Find the `nyu_torch` profile and update:

- `--account=YOUR_ACCOUNT_NAME` → your actual SLURM account (e.g., `torch_pr_608_general`)

> **Note:** The conda path uses `System.getenv('USER')` to automatically resolve to your username. No changes needed there.

#### 4b. Update `submit_nextflow.sh`

```bash
nano submit_nextflow.sh
```

Update these lines:

- `#SBATCH --account=` → your SLURM account name
- `#SBATCH --mail-user=` → your NYU email
- All paths containing a specific username → replace with your username or `$USER`

#### 4c. Update `submit_rechunk.sh`

```bash
nano submit_rechunk.sh
```

Update:

- `#SBATCH --account=` → your SLURM account name
- `#SBATCH --mail-user=` → your NYU email
- The `--source` URL if using a different dataset

### Step 5: Rechunk the Data (One-Time Setup)

Remote data is ~100x slower than local. Download and rechunk the dataset to scratch:

```bash
mkdir -p /scratch/$USER/BIOSET-NEXTFLOW/data
mkdir -p /scratch/$USER/BIOSET-NEXTFLOW/logs

sbatch submit_rechunk.sh
```

This submits a CPU job that downloads and rechunks the source zarr to `data/rechunked.zarr`. Expect **~3-4 hours** for a 150GB dataset. You'll get an email when it finishes.

Verify the output:

```bash
source /share/apps/anaconda3/2025.06/etc/profile.d/conda.sh
conda activate /scratch/$USER/envs/bioset

python -c "
import zarr
z = zarr.open('/scratch/$USER/BIOSET-NEXTFLOW/data/rechunked.zarr', mode='r')
print('Shape:', z.shape, 'Chunks:', z.chunks, 'Dtype:', z.dtype)
"
```

### Step 6: Run the Pipeline

#### Option A: Submit as SLURM Job (Recommended)

This runs the Nextflow orchestrator as a SLURM job, which in turn submits GPU and CPU sub-jobs. You can log out safely.

```bash
cd /scratch/$USER/BIOSET-NEXTFLOW/BioSET_Preprocessing-master
mkdir -p logs

sbatch submit_nextflow.sh
```

You'll receive emails when jobs start and finish.

#### Option B: Run from Login Node with tmux

The Nextflow orchestrator is lightweight and can run on a login node. Use `tmux` so it survives SSH disconnections:

```bash
cd /scratch/$USER/BIOSET-NEXTFLOW/BioSET_Preprocessing-master

# Start a tmux session
tmux new -s bioset

# Load modules
module load openjdk/21.0.3
module load nextflow/24.04.6
module load anaconda3/2025.06
source /share/apps/anaconda3/2025.06/etc/profile.d/conda.sh
conda activate /scratch/$USER/envs/bioset

# Run the pipeline
nextflow run main.nf \
    -profile nyu_torch \
    -w /scratch/$USER/BIOSET-NEXTFLOW/nextflow_work \
    --mode legacy \
    --zarr_path "/scratch/$USER/BIOSET-NEXTFLOW/data/rechunked.zarr" \
    --metadata_url "https://lsp-public-data.s3.amazonaws.com/biomedvis-challenge-2025/Dataset1-LSP13626-melanoma-in-situ/OME/METADATA.ome.xml" \
    --channels "0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69" \
    --channel_batch 70 \
    --output_name "melanoma_in_situ" \
    --checkpoint_dir "/scratch/$USER/BIOSET-NEXTFLOW/BioSET_Preprocessing-master/checkpoints" \
    --outdir "/scratch/$USER/BIOSET-NEXTFLOW/BioSET_Preprocessing-master/bioset_results"
```

tmux commands:

- **Detach:** `Ctrl+B` then `D`
- **Reattach:** `tmux attach -t bioset`

### Step 7: Monitor Progress

```bash
# Check SLURM job status
squeue -u $USER

# Count processed tiles (3784 = all done for the melanoma dataset)
ls /scratch/$USER/BIOSET-NEXTFLOW/BioSET_Preprocessing-master/checkpoints/melanoma_in_situ/*.json.gz 2>/dev/null | wc -l

# Tail the GPU stage log (find the latest work directory)
tail -f $(find /scratch/$USER/BIOSET-NEXTFLOW/nextflow_work -name ".command.log" -mmin -60 2>/dev/null | head -1)
```

### Step 8: Check Output

```bash
source /share/apps/anaconda3/2025.06/etc/profile.d/conda.sh
conda activate /scratch/$USER/envs/bioset

python3 << 'EOF'
import gzip, sqlite3, shutil

tmp_path = "/scratch/$USER/temp_bioset.db"

with gzip.open("bioset_results/melanoma_in_situ.bioset", "rb") as f_in:
    with open(tmp_path, "wb") as f_out:
        shutil.copyfileobj(f_in, f_out, length=1024*1024)

conn = sqlite3.connect(tmp_path)
print("Pairs:", conn.execute("SELECT COUNT(*) FROM combinations WHERE channel_count=2").fetchone()[0])

print("\nTop 10 co-localizing pairs (dilation=2.0):")
for row in conn.execute("""
    SELECT channels, CAST(SUM(total_count) AS REAL)/SUM(total_union) as iou
    FROM combinations WHERE dilation=2.0 AND hierarchy_level=0 AND channel_count=2
    GROUP BY channels HAVING SUM(total_union)>0
    ORDER BY iou DESC LIMIT 10
"""):
    print(f"  {row[0]}: IoU={row[1]:.4f}")
conn.close()
import os; os.unlink(tmp_path)
EOF
```

### Step 9: Download Results

From your **local machine** (not HPC):

```bash
scp YOUR_NETID@torch.hpc.nyu.edu:/scratch/YOUR_NETID/BIOSET-NEXTFLOW/BioSET_Preprocessing-master/bioset_results/melanoma_in_situ.bioset .
```

### Resume After Interruption

The pipeline saves a checkpoint after every tile. If interrupted, just resubmit:

```bash
sbatch submit_nextflow.sh   # -resume flag is already in the script
```

Or with Option B, add `-resume` to the nextflow command.

> **Important:** If you change pipeline parameters (e.g., `min_obj_vol_um3`, `trim_q`), you must delete old checkpoints before rerunning, otherwise the pipeline will resume from stale checkpoints:
>
> ```bash
> rm -rf checkpoints/melanoma_in_situ/
> ```

---

## Expected Runtime

For the BiomedVis Challenge 2025 melanoma dataset (70 channels, 194×5508×10908 voxels, 3,784 tiles):

| Stage                      | Time          | Hardware                  |
| -------------------------- | ------------- | ------------------------- |
| Data rechunking (one-time) | ~3-4 hours    | CPU (8 cores, 64GB RAM)   |
| Global thresholds          | ~2 hours      | GPU (L40S)                |
| Tile processing            | ~12 hours     | GPU (L40S)                |
| CPU aggregation            | ~10 minutes   | CPU (16 cores, 128GB RAM) |
| **Total pipeline**         | **~15 hours** | —                         |

> **Note:** Runtime depends heavily on the number of active channels per tile. Faster GPUs (A100, H100) will reduce tile processing time by 2-3x.

---

## Pipeline Architecture

```
Input (OME-Zarr, local)
       │
       ▼
┌──────────────────┐
│ GLOBAL_THRESHOLDS│  GPU — Computes per-channel intensity thresholds
│  (~2 hours)      │        from subsampled high-res volume
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ GPU_TILE_PROCESS │  GPU — Processes 3,784 tiles (128×128 pixels each)
│  (~12 hours)     │       Thresholding → CC filter → Dilation → Overlap mining
│                  │       Saves checkpoint per tile (fully resumable)
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ CPU_AGGREGATION  │  CPU — Loads checkpoints, aggregates across tiles
│  (~10 min)       │       Computes 4-level spatial hierarchy
│                  │       Writes final .bioset database
└────────┬─────────┘
         │
         ▼
   bioset_results/*.bioset
```

### How Co-localization is Computed

For each spatial tile and each channel:

1. **Alpha Thresholding** — Adaptive threshold using median + MAD from subsampled volume. Final threshold = max(global, local) to prevent under-thresholding.

2. **Connected Components Filtering** — Removes small noise objects below `min_obj_vol_um3` (default: 0.01 µm³ ≈ 2 voxels).

3. **Multi-radius Dilation** — Euclidean Distance Transform swept at multiple radii (0, 0.5, 1.0, 1.5, 2.0 µm) to capture proximity-based co-localization.

4. **Set Overlap Mining** — Apriori-style frequent itemset mining for all channel pairs (and optionally higher-order sets up to `max_set_size`). Computes IoU and Overlap Coefficient.

---

## Configuration Reference

### Pipeline Parameters

| Parameter            | Default             | Description                                                                              |
| -------------------- | ------------------- | ---------------------------------------------------------------------------------------- |
| `--mode`             | `legacy`            | `legacy` or `mcmicro`                                                                    |
| `--zarr_path`        | `null`              | Local OME-Zarr path (required)                                                           |
| `--metadata_url`     | `''`                | OME-XML metadata URL (for channel names)                                                 |
| `--channels`         | `''`                | Comma-separated channel indices                                                          |
| `--tile_size`        | `128`               | Tile size in pixels                                                                      |
| `--channel_batch`    | `70`                | Channels per GPU batch (set to total channels for max GPU utilization)                   |
| `--alpha`            | `0.4`               | Threshold aggressiveness (higher = more selective)                                       |
| `--trim_q`           | `1.0`               | Background trimming quantile (1.0 = no trimming)                                         |
| `--voxel_size`       | `0.14,0.14,0.28`    | Physical voxel size (x,y,z) in µm                                                        |
| `--min_obj_vol_um3`  | `0.01`              | Min object volume in µm³ for CC filter (~2 voxels)                                       |
| `--dilate_um`        | `0,0.5,1.0,1.5,2.0` | Dilation radii in microns                                                                |
| `--max_set_size`     | `2`                 | Maximum combination size (2 = pairs only, 3-4 adds higher-order sets but is ~10x slower) |
| `--min_marker_vox`   | `20`                | Min voxels per marker to be active                                                       |
| `--min_support_pair` | `10`                | Min intersection voxels for pairs                                                        |
| `--min_support_set`  | `10`                | Min intersection voxels for sets                                                         |
| `--hierarchy_levels` | `4`                 | Number of spatial aggregation levels                                                     |
| `--output_name`      | `analysis`          | Output filename (without .bioset)                                                        |

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
import gzip, sqlite3, shutil

# Decompress to a temp file (use scratch on HPC to avoid /tmp space issues)
tmp_path = "/scratch/YOUR_NETID/temp_bioset.db"
with gzip.open("analysis.bioset", "rb") as f_in:
    with open(tmp_path, "wb") as f_out:
        shutil.copyfileobj(f_in, f_out)

conn = sqlite3.connect(tmp_path)

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
    print(f"{row[0]}: IoU={row[1]:.4f}")

conn.close()
import os; os.unlink(tmp_path)
```

---

## Important Notes

### Big-Endian Zarr Data

Some OME-Zarr datasets (e.g., BiomedVis Challenge 2025) use big-endian (`>u2`) encoding. The pipeline automatically converts to `float32` before GPU processing to handle this correctly.

### Local Data is Critical

The pipeline is IO-bound when reading from remote URLs. **Always rechunk data locally first** using `bioset convert` or `submit_rechunk.sh`. Expect ~100x speedup with local data.

### GPU Utilization

Set `--channel_batch` equal to your total number of channels to maximize GPU utilization. Loading all channels in one batch minimizes I/O and keeps GPU usage high.

### Higher-Order Sets

Setting `--max_set_size` to 3 or 4 enables 3-way and 4-way co-localization analysis but increases runtime by ~10x due to combinatorial explosion. Start with `max_set_size=2` (pairs only) to validate results, then increase if needed.

### Scratch Purge Policy (NYU Torch)

Files in `/scratch` are deleted after 60 days of no access. Archive your `.bioset` output:

```bash
cp /scratch/$USER/BIOSET-NEXTFLOW/BioSET_Preprocessing-master/bioset_results/analysis.bioset /archive/$USER/
```

---

## Troubleshooting

| Issue                                    | Solution                                                                                |
| ---------------------------------------- | --------------------------------------------------------------------------------------- |
| `No module named 'bioset_preprocessing'` | Conda env not activating — check `beforeScript` in `nextflow.config` nyu_torch profile  |
| `Conda prefix path does not exist`       | `$USER` not expanding in `nextflow.config` — ensure double quotes around the conda path |
| `No readable zarr components`            | Using `path` instead of `val` for zarr input in Nextflow process                        |
| `No checkpoints found`                   | CPU aggregation looking in wrong directory — pass absolute `--checkpoint-dir`           |
| `0 active channels`                      | Multiple possible causes (see below)                                                    |
| `0 pairs, 0 sets`                        | Check checkpoint files for `n_active_channels` — if 0, see "0 active channels"          |
| `Low GPU utilization (job cancelled)`    | Increase `--channel_batch` to total number of channels                                  |
| `OutOfMemoryError (exit code 137)`       | Reduce `--channel_batch` or ensure global thresholds use subsampled data                |
| `503 Service Unavailable`                | Remote S3 data unavailable — use local zarr only (drop `--zarr_url`)                    |
| `cudaErrorMpsConnectionFailed`           | Remove `preemption=yes` from SBATCH comment flags                                       |
| Slow processing                          | Use local zarr, reduce `--max_set_size`, or increase `--min_marker_vox`                 |
| SSH disconnects kill pipeline            | Use `tmux` or submit as SLURM job via `submit_nextflow.sh`                              |
| Stale checkpoints after param change     | Delete `checkpoints/` directory before rerunning                                        |

### Debugging "0 Active Channels"

If all tiles show 0 active channels:

1. **Check zarr store type:** The I/O layer must use `zarr.DirectoryStore` (not `NestedDirectoryStore`) for rechunked data with flat chunk naming. Verify in `src/bioset_preprocessing/io.py`.

2. **Check global thresholds:** Run a diagnostic to print `t_global` values. If all are 0.0, data is being read as zeros (likely a store type mismatch).

3. **Check `min_obj_vol_um3`:** If set too high (e.g., 1.0 µm³ = 183 voxels), the connected components filter removes all sparse signal. Use 0.01 µm³ (~2 voxels) for sparse datasets.

4. **Check `trim_q`:** Values below 1.0 clip background aggressively. Use 1.0 for datasets where most voxels are background.

---

## Project Structure

```
BIOSET-NEXTFLOW/
├── main.nf                          # Nextflow pipeline definition
├── nextflow.config                  # Profiles (nyu_torch, docker, etc.)
├── submit_nextflow.sh               # SLURM job submission script (NYU Torch)
├── submit_rechunk.sh                # SLURM job to rechunk remote zarr → local scratch
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

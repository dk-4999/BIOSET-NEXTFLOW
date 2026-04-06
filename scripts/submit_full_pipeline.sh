#!/bin/bash
# ===================================================================
# BioSET Full Pipeline Submission Script
# ===================================================================
#
# Submits the entire BioSET Nextflow pipeline to an HPC cluster.
# Handles both legacy and MCMICRO modes.
#
# Usage:
#   # Legacy mode
#   ./submit_full_pipeline.sh legacy \
#       --zarr_path /scratch/user/rechunked.zarr \
#       --channels "0,1,2,...,69" \
#       --metadata_url "https://..."
#
#   # MCMICRO mode
#   ./submit_full_pipeline.sh mcmicro \
#       --zarr_path /scratch/user/converted.zarr \
#       --segmentation /scratch/user/seg_mask.tif \
#       --quantification /scratch/user/quant.csv \
#       --marker_thresholds configs/mcmicro/marker_thresholds.yml
#
#   # Resume after interruption
#   ./submit_full_pipeline.sh resume
#
# ===================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# Defaults
PROFILE="${NF_PROFILE:-nyu_hpc}"
WORK_DIR="${NF_WORK_DIR:-/scratch/$USER/nextflow_work}"
OUTPUT_DIR="${NF_OUTPUT_DIR:-/scratch/$USER/bioset_results}"
NF_LOG="${OUTPUT_DIR}/nextflow.log"

# Ensure output directory exists
mkdir -p "$OUTPUT_DIR"

MODE="${1:-}"
shift || true

case "$MODE" in
    legacy)
        echo "========================================="
        echo " BioSET Pipeline - Legacy Mode"
        echo "========================================="
        echo " Profile: $PROFILE"
        echo " Work dir: $WORK_DIR"
        echo " Output: $OUTPUT_DIR"
        echo "========================================="

        nextflow run "$PROJECT_DIR/main.nf" \
            -profile "$PROFILE" \
            -w "$WORK_DIR" \
            --mode legacy \
            --outdir "$OUTPUT_DIR" \
            "$@" \
            -with-report "$OUTPUT_DIR/report.html" \
            -with-timeline "$OUTPUT_DIR/timeline.html" \
            -with-trace "$OUTPUT_DIR/trace.txt" \
            2>&1 | tee "$NF_LOG"
        ;;

    mcmicro)
        echo "========================================="
        echo " BioSET Pipeline - MCMICRO Mode"
        echo "========================================="
        echo " Profile: $PROFILE"
        echo " Work dir: $WORK_DIR"
        echo " Output: $OUTPUT_DIR"
        echo "========================================="

        nextflow run "$PROJECT_DIR/main.nf" \
            -profile "$PROFILE" \
            -w "$WORK_DIR" \
            --mode mcmicro \
            --outdir "$OUTPUT_DIR" \
            "$@" \
            -with-report "$OUTPUT_DIR/report.html" \
            -with-timeline "$OUTPUT_DIR/timeline.html" \
            -with-trace "$OUTPUT_DIR/trace.txt" \
            2>&1 | tee "$NF_LOG"
        ;;

    resume)
        echo "========================================="
        echo " BioSET Pipeline - Resuming"
        echo "========================================="

        nextflow run "$PROJECT_DIR/main.nf" \
            -profile "$PROFILE" \
            -w "$WORK_DIR" \
            -resume \
            --outdir "$OUTPUT_DIR" \
            "$@" \
            2>&1 | tee -a "$NF_LOG"
        ;;

    *)
        echo "Usage: $0 {legacy|mcmicro|resume} [nextflow params...]"
        echo ""
        echo "Environment variables:"
        echo "  NF_PROFILE     Nextflow profile (default: nyu_hpc)"
        echo "  NF_WORK_DIR    Work directory (default: /scratch/\$USER/nextflow_work)"
        echo "  NF_OUTPUT_DIR  Output directory (default: /scratch/\$USER/bioset_results)"
        exit 1
        ;;
esac

echo ""
echo "========================================="
echo " Pipeline complete!"
echo " Output: $OUTPUT_DIR"
echo " Log: $NF_LOG"
echo "========================================="

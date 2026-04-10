#!/usr/bin/env nextflow

/*
 * BioSET Preprocessing Pipeline
 * ==============================
 *
 * Nextflow pipeline for GPU-accelerated spatial overlap analysis.
 * Supports two input modes:
 *   - legacy:  raw OME-Zarr → thresholding → overlap analysis
 *   - mcmicro: MCMICRO outputs → adapter → overlap analysis
 *
 * Usage:
 *   # Legacy mode
 *   nextflow run main.nf --mode legacy \
 *       --zarr_path /data/rechunked.zarr \
 *       --zarr_url "https://..." \
 *       --metadata_url "https://..." \
 *       --channels "0,1,2,...,69"
 *
 *   # MCMICRO mode
 *   nextflow run main.nf --mode mcmicro \
 *       --mcmicro_project /data/mcmicro_project \
 *       --marker_thresholds configs/mcmicro/marker_thresholds.yml
 *
 *   # Resume after interruption
 *   nextflow run main.nf -resume
 */

nextflow.enable.dsl=2

// =========================================================================
// Parameters
// =========================================================================

params.mode             = 'legacy'      // 'legacy' or 'mcmicro'

// --- Legacy mode ---
params.zarr_path        = null          // Local OME-Zarr path
params.zarr_url         = null          // Remote OME-Zarr URL
params.metadata_url     = ''            // OME-XML metadata URL
params.channels         = ''            // Comma-separated channel indices

// --- MCMICRO mode ---
params.mcmicro_project  = null          // MCMICRO project directory
params.segmentation     = null          // Path to segmentation mask
params.quantification   = null          // Path to quantification CSV
params.marker_thresholds = null         // Path to marker_thresholds.yml
params.channel_names    = ''            // Comma-separated channel names

// --- Zarr conversion ---
params.source_tiff      = null          // OME-TIFF to convert (mcmicro mode)
params.source_zarr_url  = null          // Remote zarr to rechunk (legacy mode)
params.converted_zarr   = 'zarr_converted/rechunked.zarr'

// --- Pipeline parameters ---
params.tile_size        = '128'
params.channel_batch    = 70
params.alpha            = 0.4
params.trim_q           = 0.98
params.voxel_size       = '0.14,0.14,0.28'
params.min_obj_vol_um3  = 1.0
params.dilate_um        = '0,0.5,1.0,1.5,2.0'
params.max_set_size     = 4
params.min_marker_vox   = 100
params.min_support_pair = 50
params.min_support_set  = 10
params.hierarchy_levels = 4

// --- Output ---
params.output_dir       = 'results'
params.output_name      = 'analysis'
params.checkpoint_dir   = 'checkpoints'
params.outdir           = 'bioset_results'


// =========================================================================
// Process 1: MCMICRO Upstream (optional, mcmicro mode only)
// =========================================================================

process MCMICRO_UPSTREAM {
    label 'cpu_large'
    publishDir "${params.outdir}/mcmicro_outputs", mode: 'copy'

    when:
    params.mode == 'mcmicro' && params.mcmicro_project != null && params.segmentation == null

    input:
    path mcmicro_project

    output:
    path "mcmicro_outputs/segmentation/*", emit: segmentation
    path "mcmicro_outputs/quantification/*", emit: quantification
    path "mcmicro_outputs/registration/*", emit: registration

    script:
    """
    # Run MCMICRO pipeline on the project directory
    # This is a placeholder — adapt to your MCMICRO installation
    nextflow run labsyspharm/mcmicro \
        --in ${mcmicro_project} \
        --start-at registration \
        --stop-at quantification \
        -profile ${params.mcmicro_profile ?: 'standard'}

    # Copy outputs to expected locations
    mkdir -p mcmicro_outputs/{segmentation,quantification,registration}
    cp ${mcmicro_project}/segmentation/* mcmicro_outputs/segmentation/ || true
    cp ${mcmicro_project}/quantification/* mcmicro_outputs/quantification/ || true
    cp ${mcmicro_project}/registration/* mcmicro_outputs/registration/ || true
    """
}


// =========================================================================
// Process 2: TIFF-to-Zarr / Rechunk Zarr
// =========================================================================

process CONVERT_TO_ZARR {
    label 'cpu_large'
    publishDir "${params.outdir}/zarr_converted", mode: 'copy'

    input:
    val source_path

    output:
    path "rechunked.zarr", emit: zarr

    script:
    if (params.source_tiff != null)
        """
        bioset convert \
            --source ${source_path} \
            --target rechunked.zarr \
            --tile ${params.tile_size} \
            --from-tiff \
            --overwrite
        """
    else
        """
        bioset convert \
            --source ${source_path} \
            --target rechunked.zarr \
            --tile ${params.tile_size} \
            --overwrite
        """
}


// =========================================================================
// Process 3: Global Thresholds (legacy mode only)
// =========================================================================

process GLOBAL_THRESHOLDS {
    label 'cpu_medium'

    when:
    params.mode == 'legacy'

    input:
    val zarr_path

    output:
    path "global_thresholds.json", emit: thresholds

    script:
    """
    #!/usr/bin/env python3
    import json
    import numpy as np
    from bioset_preprocessing.config import PipelineConfig, VoxelSizeUM
    from bioset_preprocessing.io import ZarrPyramid
    from bioset_preprocessing.stages.threshold import AlphaThreshold

    channels = [int(c) for c in "${params.channels}".split(",") if c.strip()]
    vx, vy, vz = [float(v) for v in "${params.voxel_size}".split(",")]

    zarr_location = "${params.zarr_url}" if "${params.zarr_url}" != "null" else "${zarr_path}"
    pyr = ZarrPyramid.open(zarr_location)

    # Use lowest resolution for global thresholds
    if len(pyr.arrays) > 1:
        _, A_lo = pyr.lowest_res()
    else:
        _, A_lo = pyr.highest_res()

    th = AlphaThreshold(alpha=${params.alpha}, trim_q=${params.trim_q})

    thresholds = {}
    for ch in channels:
        vol_lr = A_lo[0, ch, :, :, :].compute().astype(np.float32)
        thresholds[str(ch)] = float(th.compute_global(vol_lr))
        print(f"  Channel {ch}: threshold = {thresholds[str(ch)]:.4f}")

    with open("global_thresholds.json", "w") as f:
        json.dump(thresholds, f)

    print(f"Computed thresholds for {len(thresholds)} channels")
    """
}


// =========================================================================
// Process 4: GPU Tile Processing
// =========================================================================

process GPU_TILE_PROCESSING {
    label 'gpu'
    publishDir "${params.outdir}/bioset_checkpoints", mode: 'copy', pattern: 'checkpoints/**'

    input:
    val zarr_path
    path thresholds_file  // global_thresholds.json (legacy) or empty (mcmicro)

    output:
    path "checkpoints/${params.output_name}/*", emit: checkpoints

    script:
    if (params.mode == 'legacy')
        """
        bioset run \
            --zarr-path ${zarr_path} \
            ${params.zarr_url != null ? "--zarr-url ${params.zarr_url}" : ""} \
            --meta "${params.metadata_url}" \
            --channels "${params.channels}" \
            --tile ${params.tile_size} \
            --batch ${params.channel_batch} \
            --alpha ${params.alpha} \
            --trim-q ${params.trim_q} \
            --vox ${params.voxel_size} \
            --min-vol-um3 ${params.min_obj_vol_um3} \
            --dilate-um ${params.dilate_um} \
            --max-set-size ${params.max_set_size} \
            --min-marker-vox ${params.min_marker_vox} \
            --min-support-pair ${params.min_support_pair} \
            --min-support-set ${params.min_support_set} \
            --hierarchy-levels ${params.hierarchy_levels} \
            --output-dir results \
            --output-name ${params.output_name} \
            --checkpoint-dir checkpoints \
            --stage gpu
        """
    else
        """
        bioset mcmicro-run \
            --zarr-path ${zarr_path} \
            --segmentation ${params.segmentation} \
            --quantification ${params.quantification} \
            --marker-thresholds ${params.marker_thresholds} \
            ${params.channel_names ? "--channel-names ${params.channel_names}" : ''} \
            --tile ${params.tile_size} \
            --batch ${params.channel_batch} \
            --vox ${params.voxel_size} \
            --dilate-um ${params.dilate_um} \
            --max-set-size ${params.max_set_size} \
            --min-marker-vox ${params.min_marker_vox} \
            --min-support-pair ${params.min_support_pair} \
            --min-support-set ${params.min_support_set} \
            --hierarchy-levels ${params.hierarchy_levels} \
            --output-dir results \
            --output-name ${params.output_name} \
            --checkpoint-dir checkpoints \
            --stage gpu
        """
}


// =========================================================================
// Process 5: CPU Aggregation
// =========================================================================

process CPU_AGGREGATION {
    label 'cpu_large'
    publishDir "${params.outdir}/bioset_results", mode: 'copy'

    input:
    path zarr_path
    path checkpoints

    output:
    path "results/*.bioset", emit: bioset

    script:
    if (params.mode == 'legacy')
        """
        bioset run \
            --zarr-path ${zarr_path} \
            ${params.zarr_url != null ? "--zarr-url ${params.zarr_url}" : "--zarr-path ${zarr_path}"} \
            --meta "${params.metadata_url}" \
            --channels "${params.channels}" \
            --tile ${params.tile_size} \
            --batch ${params.channel_batch} \
            --alpha ${params.alpha} \
            --trim-q ${params.trim_q} \
            --vox ${params.voxel_size} \
            --min-vol-um3 ${params.min_obj_vol_um3} \
            --dilate-um ${params.dilate_um} \
            --max-set-size ${params.max_set_size} \
            --min-marker-vox ${params.min_marker_vox} \
            --min-support-pair ${params.min_support_pair} \
            --min-support-set ${params.min_support_set} \
            --hierarchy-levels ${params.hierarchy_levels} \
            --output-dir results \
            --output-name ${params.output_name} \
            --checkpoint-dir checkpoints \
            --stage cpu
        """
    else
        """
        bioset mcmicro-run \
            --zarr-path ${zarr_path} \
            --segmentation ${params.segmentation} \
            --quantification ${params.quantification} \
            --marker-thresholds ${params.marker_thresholds} \
            ${params.channel_names ? "--channel-names ${params.channel_names}" : ''} \
            --tile ${params.tile_size} \
            --batch ${params.channel_batch} \
            --vox ${params.voxel_size} \
            --dilate-um ${params.dilate_um} \
            --max-set-size ${params.max_set_size} \
            --min-marker-vox ${params.min_marker_vox} \
            --min-support-pair ${params.min_support_pair} \
            --min-support-set ${params.min_support_set} \
            --hierarchy-levels ${params.hierarchy_levels} \
            --output-dir results \
            --output-name ${params.output_name} \
            --checkpoint-dir checkpoints \
            --stage cpu
        """
}


// =========================================================================
// Workflow
// =========================================================================

workflow {

    // Determine zarr source
    if (params.zarr_path != null) {
        zarr_ch = Channel.of(params.zarr_path)
    } else if (params.source_tiff != null || params.source_zarr_url != null) {
        // Need conversion
        source = params.source_tiff ?: params.source_zarr_url
        CONVERT_TO_ZARR(source)
        zarr_ch = CONVERT_TO_ZARR.out.zarr
    } else if (params.zarr_url != null) {
        // Remote zarr - create a dummy path channel; pipeline.py handles remote access
        zarr_ch = Channel.of(file('.'))
    } else {
        error "No zarr source specified. Provide --zarr_path, --source_tiff, --source_zarr_url, or --zarr_url"
    }

    // Run MCMICRO upstream if needed
    if (params.mode == 'mcmicro' && params.mcmicro_project != null && params.segmentation == null) {
        MCMICRO_UPSTREAM(Channel.fromPath(params.mcmicro_project))
    }

    // Global thresholds (legacy mode only)
    if (params.mode == 'legacy') {
        GLOBAL_THRESHOLDS(zarr_ch)
        thresholds_ch = GLOBAL_THRESHOLDS.out.thresholds
    } else {
        // MCMICRO mode: no global thresholds needed
        thresholds_ch = Channel.of(file('NO_THRESHOLDS'))
    }

    // GPU tile processing
    GPU_TILE_PROCESSING(zarr_ch, thresholds_ch)

    // CPU aggregation
    CPU_AGGREGATION(zarr_ch, GPU_TILE_PROCESSING.out.checkpoints.collect())
}

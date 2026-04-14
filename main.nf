#!/usr/bin/env nextflow

/*
 * BioSET Preprocessing Pipeline
 *
 * Two-stage pipeline: GPU tile processing → CPU aggregation.
 * Checkpoints are written to an absolute shared directory so both
 * stages can find them regardless of Nextflow work-dir hashing.
 *
 * Usage:
 *   nextflow run main.nf -profile nyu_torch \
 *       --mode legacy \
 *       --zarr_path /scratch/$USER/BIOSET-NEXTFLOW/data/rechunked.zarr \
 *       --zarr_url  "https://..." \
 *       --metadata_url "https://.../METADATA.ome.xml" \
 *       --channels "0,1,...,69" \
 *       --channel_batch 70 \
 *       -resume
 */

nextflow.enable.dsl=2

// =========================================================================
// Parameters
// =========================================================================

params.mode             = 'legacy'
params.zarr_path        = null
params.zarr_url         = null
params.metadata_url     = ''
params.channels         = ''

params.mcmicro_project  = null
params.segmentation     = null
params.quantification   = null
params.marker_thresholds = null
params.channel_names    = ''

params.source_tiff      = null
params.source_zarr_url  = null
params.converted_zarr   = 'zarr_converted/rechunked.zarr'

params.tile_size        = '128'
params.channel_batch    = 70
params.alpha            = 0.4
params.trim_q           = 1.0
params.voxel_size       = '0.14,0.14,0.28'
params.min_obj_vol_um3  = 1.0
params.dilate_um        = '0,0.5,1.0,1.5,2.0'
params.max_set_size     = 4
params.min_marker_vox   = 20
params.min_support_pair = 10
params.min_support_set  = 10
params.hierarchy_levels = 4

params.output_name      = 'analysis'
params.outdir           = "${launchDir}/results"

// Absolute checkpoint directory shared between GPU and CPU stages.
// Using launchDir (the directory where nextflow is invoked) keeps
// this stable across re-runs and separate from per-run work dirs.
params.checkpoint_dir   = "${launchDir}/checkpoints"

// MCMICRO mode extras
params.mcmicro_dir      = null

// =========================================================================
// Process: CONVERT_TO_ZARR (optional rechunking)
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
// Process: GPU_TILE_PROCESSING
//
// Key design decisions vs old version:
//   - zarr_path and checkpoint_dir are `val` (plain strings), NOT `path`.
//     Using `path` would cause Nextflow to stage them as files, breaking
//     directory-based zarr access and absolute checkpoint paths.
//   - publishDir copies checkpoints to params.checkpoint_dir (absolute).
//     This survives Nextflow work-dir cleanup and is visible to CPU stage.
//   - Emits checkpoint_dir as a val so CPU_AGGREGATION waits on this
//     process completing before running (ordering guarantee).
// =========================================================================

process GPU_TILE_PROCESSING {
    label 'gpu'

    // Copy checkpoints to the shared absolute directory.
    // saveAs preserves the melanoma_in_situ/ subdirectory structure.
    publishDir "${params.checkpoint_dir}", mode: 'copy', overwrite: true,
               saveAs: { filename -> filename }

    input:
    val zarr_path
    val checkpoint_dir

    output:
    val  checkpoint_dir,              emit: checkpoint_dir_val
    path "${params.output_name}/*",   emit: checkpoint_files

    script:
    def zarr_arg = zarr_path ? "--zarr-path \"${zarr_path}\"" : ""
    def url_arg  = params.zarr_url  ? "--zarr-url \"${params.zarr_url}\"" : ""
    def meta_arg = params.metadata_url ? "--meta \"${params.metadata_url}\"" : ""
    def chan_arg  = params.channels ? "--channels \"${params.channels}\"" : ""

    if (params.mode == 'legacy')
        """
        mkdir -p ${params.output_name}

        bioset run \
            ${zarr_arg} \
            ${url_arg} \
            ${meta_arg} \
            ${chan_arg} \
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
            --output-dir . \
            --output-name ${params.output_name} \
            --checkpoint-dir ${params.output_name} \
            --stage gpu
        """
    else
        """
        mkdir -p ${params.output_name}

        bioset mcmicro-run \
            ${zarr_arg} \
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
            --output-dir . \
            --output-name ${params.output_name} \
            --checkpoint-dir ${params.output_name} \
            --stage gpu
        """
}

// =========================================================================
// Process: CPU_AGGREGATION
//
// Receives checkpoint_dir as a val string (the absolute shared path).
// Passes it directly to --checkpoint-dir so bioset finds the GPU tiles.
// Also receives --meta so real marker names appear in the .bioset output
// instead of generic ch0, ch1, ...
// =========================================================================

process CPU_AGGREGATION {
    label 'cpu_large'
    publishDir "${params.outdir}", mode: 'copy'

    input:
    val checkpoint_dir

    output:
    path "${params.output_name}.bioset", emit: bioset

    script:
    def zarr_arg = params.zarr_path  ? "--zarr-path \"${params.zarr_path}\"" : ""
    def url_arg  = params.zarr_url   ? "--zarr-url \"${params.zarr_url}\"" : ""
    def meta_arg = params.metadata_url ? "--meta \"${params.metadata_url}\"" : ""
    def chan_arg  = params.channels  ? "--channels \"${params.channels}\"" : ""

    if (params.mode == 'legacy')
        """
        bioset run \
            ${zarr_arg} \
            ${url_arg} \
            ${meta_arg} \
            ${chan_arg} \
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
            --output-dir . \
            --output-name ${params.output_name} \
            --checkpoint-dir "${checkpoint_dir}/${params.output_name}" \
            --stage cpu
        """
    else
        """
        bioset mcmicro-run \
            ${zarr_arg} \
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
            --output-dir . \
            --output-name ${params.output_name} \
            --checkpoint-dir "${checkpoint_dir}/${params.output_name}" \
            --stage cpu
        """
}

// =========================================================================
// Workflow
// =========================================================================

workflow {

    // Resolve zarr source
    if (params.zarr_path != null) {
        zarr_ch = Channel.of(params.zarr_path)
    } else if (params.source_tiff != null || params.source_zarr_url != null) {
        source = params.source_tiff ?: params.source_zarr_url
        CONVERT_TO_ZARR(source)
        zarr_ch = CONVERT_TO_ZARR.out.zarr.map { it.toString() }
    } else if (params.zarr_url != null) {
        zarr_ch = Channel.of('')   // bioset will use --zarr-url flag
    } else {
        error "No zarr source. Provide --zarr_path, --source_tiff, --source_zarr_url, or --zarr_url"
    }

    // Absolute checkpoint dir passed as val to both stages
    checkpoint_dir_ch = Channel.of(params.checkpoint_dir)

    // Stage 1: GPU tile processing
    GPU_TILE_PROCESSING(zarr_ch, checkpoint_dir_ch)

    // Stage 2: CPU aggregation — depends on GPU via checkpoint_dir_val
    CPU_AGGREGATION(GPU_TILE_PROCESSING.out.checkpoint_dir_val)

    // Print output path when done
    CPU_AGGREGATION.out.bioset.view { f ->
        "\n✅  Pipeline complete! Output: ${f}\n"
    }
}
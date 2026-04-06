import argparse
from .config import PipelineConfig, VoxelSizeUM
from .pipeline import Pipeline


def parse_int_list(s: str):
    return [int(x.strip()) for x in s.split(",") if x.strip() != ""]


def parse_tile(s: str):
    if "x" in s:
        a, b = s.lower().split("x")
        return (int(a), int(b))
    n = int(s)
    return (n, n)


def _add_common_args(parser: argparse.ArgumentParser) -> None:
    """Add arguments shared between legacy and mcmicro modes."""
    parser.add_argument("--tile", default="128", help="Tile size: N or NyxNx")
    parser.add_argument("--batch", type=int, default=8, help="Channels per GPU batch")
    parser.add_argument("--dilate-um", default="0,0.5,1.0,1.5,2.0")
    parser.add_argument("--vox", default="0.14,0.14,0.28", help="Voxel size x,y,z µm")
    parser.add_argument("--float64-dist", action="store_true")
    parser.add_argument("--max-set-size", type=int, default=4)
    parser.add_argument("--min-marker-vox", type=int, default=100)
    parser.add_argument("--min-support-pair", type=int, default=50)
    parser.add_argument("--min-support-set", type=int, default=10)
    parser.add_argument("--hierarchy-levels", type=int, default=4)
    parser.add_argument("--output-dir", default="results")
    parser.add_argument("--output-name", default="analysis")
    parser.add_argument("--checkpoint-dir", default="checkpoints")
    parser.add_argument("--no-resume", action="store_true", help="Don't resume from checkpoints")

    # Stage selection
    parser.add_argument(
        "--stage",
        choices=["full", "gpu", "cpu"],
        default="full",
        help="Run stage: full (both), gpu (tile processing only), cpu (aggregation only)",
    )


def main():
    p = argparse.ArgumentParser(prog="bioset")
    sub = p.add_subparsers(dest="cmd", required=True)

    # ----------------------------------------------------------------
    # Legacy mode: bioset run
    # ----------------------------------------------------------------
    run = sub.add_parser("run", help="Run legacy BioSET pipeline (threshold-based)")

    run.add_argument("--zarr-url", help="Remote OME-Zarr root URL")
    run.add_argument("--zarr-path", help="Local OME-Zarr root directory")
    run.add_argument("--meta", required=True, help="OME-XML metadata URL")
    run.add_argument("--channels", required=True, help="Comma-separated channel indices")
    run.add_argument("--alpha", type=float, default=0.4)
    run.add_argument("--trim-q", type=float, default=0.98)
    run.add_argument("--min-vol-um3", type=float, default=1.0)
    run.add_argument("--conn", type=int, default=26)
    _add_common_args(run)

    # ----------------------------------------------------------------
    # MCMICRO mode: bioset mcmicro-run
    # ----------------------------------------------------------------
    mcr = sub.add_parser("mcmicro-run", help="Run BioSET with MCMICRO adapter")

    mcr.add_argument(
        "--zarr-path", required=True,
        help="Path to converted zarr (registered image)",
    )
    mcr.add_argument(
        "--segmentation", required=True,
        help="Path to MCMICRO segmentation mask (TIFF or zarr)",
    )
    mcr.add_argument(
        "--quantification", required=True,
        help="Path to MCMICRO quantification CSV",
    )
    mcr.add_argument(
        "--marker-thresholds", required=True,
        help="Path to marker_thresholds.yml",
    )
    mcr.add_argument(
        "--channel-names",
        help="Comma-separated channel names (optional; inferred from thresholds if omitted)",
    )
    _add_common_args(mcr)

    # ----------------------------------------------------------------
    # Zarr conversion: bioset convert
    # ----------------------------------------------------------------
    conv = sub.add_parser("convert", help="Convert TIFF to rechunked zarr")

    conv.add_argument("--source", required=True, help="Source TIFF or zarr path/URL")
    conv.add_argument("--target", required=True, help="Target zarr output path")
    conv.add_argument("--tile", default="128", help="Tile size for rechunking")
    conv.add_argument(
        "--from-tiff", action="store_true",
        help="Source is OME-TIFF (default: treat as zarr)",
    )
    conv.add_argument("--component", default="0", help="Zarr component (for zarr sources)")
    conv.add_argument("--overwrite", action="store_true")

    # ----------------------------------------------------------------
    # Validation: bioset validate
    # ----------------------------------------------------------------
    val = sub.add_parser("validate", help="Compare two .bioset files")

    val.add_argument("--baseline", required=True, help="Baseline .bioset file")
    val.add_argument("--candidate", required=True, help="Candidate .bioset file")
    val.add_argument("--dilation", type=float, default=2.0)
    val.add_argument("--top-k", type=int, default=50)
    val.add_argument("--min-overlap", type=float, default=0.8)
    val.add_argument("--min-pearson", type=float, default=0.90)

    # ----------------------------------------------------------------
    # Parse and dispatch
    # ----------------------------------------------------------------
    args = p.parse_args()

    if args.cmd == "run":
        _run_legacy(args, run)
    elif args.cmd == "mcmicro-run":
        _run_mcmicro(args)
    elif args.cmd == "convert":
        _run_convert(args)
    elif args.cmd == "validate":
        _run_validate(args)


def _run_legacy(args, parser):
    """Execute legacy (threshold-based) pipeline."""
    if not args.zarr_url and not args.zarr_path:
        parser.error("Provide at least one of --zarr-url or --zarr-path")

    channels = parse_int_list(args.channels)
    tile_xy = parse_tile(args.tile)
    dilate_um = [float(x) for x in args.dilate_um.split(",") if x.strip()]
    vx, vy, vz = [float(x) for x in args.vox.split(",")]

    cfg = PipelineConfig(
        input_mode="legacy",
        zarr_url=args.zarr_url,
        zarr_path=args.zarr_path,
        metadata_url=args.meta,
        channels=channels,
        tile_xy=tile_xy,
        channel_batch=args.batch,
        alpha=args.alpha,
        trim_q=args.trim_q,
        voxel_size_um=VoxelSizeUM(vx, vy, vz),
        min_obj_vol_um3=args.min_vol_um3,
        connectivity=args.conn,
        dilate_um=tuple(dilate_um),
        float64_distances=args.float64_dist,
        max_set_size=args.max_set_size,
        min_marker_vox=args.min_marker_vox,
        min_support_pair=args.min_support_pair,
        min_support_set=args.min_support_set,
        hierarchy_levels=args.hierarchy_levels,
        output_dir=args.output_dir,
        output_name=args.output_name,
        checkpoint_dir=args.checkpoint_dir,
    )

    pipe = Pipeline(cfg)
    _execute_stage(pipe, args)


def _run_mcmicro(args):
    """Execute MCMICRO adapter pipeline."""
    tile_xy = parse_tile(args.tile)
    dilate_um = [float(x) for x in args.dilate_um.split(",") if x.strip()]
    vx, vy, vz = [float(x) for x in args.vox.split(",")]

    channel_names = []
    if args.channel_names:
        channel_names = [n.strip() for n in args.channel_names.split(",") if n.strip()]

    cfg = PipelineConfig(
        input_mode="mcmicro",
        mcmicro_zarr_path=args.zarr_path,
        mcmicro_segmentation_path=args.segmentation,
        mcmicro_quantification_path=args.quantification,
        marker_thresholds_path=args.marker_thresholds,
        channel_names=tuple(channel_names),
        tile_xy=tile_xy,
        channel_batch=args.batch,
        voxel_size_um=VoxelSizeUM(vx, vy, vz),
        dilate_um=tuple(dilate_um),
        float64_distances=args.float64_dist,
        max_set_size=args.max_set_size,
        min_marker_vox=args.min_marker_vox,
        min_support_pair=args.min_support_pair,
        min_support_set=args.min_support_set,
        hierarchy_levels=args.hierarchy_levels,
        output_dir=args.output_dir,
        output_name=args.output_name,
        checkpoint_dir=args.checkpoint_dir,
    )

    cfg.validate()
    pipe = Pipeline(cfg)
    _execute_stage(pipe, args)


def _execute_stage(pipe: Pipeline, args):
    """Run the appropriate pipeline stage based on --stage flag."""
    resume = not args.no_resume

    if args.stage == "gpu":
        tiles = pipe.run_tile_processing(resume=resume)
        print(f"Done! Processed {tiles} tiles.")
    elif args.stage == "cpu":
        output = pipe.run_aggregation()
        print(f"Done! Output: {output}")
    else:
        output = pipe.run_full_analysis(resume=resume)
        print(f"Done! Output: {output}")


def _run_convert(args):
    """Execute zarr conversion."""
    from .converters.tiff_to_zarr import convert_tiff_to_zarr, rechunk_zarr

    tile_xy = parse_tile(args.tile)

    if args.from_tiff:
        convert_tiff_to_zarr(
            source_tiff=args.source,
            target_zarr=args.target,
            tile_size=tile_xy,
            overwrite=args.overwrite,
        )
    else:
        rechunk_zarr(
            source_zarr=args.source,
            target_zarr=args.target,
            tile_size=tile_xy,
            component=args.component,
            overwrite=args.overwrite,
        )

    print(f"Done! Output: {args.target}")


def _run_validate(args):
    """Execute validation comparison."""
    from .validation import compare_biosets

    result = compare_biosets(
        baseline_path=args.baseline,
        candidate_path=args.candidate,
        dilation=args.dilation,
        top_k=args.top_k,
        min_overlap_fraction=args.min_overlap,
        min_pearson_r=args.min_pearson,
    )

    print(result.summary())

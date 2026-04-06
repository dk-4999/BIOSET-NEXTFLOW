#!/usr/bin/env python3
"""
Generate a valid MCMICRO project layout from raw microscopy data.

Usage:
    python make_mcmicro_project.py \
        --raw-dir /data/raw_images \
        --output-dir /data/mcmicro_project \
        --markers configs/mcmicro/markers.csv \
        --params configs/mcmicro/params.yml

This creates the directory structure expected by MCMICRO:

    mcmicro_project/
    ├── raw/
    │   └── <symlinks to raw images>
    ├── markers.csv
    └── params.yml
"""
import argparse
import os
import shutil
from pathlib import Path


def make_project(
    raw_dir: Path,
    output_dir: Path,
    markers_csv: Path,
    params_yml: Path,
    symlink: bool = True,
) -> None:
    """Create a valid MCMICRO project directory."""
    output_dir = Path(output_dir)
    raw_dir = Path(raw_dir)

    # Create directory structure
    raw_out = output_dir / "raw"
    raw_out.mkdir(parents=True, exist_ok=True)

    # Create expected subdirectories
    for subdir in ["illumination", "registration", "segmentation", "quantification"]:
        (output_dir / subdir).mkdir(exist_ok=True)

    # Link or copy raw images
    image_exts = {".tif", ".tiff", ".ome.tif", ".ome.tiff", ".nd2", ".czi"}
    n_images = 0

    for f in sorted(raw_dir.iterdir()):
        if f.suffix.lower() in image_exts or f.name.endswith(".ome.tif"):
            target = raw_out / f.name
            if not target.exists():
                if symlink:
                    target.symlink_to(f.resolve())
                else:
                    shutil.copy2(f, target)
                n_images += 1

    print(f"Linked {n_images} images to {raw_out}")

    # Copy markers.csv
    markers_dst = output_dir / "markers.csv"
    shutil.copy2(markers_csv, markers_dst)
    print(f"Copied markers.csv to {markers_dst}")

    # Copy params.yml
    params_dst = output_dir / "params.yml"
    shutil.copy2(params_yml, params_dst)
    print(f"Copied params.yml to {params_dst}")

    print(f"\nMCMICRO project created at: {output_dir}")
    print(f"Structure:")
    for root, dirs, files in os.walk(output_dir):
        level = root.replace(str(output_dir), "").count(os.sep)
        indent = "  " * level
        print(f"{indent}{os.path.basename(root)}/")
        subindent = "  " * (level + 1)
        for f in files:
            print(f"{subindent}{f}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate MCMICRO project layout",
    )
    parser.add_argument(
        "--raw-dir", required=True,
        help="Directory containing raw microscopy images",
    )
    parser.add_argument(
        "--output-dir", required=True,
        help="Output directory for MCMICRO project",
    )
    parser.add_argument(
        "--markers", required=True,
        help="Path to markers.csv",
    )
    parser.add_argument(
        "--params", required=True,
        help="Path to params.yml",
    )
    parser.add_argument(
        "--copy", action="store_true",
        help="Copy files instead of symlinking",
    )

    args = parser.parse_args()

    make_project(
        raw_dir=Path(args.raw_dir),
        output_dir=Path(args.output_dir),
        markers_csv=Path(args.markers),
        params_yml=Path(args.params),
        symlink=not args.copy,
    )


if __name__ == "__main__":
    main()

"""
TIFF-to-Zarr Converter
======================

Converts OME-TIFF files (as produced by MCMICRO) or existing OME-Zarr stores
into rechunked zarr stores aligned to BioSET's tile size.

This is a critical step because:
- BioSET's I/O layer (``io.py``) assumes zarr random-access tile reads
- The tiling module requires chunk-aligned access for performance
- MCMICRO outputs OME-TIFF which does not support efficient random tile access

Usage
-----
::

    from bioset_preprocessing.converters.tiff_to_zarr import (
        convert_tiff_to_zarr,
        rechunk_zarr,
    )

    # From MCMICRO OME-TIFF
    convert_tiff_to_zarr(
        source_tiff="/path/to/registered.ome.tif",
        target_zarr="/path/to/converted.zarr",
        tile_size=(128, 128),
    )

    # Rechunk existing zarr
    rechunk_zarr(
        source_zarr="/path/to/original.zarr",
        target_zarr="/path/to/rechunked.zarr",
        tile_size=(128, 128),
    )
"""
from __future__ import annotations

import os
import shutil
import time
from pathlib import Path
from typing import Optional, Tuple

import numpy as np


def convert_tiff_to_zarr(
    source_tiff: str | Path,
    target_zarr: str | Path,
    tile_size: Tuple[int, int] = (128, 128),
    max_mem: str = "4GB",
    overwrite: bool = False,
) -> Path:
    """
    Convert an OME-TIFF file to a rechunked zarr store.

    The output zarr will have chunks aligned to ``(1, 1, Z, tile_y, tile_x)``
    where Z is the full depth of the volume.

    Parameters
    ----------
    source_tiff : str or Path
        Path to the input OME-TIFF file.
    target_zarr : str or Path
        Path for the output zarr directory.
    tile_size : tuple of int
        (tile_y, tile_x) in pixels — must match BioSET's ``tile_xy`` config.
    max_mem : str
        Maximum memory for rechunking (passed to dask).
    overwrite : bool
        If True, delete existing target zarr.

    Returns
    -------
    Path
        Path to the created zarr store.
    """
    try:
        import tifffile
    except ImportError:
        raise ImportError("tifffile is required: pip install tifffile")

    import zarr
    import dask.array as da

    source_tiff = Path(source_tiff)
    target_zarr = Path(target_zarr)

    if not source_tiff.exists():
        raise FileNotFoundError(f"Source TIFF not found: {source_tiff}")

    if target_zarr.exists():
        if overwrite:
            shutil.rmtree(target_zarr)
        else:
            raise FileExistsError(
                f"Target zarr already exists: {target_zarr}. "
                f"Use overwrite=True or delete manually."
            )

    print(f"[tiff_to_zarr] Reading: {source_tiff}")
    start = time.perf_counter()

    # Read TIFF lazily via tifffile + dask
    img = tifffile.imread(str(source_tiff), aszarr=True)
    arr = da.from_zarr(img)

    print(f"[tiff_to_zarr] Source shape: {arr.shape}, dtype: {arr.dtype}")

    # Normalise to 5D: (T, C, Z, Y, X)
    arr = _normalise_5d(arr)
    _, n_channels, n_z, n_y, n_x = arr.shape

    print(f"[tiff_to_zarr] Normalised 5D shape: {arr.shape}")

    # Target chunks: (1, 1, full_Z, tile_y, tile_x)
    tile_y, tile_x = tile_size
    target_chunks = (1, 1, n_z, tile_y, tile_x)

    print(f"[tiff_to_zarr] Target chunks: {target_chunks}")
    print(f"[tiff_to_zarr] Rechunking...")

    # Rechunk via dask
    rechunked = arr.rechunk(target_chunks)

    # Ensure parent directory exists
    target_zarr.parent.mkdir(parents=True, exist_ok=True)

    # Write
    rechunked.to_zarr(str(target_zarr), overwrite=True)

    elapsed = time.perf_counter() - start
    print(f"[tiff_to_zarr] Done in {elapsed:.1f}s ({elapsed / 60:.1f} min)")

    # Verify
    out = zarr.open(str(target_zarr), mode="r")
    print(f"[tiff_to_zarr] Output shape: {out.shape}, chunks: {out.chunks}")

    return target_zarr


def rechunk_zarr(
    source_zarr: str | Path,
    target_zarr: str | Path,
    tile_size: Tuple[int, int] = (128, 128),
    component: str = "0",
    max_mem: str = "4GB",
    overwrite: bool = False,
    use_rechunker: bool = False,
) -> Path:
    """
    Rechunk an existing OME-Zarr store to align with BioSET's tile size.

    This replaces the ad-hoc ``examples/rechunking.py`` with a proper utility.

    Parameters
    ----------
    source_zarr : str or Path
        Path or URL to the source zarr.
    target_zarr : str or Path
        Path for the output rechunked zarr.
    tile_size : tuple of int
        (tile_y, tile_x) in pixels.
    component : str
        Zarr component to read (default ``"0"`` = highest resolution).
    max_mem : str
        Maximum memory for rechunking.
    overwrite : bool
        If True, delete existing target.
    use_rechunker : bool
        If True, use the ``rechunker`` library (requires ``pip install rechunker``).
        If False (default), use dask rechunking.

    Returns
    -------
    Path
        Path to the rechunked zarr.
    """
    import zarr
    import dask.array as da

    source_zarr = str(source_zarr)
    target_zarr = Path(target_zarr)

    if target_zarr.exists():
        if overwrite:
            shutil.rmtree(target_zarr)
        else:
            raise FileExistsError(f"Target exists: {target_zarr}")

    print(f"[rechunk_zarr] Source: {source_zarr}")

    # Open source
    is_remote = source_zarr.startswith(("http://", "https://", "s3://"))
    if is_remote:
        from ome_zarr.io import parse_url
        root = parse_url(source_zarr, mode="r")
        store = root.store
    else:
        store = zarr.NestedDirectoryStore(source_zarr)

    arr = da.from_zarr(store, component=component)
    print(f"[rechunk_zarr] Source shape: {arr.shape}, chunks: {arr.chunksize}")

    # Normalise to 5D
    arr = _normalise_5d(arr)
    _, n_channels, n_z, n_y, n_x = arr.shape

    tile_y, tile_x = tile_size
    target_chunks = (1, 1, n_z, tile_y, tile_x)

    print(f"[rechunk_zarr] Target chunks: {target_chunks}")

    target_zarr.parent.mkdir(parents=True, exist_ok=True)

    if use_rechunker:
        _rechunk_with_rechunker(arr, target_zarr, target_chunks, max_mem)
    else:
        _rechunk_with_dask(arr, target_zarr, target_chunks)

    out = zarr.open(str(target_zarr), mode="r")
    print(f"[rechunk_zarr] Output shape: {out.shape}, chunks: {out.chunks}")

    return target_zarr


def _normalise_5d(arr) -> "da.Array":
    """Normalise an array to 5D (T, C, Z, Y, X)."""
    import dask.array as da

    if arr.ndim == 2:
        arr = arr[np.newaxis, np.newaxis, np.newaxis, :, :]
    elif arr.ndim == 3:
        arr = arr[np.newaxis, np.newaxis, :, :, :]
    elif arr.ndim == 4:
        arr = arr[np.newaxis, :, :, :, :]
    elif arr.ndim == 5:
        pass
    else:
        raise ValueError(f"Cannot normalise {arr.ndim}D array to 5D")

    return arr


def _rechunk_with_dask(arr, target_zarr, target_chunks):
    """Rechunk using dask (no extra dependencies)."""
    start = time.perf_counter()

    rechunked = arr.rechunk(target_chunks)
    rechunked.to_zarr(str(target_zarr), overwrite=True)

    elapsed = time.perf_counter() - start
    print(f"[rechunk_zarr] Dask rechunk completed in {elapsed:.1f}s")


def _rechunk_with_rechunker(arr, target_zarr, target_chunks, max_mem):
    """Rechunk using the rechunker library (more memory-efficient for large data)."""
    import zarr

    try:
        from rechunker import rechunk
    except ImportError:
        raise ImportError(
            "rechunker is required: pip install rechunker  "
            "or use use_rechunker=False for dask-based rechunking"
        )

    temp_path = str(target_zarr) + "_tmp"
    target_store = zarr.NestedDirectoryStore(str(target_zarr))
    temp_store = zarr.NestedDirectoryStore(temp_path)

    start = time.perf_counter()

    plan = rechunk(
        arr,
        target_chunks=target_chunks,
        max_mem=max_mem,
        target_store=target_store,
        temp_store=temp_store,
    )
    plan.execute()

    elapsed = time.perf_counter() - start
    print(f"[rechunk_zarr] Rechunker completed in {elapsed:.1f}s")

    # Cleanup temp
    if os.path.exists(temp_path):
        shutil.rmtree(temp_path, ignore_errors=True)


def validate_zarr_conversion(
    original_path: str | Path,
    converted_zarr: str | Path,
    n_random_tiles: int = 5,
    tile_size: Tuple[int, int] = (128, 128),
    atol: float = 1e-6,
) -> bool:
    """
    Validate a converted zarr by comparing random tile reads against the source.

    Parameters
    ----------
    original_path : str or Path
        Path to the original file (TIFF or zarr).
    converted_zarr : str or Path
        Path to the converted zarr.
    n_random_tiles : int
        Number of random tiles to compare.
    tile_size : tuple
        (tile_y, tile_x).
    atol : float
        Absolute tolerance for value comparison.

    Returns
    -------
    bool
        True if all tiles match.
    """
    import zarr
    import dask.array as da

    converted_zarr = Path(converted_zarr)
    original_path = Path(original_path)

    # Open converted
    conv = da.from_zarr(str(converted_zarr))
    conv = _normalise_5d(conv)

    # Open original
    suffix = original_path.suffix.lower()
    if suffix in (".tif", ".tiff"):
        import tifffile
        orig_raw = tifffile.imread(str(original_path), aszarr=True)
        orig = da.from_zarr(orig_raw)
    else:
        orig = da.from_zarr(str(original_path), component="0")
    orig = _normalise_5d(orig)

    _, n_c, n_z, n_y, n_x = conv.shape
    tile_y, tile_x = tile_size

    rng = np.random.default_rng(42)
    max_ty = (n_y + tile_y - 1) // tile_y
    max_tx = (n_x + tile_x - 1) // tile_x

    all_pass = True
    for i in range(n_random_tiles):
        ty = rng.integers(0, max_ty)
        tx = rng.integers(0, max_tx)
        ch = rng.integers(0, n_c)

        y0, y1 = ty * tile_y, min((ty + 1) * tile_y, n_y)
        x0, x1 = tx * tile_x, min((tx + 1) * tile_x, n_x)

        tile_conv = conv[0, ch, :, y0:y1, x0:x1].compute()
        tile_orig = orig[0, ch, :, y0:y1, x0:x1].compute()

        if not np.allclose(tile_conv, tile_orig, atol=atol):
            print(f"[validate] MISMATCH at tile ({ty},{tx}) ch={ch}")
            max_diff = float(np.max(np.abs(tile_conv.astype(float) - tile_orig.astype(float))))
            print(f"  Max difference: {max_diff}")
            all_pass = False
        else:
            print(f"[validate] tile ({ty},{tx}) ch={ch}: OK")

    return all_pass

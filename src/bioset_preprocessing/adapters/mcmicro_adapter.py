"""
MCMICRO-to-BioSET Adapter
==========================

Bridges the cell-level outputs of MCMICRO (segmentation masks + quantification
tables) back to the voxel-level binary masks that BioSET's downstream stages
(dilation, overlap mining) expect.

Algorithm
---------
1.  Read the cell segmentation label image (each voxel's value = cell ID, 0 = bg).
2.  Read the quantification CSV (one row per cell, columns for marker intensities).
3.  For each marker, classify cells as positive/negative using a configurable
    threshold from ``marker_thresholds.yml``.
4.  For each positive cell, set all voxels belonging to that cell to ``True``
    in a per-marker boolean mask.
5.  Return ``Dict[int, ndarray[bool]]`` — same shape as the threshold + CC
    pipeline's output, keyed by channel index.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np

try:
    import yaml
except ImportError:
    yaml = None

try:
    import cupy as cp
except ImportError:
    cp = None


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class MarkerThreshold:
    """Positivity threshold for a single marker."""
    marker_name: str
    threshold: float
    column_name: str  # column in the quantification CSV


@dataclass
class CellQuantification:
    """Parsed quantification table."""
    cell_ids: np.ndarray           # shape (N_cells,), int
    marker_columns: List[str]      # column names for each marker
    intensities: np.ndarray        # shape (N_cells, N_markers), float


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def load_marker_thresholds(path: str | Path) -> Dict[str, MarkerThreshold]:
    """
    Load marker thresholds from a YAML file.

    Expected format::

        markers:
          - name: CD8
            column: CD8_mean        # column in quant CSV
            threshold: 0.5
          - name: MART1
            column: MART1_mean
            threshold: 0.3

    Returns a dict keyed by marker *name*.
    """
    if yaml is None:
        raise ImportError("PyYAML is required for MCMICRO mode: pip install pyyaml")

    path = Path(path)
    with open(path) as f:
        data = yaml.safe_load(f)

    thresholds: Dict[str, MarkerThreshold] = {}
    for entry in data.get("markers", []):
        name = str(entry["name"])
        thresholds[name] = MarkerThreshold(
            marker_name=name,
            threshold=float(entry["threshold"]),
            column_name=str(entry.get("column", f"{name}_mean")),
        )
    return thresholds


def load_quantification_csv(
    path: str | Path,
    marker_columns: List[str],
    cell_id_column: str = "CellID",
) -> CellQuantification:
    """
    Parse MCMICRO quantification CSV into structured arrays.

    Parameters
    ----------
    path : str or Path
        Path to the CSV file.
    marker_columns : list of str
        Column names for marker intensities to extract.
    cell_id_column : str
        Column name containing cell IDs (default ``"CellID"``).

    Returns
    -------
    CellQuantification
        Parsed data with cell IDs and intensity matrix.
    """
    path = Path(path)

    cell_ids = []
    intensities = []

    with open(path, newline="") as fh:
        reader = csv.DictReader(fh)

        # Validate columns exist
        available = set(reader.fieldnames or [])
        missing = {cell_id_column} | set(marker_columns) - available
        if missing:
            raise KeyError(
                f"Columns not found in {path.name}: {missing}. "
                f"Available: {sorted(available)}"
            )

        for row in reader:
            cell_ids.append(int(row[cell_id_column]))
            intensities.append([float(row[col]) for col in marker_columns])

    return CellQuantification(
        cell_ids=np.asarray(cell_ids, dtype=np.int32),
        marker_columns=list(marker_columns),
        intensities=np.asarray(intensities, dtype=np.float32),
    )


def load_segmentation_mask(
    path: str | Path,
    tile_slice_y: slice | None = None,
    tile_slice_x: slice | None = None,
) -> np.ndarray:
    """
    Load a segmentation label image (or a tile subregion of it).

    Supports:
    - .tif / .tiff  (via tifffile)
    - .zarr         (via zarr + dask)

    Parameters
    ----------
    path : str or Path
        Path to the segmentation mask file.
    tile_slice_y, tile_slice_x : slice, optional
        If provided, read only this spatial subregion (Z is always read in full).

    Returns
    -------
    np.ndarray
        Label image, shape ``(Z, Y, X)`` or ``(Y, X)`` with integer cell IDs.
    """
    path = Path(path)
    suffix = path.suffix.lower()

    if suffix in (".tif", ".tiff"):
        try:
            import tifffile
        except ImportError:
            raise ImportError("tifffile is required for TIFF mask reading: pip install tifffile")

        img = tifffile.imread(str(path))

        # Normalise to at least 3D (Z, Y, X)
        if img.ndim == 2:
            img = img[np.newaxis, :, :]
        elif img.ndim == 3:
            pass  # assume (Z, Y, X) already
        elif img.ndim == 4:
            # (T, Z, Y, X) or (C, Z, Y, X) — take first
            img = img[0]
        elif img.ndim == 5:
            # (T, C, Z, Y, X) — take first T and C
            img = img[0, 0]
        else:
            raise ValueError(f"Unsupported mask dimensions: {img.ndim}")

        if tile_slice_y is not None and tile_slice_x is not None:
            img = img[:, tile_slice_y, tile_slice_x]

        return img

    elif suffix == ".zarr" or path.is_dir():
        import zarr
        import dask.array as da

        store = zarr.open(str(path), mode="r")

        # Navigate possible zarr layouts
        if isinstance(store, zarr.Array):
            arr = da.from_zarr(store)
        elif "0" in store:
            arr = da.from_zarr(store["0"])
        else:
            arr = da.from_zarr(store)

        # Normalise dimensions
        while arr.ndim > 3:
            arr = arr[0]
        if arr.ndim == 2:
            arr = arr[np.newaxis, :, :]

        if tile_slice_y is not None and tile_slice_x is not None:
            arr = arr[:, tile_slice_y, tile_slice_x]

        return arr.compute()
    else:
        raise ValueError(f"Unsupported segmentation mask format: {suffix}")


# ---------------------------------------------------------------------------
# Core adapter
# ---------------------------------------------------------------------------

class MCMICROAdapter:
    """
    Converts MCMICRO segmentation + quantification outputs into per-marker
    voxel-level boolean masks, compatible with BioSET's downstream pipeline.

    Usage::

        adapter = MCMICROAdapter(
            segmentation_path="/path/to/seg_mask.tif",
            quantification_path="/path/to/quant.csv",
            marker_thresholds_path="/path/to/marker_thresholds.yml",
        )
        adapter.load()

        # For a given tile:
        masks = adapter.get_tile_masks(
            tile_slice_y=slice(0, 128),
            tile_slice_x=slice(0, 128),
        )
        # masks: Dict[int, cp.ndarray]  — channel_idx -> boolean mask (Z, Y, X)
    """

    def __init__(
        self,
        segmentation_path: str | Path,
        quantification_path: str | Path,
        marker_thresholds_path: str | Path,
        channel_names: Sequence[str] | None = None,
    ):
        self.segmentation_path = Path(segmentation_path)
        self.quantification_path = Path(quantification_path)
        self.marker_thresholds_path = Path(marker_thresholds_path)

        # Will be populated by load()
        self._thresholds: Dict[str, MarkerThreshold] = {}
        self._quant: Optional[CellQuantification] = None
        self._positive_cells: Dict[str, Set[int]] = {}  # marker_name -> set of cell IDs
        self._channel_names: List[str] = list(channel_names or [])
        self._channel_name_to_idx: Dict[str, int] = {}
        self._loaded = False

    def load(self) -> None:
        """
        Load thresholds and quantification data.  Determines positive cells
        for each marker.  Must be called before ``get_tile_masks``.
        """
        # 1) Load thresholds
        self._thresholds = load_marker_thresholds(self.marker_thresholds_path)

        # 2) Build column list for markers that have thresholds
        marker_columns = [t.column_name for t in self._thresholds.values()]

        # 3) Load quantification
        self._quant = load_quantification_csv(
            self.quantification_path,
            marker_columns=marker_columns,
        )

        # 4) Determine positive cells per marker
        self._positive_cells = {}
        for marker_name, mt in self._thresholds.items():
            col_idx = self._quant.marker_columns.index(mt.column_name)
            mask = self._quant.intensities[:, col_idx] >= mt.threshold
            pos_ids = set(self._quant.cell_ids[mask].tolist())
            self._positive_cells[marker_name] = pos_ids

        # 5) Build channel name → index mapping
        if not self._channel_names:
            # Default: use marker names from thresholds in sorted order
            self._channel_names = sorted(self._thresholds.keys())

        self._channel_name_to_idx = {
            name: idx for idx, name in enumerate(self._channel_names)
        }

        self._loaded = True

        n_markers = len(self._thresholds)
        n_cells = len(self._quant.cell_ids)
        print(f"[MCMICROAdapter] Loaded: {n_markers} markers, {n_cells} cells")
        for name, pos in self._positive_cells.items():
            pct = 100.0 * len(pos) / max(1, n_cells)
            print(f"  {name}: {len(pos)} positive cells ({pct:.1f}%)")

    @property
    def channel_names(self) -> List[str]:
        return list(self._channel_names)

    @property
    def channel_indices(self) -> List[int]:
        return list(range(len(self._channel_names)))

    def get_tile_masks(
        self,
        tile_slice_y: slice,
        tile_slice_x: slice,
        device: str = "gpu",
    ) -> Dict[int, "cp.ndarray | np.ndarray"]:
        """
        Generate per-marker boolean masks for a single tile.

        Parameters
        ----------
        tile_slice_y, tile_slice_x : slice
            Spatial subregion to read from the segmentation mask.
        device : str
            ``"gpu"`` to return CuPy arrays, ``"cpu"`` for NumPy.

        Returns
        -------
        Dict[int, ndarray]
            Channel index -> boolean mask of shape ``(Z, Y, X)``.
            Same structure as the output of threshold + CC filter.
        """
        if not self._loaded:
            raise RuntimeError("Call adapter.load() before get_tile_masks()")

        # Read label image tile
        label_tile = load_segmentation_mask(
            self.segmentation_path,
            tile_slice_y=tile_slice_y,
            tile_slice_x=tile_slice_x,
        )
        # label_tile: (Z, Y, X) int, where 0 = background

        masks: Dict[int, np.ndarray] = {}

        for marker_name, threshold_spec in self._thresholds.items():
            if marker_name not in self._channel_name_to_idx:
                continue

            ch_idx = self._channel_name_to_idx[marker_name]
            pos_cells = self._positive_cells[marker_name]

            if not pos_cells:
                masks[ch_idx] = np.zeros(label_tile.shape, dtype=np.bool_)
                continue

            # Find which cell IDs are present in this tile
            unique_labels = np.unique(label_tile)
            # Intersect with positive cells
            present_positive = np.array(
                [cid for cid in unique_labels if cid != 0 and cid in pos_cells],
                dtype=label_tile.dtype,
            )

            if len(present_positive) == 0:
                masks[ch_idx] = np.zeros(label_tile.shape, dtype=np.bool_)
            else:
                # Vectorised: mask = label_tile in present_positive
                masks[ch_idx] = np.isin(label_tile, present_positive)

        # Transfer to GPU if requested
        if device == "gpu" and cp is not None:
            return {ch: cp.asarray(m) for ch, m in masks.items()}
        return masks

    def get_tile_masks_with_intensity(
        self,
        tile_slice_y: slice,
        tile_slice_x: slice,
        intensity_volume: "np.ndarray | None" = None,
    ) -> Tuple[Dict[int, "cp.ndarray"], Dict[int, float]]:
        """
        Get masks and compute sum-of-intensity for each channel in the tile.

        If ``intensity_volume`` is not provided, sum_intensity defaults to 0
        for all channels (since MCMICRO mode may not always have raw intensities
        available per tile).

        Returns
        -------
        masks : Dict[int, cp.ndarray]
        sum_intensity : Dict[int, float]
        """
        masks = self.get_tile_masks(tile_slice_y, tile_slice_x, device="gpu")

        sum_intensity: Dict[int, float] = {}
        for ch_idx in masks:
            if intensity_volume is not None and cp is not None:
                m = masks[ch_idx]
                if cp.any(m):
                    # intensity_volume expected shape: (Z, Y, X)
                    vol_gpu = cp.asarray(intensity_volume)
                    sum_intensity[ch_idx] = float(cp.sum(vol_gpu[m]).get())
                else:
                    sum_intensity[ch_idx] = 0.0
            else:
                sum_intensity[ch_idx] = 0.0

        return masks, sum_intensity

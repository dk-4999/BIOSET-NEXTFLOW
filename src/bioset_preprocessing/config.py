from dataclasses import dataclass, field
from typing import Sequence, Tuple, Dict, Optional, Literal

@dataclass(frozen=True)
class VoxelSizeUM:
    x: float
    y: float
    z: float
    @property
    def sampling_zyx(self) -> Tuple[float, float, float]:
        return (self.z, self.y, self.x)
    @property
    def voxel_volume_um3(self) -> float:
        return self.x * self.y * self.z

@dataclass
class PipelineConfig:
    # ---------- Input mode ----------
    # "legacy"  = raw zarr + BioSET thresholding (current behaviour)
    # "mcmicro" = MCMICRO segmentation masks + quantification tables
    input_mode: Literal["legacy", "mcmicro"] = "legacy"

    # ---------- Data sources (legacy mode) ----------
    zarr_url: str | None = None     # remote
    zarr_path: str | None = None    # local
    metadata_url: str = ""
    channels: Sequence[int] = ()

    # ---------- Data sources (mcmicro mode) ----------
    # Path to zarr converted from MCMICRO OME-TIFF output
    mcmicro_zarr_path: str | None = None
    # Path to MCMICRO segmentation mask (label image, zarr or TIFF)
    mcmicro_segmentation_path: str | None = None
    # Path to MCMICRO quantification CSV
    mcmicro_quantification_path: str | None = None
    # Path to marker_thresholds.yml (marker name -> positivity threshold)
    marker_thresholds_path: str | None = None
    # Channel names (required in mcmicro mode; in legacy mode, inferred or passed to run_aggregation)
    channel_names: Sequence[str] = ()

    # ---------- Zarr conversion ----------
    # Source for TIFF-to-zarr conversion (OME-TIFF path)
    zarr_conversion_source: str | None = None
    # Output path for converted zarr
    zarr_conversion_target: str | None = None

    # ---------- Tiling ----------
    tile_xy: Tuple[int, int] = (128, 128)   # (tile_y, tile_x)
    channel_batch: int = 8                  # how many channels per batch

    # ---------- Thresholding (legacy mode only) ----------
    alpha: float = 0.4
    trim_q: float = 0.98

    # ---------- Segmentation ----------
    voxel_size_um: VoxelSizeUM = VoxelSizeUM(0.14, 0.14, 0.28)
    min_obj_vol_um3: float = 1.0
    connectivity: int = 26

    # ---------- Dilation ----------
    dilate_um: Sequence[float] = (0.0, 1.0, 2.0, 3.0)

    # ---------- Advanced ----------
    prefer_lowest_res_for_global: bool = True
    float64_distances: bool = False      # for edt

    # ---------- Overlaps ----------
    max_set_size: int = 4
    min_marker_vox: Dict[float, int] | int = 1000
    min_support_pair: Dict[float, int] | int = 100
    min_support_set: Dict[float, int] | int = 50
    aggressive_stop_on_fail: bool = True

    # ---------- Hierarchical aggregation ----------
    hierarchy_levels: int = 4

    # ---------- Output ----------
    output_dir: str = "results"
    output_name: str = "analysis"

    # ---------- Checkpointing ----------
    checkpoint_dir: str = "checkpoints"

    def validate(self) -> None:
        """Validate configuration consistency."""
        if self.input_mode == "legacy":
            if not self.zarr_url and not self.zarr_path:
                raise ValueError(
                    "Legacy mode requires at least one of zarr_path or zarr_url."
                )
        elif self.input_mode == "mcmicro":
            if not self.mcmicro_segmentation_path:
                raise ValueError(
                    "MCMICRO mode requires mcmicro_segmentation_path."
                )
            if not self.mcmicro_quantification_path:
                raise ValueError(
                    "MCMICRO mode requires mcmicro_quantification_path."
                )
            if not self.marker_thresholds_path:
                raise ValueError(
                    "MCMICRO mode requires marker_thresholds_path."
                )
            # In mcmicro mode we need a zarr for tile reading (the converted registered image)
            if not self.mcmicro_zarr_path and not self.zarr_path:
                raise ValueError(
                    "MCMICRO mode requires mcmicro_zarr_path or zarr_path "
                    "pointing to the converted registered image."
                )
        else:
            raise ValueError(f"Unknown input_mode: {self.input_mode!r}")

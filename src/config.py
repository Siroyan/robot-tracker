import argparse
import json
from dataclasses import asdict, dataclass
from typing import List, Optional, Tuple

from tracker_types import Point


@dataclass
class TrackerConfig:
    hsv_lower: Tuple[int, int, int] = (0, 40, 20)
    hsv_upper: Tuple[int, int, int] = (40, 255, 255)
    min_area_px: float = 5.0
    max_area_px: float = 30000.0
    cluster_radius_px: float = 80.0
    max_jump_px: float = 180.0
    smoothing_alpha: float = 0.35
    num_thrusters: int = 4
    min_thruster_distance_px: float = 50.0
    max_thruster_distance_px: float = 190.0
    thruster_search_radius_px: float = 18.0
    thruster_reacquire_radius_px: float = 30.0
    thruster_max_step_px: float = 8.0
    orange_clahe_clip_limit: float = 2.0
    orange_red_minus_green_min: int = 8
    orange_green_minus_blue_min: int = -5
    orange_min_red: int = 35
    orange_min_green: int = 20
    pool_corners_px: Optional[List[Point]] = None
    pool_width_m: Optional[float] = None
    pool_height_m: Optional[float] = None
    init_point_px: Optional[Point] = None
    reference_frame: int = 0


def target_thruster_count(cfg: TrackerConfig) -> int:
    """Return the configured thruster count clamped to at least one."""
    return max(1, int(cfg.num_thrusters))


def load_config(path: Optional[str]) -> TrackerConfig:
    """Load tracker settings from JSON and normalize tuple/list fields."""
    cfg = TrackerConfig()
    if path is None:
        return cfg
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    for key, value in data.items():
        if hasattr(cfg, key):
            setattr(cfg, key, value)
    cfg.hsv_lower = tuple(cfg.hsv_lower)  # type: ignore[arg-type]
    cfg.hsv_upper = tuple(cfg.hsv_upper)  # type: ignore[arg-type]
    if cfg.pool_corners_px is not None:
        cfg.pool_corners_px = [(float(x), float(y)) for x, y in cfg.pool_corners_px]
    if cfg.init_point_px is not None:
        cfg.init_point_px = (float(cfg.init_point_px[0]), float(cfg.init_point_px[1]))
    return cfg


def save_config(cfg: TrackerConfig, path: str) -> None:
    """Serialize the current tracker settings to a JSON file."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(asdict(cfg), f, ensure_ascii=False, indent=2)


def parse_point_text(text: str) -> Point:
    """Parse a CLI point argument in `x,y` format."""
    parts = text.split(",")
    if len(parts) != 2:
        raise argparse.ArgumentTypeError(f"Point must be x,y format: {text}")
    try:
        return (float(parts[0]), float(parts[1]))
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"Point must be numeric x,y format: {text}") from exc

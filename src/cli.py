import argparse

from pipeline import make_config_headless, save_orange_preview, save_reference_frame, track_video
from config import TrackerConfig, load_config, parse_point_text


def parse_args() -> argparse.Namespace:
    """Define and parse the command-line interface."""
    parser = argparse.ArgumentParser(description="Headless tracker for an underwater robot from orange thrusters and pool homography.")
    parser.add_argument("video", help="Input video path")
    parser.add_argument("--config", help="JSON config path")
    parser.add_argument("--make-config", help="Create JSON config and exit. This headless version does not open GUI windows.")
    parser.add_argument("--reference-frame", type=int, default=0, help="Frame index used for reference/config creation")
    parser.add_argument("--export-reference-frame", help="Save the reference frame as an image and exit unless --make-config is also specified")
    parser.add_argument("--export-orange-preview", help="Save a reference image with detected orange candidates and exit unless --make-config is also specified")
    parser.add_argument("--pool-corners-px", nargs=4, type=parse_point_text, metavar=("TL", "TR", "BR", "BL"), help="Pool corners in pixels: x,y x,y x,y x,y")
    parser.add_argument("--init-point-px", type=parse_point_text, help="Initial robot/thruster point in pixels: x,y")
    parser.add_argument("--pool-width-m", type=float, help="Real pool width in meters, mapped from top-left to top-right")
    parser.add_argument("--pool-height-m", type=float, help="Real pool height/length in meters, mapped from top-left to bottom-left")
    parser.add_argument("--csv", help="Output CSV path")
    parser.add_argument("--annotated", help="Optional annotated MP4 output path")
    parser.add_argument("--hsv-lower", nargs=3, type=int, metavar=("H", "S", "V"), help="Override HSV lower threshold")
    parser.add_argument("--hsv-upper", nargs=3, type=int, metavar=("H", "S", "V"), help="Override HSV upper threshold")
    return parser.parse_args()


def build_runtime_config(args: argparse.Namespace) -> TrackerConfig:
    """Merge config-file values with CLI overrides into one runtime config."""
    cfg = load_config(args.config)
    if args.pool_width_m is not None:
        cfg.pool_width_m = args.pool_width_m
    if args.pool_height_m is not None:
        cfg.pool_height_m = args.pool_height_m
    if args.hsv_lower is not None:
        cfg.hsv_lower = tuple(args.hsv_lower)
    if args.hsv_upper is not None:
        cfg.hsv_upper = tuple(args.hsv_upper)
    if args.pool_corners_px is not None:
        cfg.pool_corners_px = list(args.pool_corners_px)
    if args.init_point_px is not None:
        cfg.init_point_px = args.init_point_px
    return cfg


def main() -> None:
    """CLI entrypoint that routes export, config generation, and tracking modes."""
    args = parse_args()
    cfg = build_runtime_config(args)

    did_export = False
    if args.export_reference_frame:
        save_reference_frame(args.video, args.reference_frame, args.export_reference_frame)
        did_export = True
    if args.export_orange_preview:
        save_orange_preview(args.video, args.reference_frame, args.export_orange_preview, cfg)
        did_export = True
    if args.make_config:
        make_config_headless(
            cfg,
            args.make_config,
            reference_frame=args.reference_frame,
            pool_width_m=args.pool_width_m,
            pool_height_m=args.pool_height_m,
            pool_corners_px=args.pool_corners_px,
            init_point_px=args.init_point_px,
        )
        return
    if did_export and not args.annotated and args.csv is None:
        return
    if args.csv is None:
        args.csv = "positions.csv"

    rows = track_video(
        args.video,
        cfg,
        csv_path=args.csv,
        annotated_path=args.annotated,
    )
    detected_rows = [row for row in rows if bool(row["detected"])]
    detected_ratio = len(detected_rows) / len(rows) if rows else 0.0
    print(f"Frames: {len(rows)}, detected: {detected_ratio * 100.0:.1f}%")
    xs = [row["pool_x_m"] for row in detected_rows if row["pool_x_m"] == row["pool_x_m"]]
    ys = [row["pool_y_m"] for row in detected_rows if row["pool_y_m"] == row["pool_y_m"]]
    if xs and ys:
        print(f"Metric coordinate range: x={min(xs):.3f}..{max(xs):.3f} m, y={min(ys):.3f}..{max(ys):.3f} m")


if __name__ == "__main__":
    main()

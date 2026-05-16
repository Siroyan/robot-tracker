import math
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np

from config import TrackerConfig, save_config, target_thruster_count
from detection import detect_orange_contours
from geometry import build_homography, build_pool_mask, distance_stats, transform_point
from output import compute_speeds, draw_annotation, write_csv
from tracking import choose_robot_position, predict_thruster_centers, select_initial_thruster_points
from tracker_types import Point, ThrusterPoints, ThrusterVelocities


def detection_area_corners(cfg: TrackerConfig) -> Optional[List[Point]]:
    """Return the polygon used to limit orange detection."""
    # 水面領域が指定されていれば検出範囲に使い、未指定なら従来どおりプール四隅を使う。
    return cfg.water_area_corners_px if cfg.water_area_corners_px is not None else cfg.pool_corners_px


def read_frame(video_path: str, frame_index: int) -> np.ndarray:
    """Load a single frame by index from a video file."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"Cannot read frame {frame_index} from {video_path}")
    return frame


def save_reference_frame(video_path: str, frame_index: int, output_path: str) -> None:
    """Export one frame as an image for manual calibration work."""
    frame = read_frame(video_path, frame_index)
    ok = cv2.imwrite(output_path, frame)
    if not ok:
        raise RuntimeError(f"Cannot write reference frame: {output_path}")
    print(f"Saved reference frame: {output_path}")


def save_orange_preview(video_path: str, frame_index: int, output_path: str, cfg: TrackerConfig) -> None:
    """Export a preview image that overlays orange detections and initial thruster picks."""
    frame = read_frame(video_path, frame_index)
    pool_mask = build_pool_mask(frame.shape, detection_area_corners(cfg))
    detections, mask, _relaxed_mask = detect_orange_contours(frame, cfg, pool_mask)
    max_markers = target_thruster_count(cfg)
    # プレビューでも本番と同じ初期フレーム選択ロジックを使い、
    # 動画処理前にマーカー数や順序の問題を確認できるようにする。
    selected_points = select_initial_thruster_points(mask, detections, cfg)
    out = frame.copy()
    for detection in detections[: max(max_markers + 4, 8)]:
        x, y, w, h = int(detection["x"]), int(detection["y"]), int(detection["w"]), int(detection["h"])
        cx, cy = int(detection["cx"]), int(detection["cy"])
        cv2.rectangle(out, (x, y), (x + w, y + h), (0, 128, 255), 1)
        cv2.circle(out, (cx, cy), 4, (0, 165, 255), -1)

    for idx, point in enumerate(selected_points[:max_markers], start=1):
        cx, cy = int(round(point[0])), int(round(point[1]))
        cv2.circle(out, (cx, cy), 8, (0, 255, 0), 2)
        cv2.putText(out, str(idx), (cx + 8, cy - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(out, str(idx), (cx + 8, cy - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)

    ok = cv2.imwrite(output_path, out)
    if not ok:
        raise RuntimeError(f"Cannot write orange preview: {output_path}")
    print(f"Saved orange preview: {output_path}")
    print(f"Orange candidates on reference frame (showing up to {max_markers}):")
    for idx, detection in enumerate(detections[:max_markers], start=1):
        print(
            f"  #{idx}: cx={detection['cx']:.1f}, cy={detection['cy']:.1f}, area={detection['area']:.1f}, "
            f"bbox=({detection['x']:.0f},{detection['y']:.0f},{detection['w']:.0f},{detection['h']:.0f})"
        )
    if selected_points:
        print("Selected initial thruster points:")
        for idx, point in enumerate(selected_points[:max_markers], start=1):
            print(f"  T{idx}: x={point[0]:.1f}, y={point[1]:.1f}")


def make_config_headless(
    cfg: TrackerConfig,
    output_path: str,
    *,
    reference_frame: int,
    pool_width_m: Optional[float],
    pool_height_m: Optional[float],
    pool_corners_px: Optional[List[Point]],
    water_area_corners_px: Optional[List[Point]],
    init_point_px: Optional[Point],
) -> None:
    """Build and save a config file from explicit headless inputs."""
    # CLI解析はpipeline外に置き、この関数では検証済み入力を保存用設定へ写すだけにする。
    cfg.reference_frame = reference_frame
    cfg.pool_width_m = pool_width_m
    cfg.pool_height_m = pool_height_m

    if pool_corners_px is not None:
        if len(pool_corners_px) != 4:
            raise RuntimeError("--pool-corners-px requires exactly 4 points: TL TR BR BL")
        cfg.pool_corners_px = list(pool_corners_px)
    elif cfg.pool_width_m is not None or cfg.pool_height_m is not None:
        print("Warning: pool size is set but --pool-corners-px was not provided. Metric conversion will be disabled.")

    if water_area_corners_px is not None:
        if len(water_area_corners_px) != 4:
            raise RuntimeError("--water-area-corners-px requires exactly 4 points: TL TR BR BL")
        cfg.water_area_corners_px = list(water_area_corners_px)

    if init_point_px is not None:
        cfg.init_point_px = init_point_px
    elif cfg.init_point_px is None:
        print("Warning: --init-point-px was not provided. The tracker will start from the largest orange object, which may be wrong.")

    save_config(cfg, output_path)
    print(f"Saved config: {output_path}")


def track_video(
    video_path: str,
    cfg: TrackerConfig,
    *,
    csv_path: Optional[str],
    annotated_path: Optional[str],
) -> List[Dict[str, Any]]:
    """Run the end-to-end tracking pipeline for one video."""
    def point_velocities() -> Optional[ThrusterVelocities]:
        # スラスタ別速度は、安定した点数で2フレーム以上追跡できた後だけ使える。
        # それまでは重心速度だけを使う。
        if prev_thruster_points is None or prev_prev_thruster_points is None:
            return None
        if len(prev_thruster_points) != len(prev_prev_thruster_points) or len(prev_thruster_points) != n_thrusters:
            return None
        return [
            (
                prev_thruster_points[i][0] - prev_prev_thruster_points[i][0],
                prev_thruster_points[i][1] - prev_prev_thruster_points[i][1],
            )
            for i in range(n_thrusters)
        ]

    def process_frame(
        frame_idx: int,
        detections: List[Dict[str, float]],
        mask: np.ndarray,
        relaxed_mask: np.ndarray,
    ) -> tuple[ThrusterPoints, Optional[Point], float, int, bool, str, Optional[ThrusterPoints]]:
        # 0フレーム目だけは特別扱いし、このフレームだけを根拠に全スラスタIDを初期化する。
        # 以降のフレームでは、その固定IDを追跡する。
        if frame_idx == 0 and len(initial_thruster_points) == n_thrusters:
            thruster_points = initial_thruster_points
            raw_pos = tuple(sum(point[i] for point in thruster_points) / n_thrusters for i in (0, 1))
            return thruster_points, raw_pos, float(sum(max(d["area"], 1.0) for d in detections)), n_thrusters, True, "init", None

        velocities = point_velocities()
        roi_centers = None
        if prev_thruster_points is not None and len(prev_thruster_points) == n_thrusters:
            # ROI中心は注釈描画にも渡し、古い予測や誤った予測の周辺を探していないか
            # 動画上で確認できるようにする。
            roi_centers = predict_thruster_centers(cfg, prev_thruster_points, velocities, velocity)
        thruster_points, raw_pos, area, n_cluster, reliable, tracking_mode = choose_robot_position(
            detections,
            mask,
            relaxed_mask,
            cfg,
            prev_pos,
            velocity,
            prev_thruster_points,
            velocities,
        )
        return thruster_points, raw_pos, area, n_cluster, reliable, tracking_mode, roi_centers

    def build_row(
        frame_idx: int,
        time_s: float,
        detected: bool,
        pos_for_output: Optional[Point],
        pool_xy: tuple[float, float],
        area: float,
        n_cluster: int,
        detections: List[Dict[str, float]],
        thruster_points: ThrusterPoints,
    ) -> Dict[str, Any]:
        # CSV列は設定されたスラスタ数で固定する。欠損点はNaNにして、
        # 後段の解析で除外しやすくする。
        thruster_min_distance, thruster_max_distance = math.nan, math.nan
        thruster_xy = [(math.nan, math.nan)] * n_thrusters
        if len(thruster_points) == n_thrusters:
            thruster_xy = [(point[0], point[1]) for point in thruster_points]
            thruster_min_distance, thruster_max_distance = distance_stats(thruster_points)
        row = {
            "frame": frame_idx,
            "time_s": time_s,
            "detected": bool(detected),
            "px_x": pos_for_output[0] if pos_for_output is not None else math.nan,
            "px_y": pos_for_output[1] if pos_for_output is not None else math.nan,
            "pool_x_m": pool_xy[0],
            "pool_y_m": pool_xy[1],
            "orange_area_px2": area if detected else 0.0,
            "cluster_contours": n_cluster if detected else 0,
            "num_orange_candidates": len(detections),
            "thruster_min_distance_px": thruster_min_distance,
            "thruster_max_distance_px": thruster_max_distance,
        }
        for idx in range(n_thrusters):
            row[f"thruster_{idx + 1}_x"] = thruster_xy[idx][0]
            row[f"thruster_{idx + 1}_y"] = thruster_xy[idx][1]
        return row

    n_thrusters = target_thruster_count(cfg)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    first_ok, first_frame = cap.read()
    if not first_ok:
        raise RuntimeError("Cannot read first frame")
    pool_mask = build_pool_mask(first_frame.shape, detection_area_corners(cfg))
    homography = build_homography(cfg)
    initial_detections, initial_mask, initial_relaxed_mask = detect_orange_contours(first_frame, cfg, pool_mask)
    initial_thruster_points = select_initial_thruster_points(initial_mask, initial_detections, cfg)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 1)
    writer = None
    if annotated_path:
        annotated_output_path = Path(annotated_path)
        if annotated_output_path.parent != Path("."):
            annotated_output_path.parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(annotated_output_path), fourcc, fps if fps > 0 else 30.0, (width, height))
        if not writer.isOpened():
            raise RuntimeError(f"Cannot write annotated video: {annotated_path}")

    rows: List[Dict[str, Any]] = []
    prev_pos: Optional[Point] = None
    prev_thruster_points: Optional[ThrusterPoints] = initial_thruster_points if len(initial_thruster_points) == n_thrusters else None
    prev_prev_thruster_points: Optional[ThrusterPoints] = None
    smoothed_pos: Optional[Point] = None
    velocity: Point = (0.0, 0.0)
    trail: List[Point] = []

    frame_idx = 0
    # 初期化のために0フレーム目は読み込み済み。ここでキューに戻すことで、
    # 出力ループでは後続フレームと同じ経路で0フレーム目を書き出せる。
    pending_first_frame: Optional[np.ndarray] = first_frame
    pending_first_detections = initial_detections
    pending_first_mask = initial_mask
    pending_first_relaxed_mask = initial_relaxed_mask
    while True:
        if pending_first_frame is not None:
            frame = pending_first_frame
            detections = pending_first_detections
            mask = pending_first_mask
            relaxed_mask = pending_first_relaxed_mask
            pending_first_frame = None
        else:
            ok, frame = cap.read()
            if not ok:
                break
            detections, mask, relaxed_mask = detect_orange_contours(frame, cfg, pool_mask)

        time_s = frame_idx / fps if fps > 0 else float(frame_idx)
        thruster_points, raw_pos, area, n_cluster, reliable, tracking_mode, roi_centers = process_frame(
            frame_idx, detections, mask, relaxed_mask
        )

        detected = raw_pos is not None and reliable
        if detected and raw_pos is not None:
            # 平滑化するのはロボット重心だけにする。スラスタ点は平滑化せず、
            # ROI予測には直近で測定したマーカー位置を使う。
            if smoothed_pos is None:
                smoothed_pos = raw_pos
            else:
                alpha = float(cfg.smoothing_alpha)
                smoothed_pos = (
                    (1.0 - alpha) * smoothed_pos[0] + alpha * raw_pos[0],
                    (1.0 - alpha) * smoothed_pos[1] + alpha * raw_pos[1],
                )
            if prev_pos is not None:
                velocity = (smoothed_pos[0] - prev_pos[0], smoothed_pos[1] - prev_pos[1])
            # 速度計算後に履歴を更新し、次フレームで重心移動とスラスタ別移動の両方を使えるようにする。
            prev_pos = smoothed_pos
            prev_prev_thruster_points = prev_thruster_points
            prev_thruster_points = thruster_points
            pos_for_output: Optional[Point] = smoothed_pos
            trail.append(smoothed_pos)
        else:
            pos_for_output = None

        pool_xy = transform_point(homography, pos_for_output)
        rows.append(build_row(frame_idx, time_s, detected, pos_for_output, pool_xy, area, n_cluster, detections, thruster_points))

        if writer is not None:
            annotated = draw_annotation(
                frame,
                detections,
                thruster_points,
                roi_centers,
                tracking_mode,
                pos_for_output,
                pool_xy,
                trail,
                cfg,
                frame_idx,
                time_s,
                detected,
            )
            writer.write(annotated)

        frame_idx += 1

    cap.release()
    if writer is not None:
        writer.release()
        print(f"Saved annotated video: {annotated_path}")

    if homography is not None:
        rows = compute_speeds(rows)
    else:
        for row in rows:
            row["speed_mps"] = math.nan

    if csv_path:
        write_csv(rows, csv_path, n_thrusters)
        print(f"Saved CSV: {csv_path}")
    return rows

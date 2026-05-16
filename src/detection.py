from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from config import TrackerConfig


Detection = Dict[str, float]


def enhance_frame_for_orange(frame: np.ndarray, cfg: TrackerConfig) -> np.ndarray:
    """Apply contrast enhancement before orange color extraction."""
    # CLAHEで局所コントラストを上げ、薄く写ったオレンジもHSV抽出に残りやすくする。
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=float(cfg.orange_clahe_clip_limit), tileGridSize=(8, 8))
    l_channel = clahe.apply(l_channel)
    enhanced = cv2.merge([l_channel, a_channel, b_channel])
    enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
    return cv2.GaussianBlur(enhanced, (3, 3), 0)


def build_color_mask(
    frame: np.ndarray,
    cfg: TrackerConfig,
    lower_hsv: Tuple[int, int, int],
    upper_hsv: Tuple[int, int, int],
    pool_mask: Optional[np.ndarray],
    *,
    open_kernel_size: int,
    open_iterations: int,
    close_kernel_size: int,
    close_iterations: int,
) -> np.ndarray:
    """Build a binary orange mask from HSV and RGB-dominance rules."""
    enhanced = enhance_frame_for_orange(frame, cfg)
    hsv = cv2.cvtColor(enhanced, cv2.COLOR_BGR2HSV)
    lower = np.array(lower_hsv, dtype=np.uint8)
    upper = np.array(upper_hsv, dtype=np.uint8)
    mask_hsv = cv2.inRange(hsv, lower, upper)
    bgr_b, bgr_g, bgr_r = cv2.split(enhanced)
    # HSVだけではプールの反射も拾うため、オレンジ色の素材に期待される
    # RGBチャンネルの大小関係も条件に含める。
    orange_dominance = (
        (bgr_r.astype(np.int16) >= bgr_g.astype(np.int16) + int(cfg.orange_red_minus_green_min))
        & (bgr_g.astype(np.int16) >= bgr_b.astype(np.int16) + int(cfg.orange_green_minus_blue_min))
        & (bgr_r >= int(cfg.orange_min_red))
        & (bgr_g >= int(cfg.orange_min_green))
    )
    mask = cv2.bitwise_and(mask_hsv, orange_dominance.astype(np.uint8) * 255)
    if pool_mask is not None:
        mask = cv2.bitwise_and(mask, pool_mask)
    # オープニングで孤立ノイズを消し、クロージングでスラスタ領域の小さな切れ目をつなぐ。
    if open_iterations > 0:
        open_kernel = np.ones((open_kernel_size, open_kernel_size), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, open_kernel, iterations=open_iterations)
    if close_iterations > 0:
        close_kernel = np.ones((close_kernel_size, close_kernel_size), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close_kernel, iterations=close_iterations)
    return cv2.medianBlur(mask, 3)


def extract_detections_from_mask(mask: np.ndarray, cfg: TrackerConfig) -> List[Detection]:
    """Convert a binary mask into sorted contour detections."""
    contours, _hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    detections: List[Detection] = []
    for contour in contours:
        # 面積で、小さすぎるノイズと大きすぎる非スラスタ領域を除外する。
        area = float(cv2.contourArea(contour))
        if area < cfg.min_area_px or area > cfg.max_area_px:
            continue
        x, y, w, h = cv2.boundingRect(contour)
        moments = cv2.moments(contour)
        if abs(moments["m00"]) > 1e-9:
            cx = float(moments["m10"] / moments["m00"])
            cy = float(moments["m01"] / moments["m00"])
        else:
            cx = x + w / 2.0
            cy = y + h / 2.0
        detections.append(
            {
                "cx": cx,
                "cy": cy,
                "area": area,
                "x": float(x),
                "y": float(y),
                "w": float(w),
                "h": float(h),
            }
        )
    detections.sort(key=lambda item: item["area"], reverse=True)
    return detections


def detect_orange_contours(
    frame: np.ndarray,
    cfg: TrackerConfig,
    pool_mask: Optional[np.ndarray],
) -> Tuple[List[Detection], np.ndarray, np.ndarray]:
    """Return detections plus strict/relaxed masks for global and ROI tracking."""
    # 誤検知が少ないため、まず厳しめのマスクを使う。
    mask = build_color_mask(
        frame,
        cfg,
        cfg.hsv_lower,
        cfg.hsv_upper,
        pool_mask,
        open_kernel_size=3,
        open_iterations=0,
        close_kernel_size=3,
        close_iterations=1,
    )
    detections = extract_detections_from_mask(mask, cfg)

    # 暗いフレームやブレたフレームで厳しめの条件が実スラスタを逃した場合に備え、
    # ROI追跡用のフォールバックとして緩めのマスクも保持する。
    relaxed_lower = (
        max(0, int(cfg.hsv_lower[0]) - 10),
        max(0, int(cfg.hsv_lower[1]) - 40),
        max(0, int(cfg.hsv_lower[2]) - 30),
    )
    relaxed_upper = (
        min(179, int(cfg.hsv_upper[0]) + 20),
        min(255, int(cfg.hsv_upper[1])),
        min(255, int(cfg.hsv_upper[2])),
    )
    relaxed_mask = build_color_mask(
        frame,
        cfg,
        relaxed_lower,
        relaxed_upper,
        pool_mask,
        open_kernel_size=3,
        open_iterations=0,
        close_kernel_size=3,
        close_iterations=0,
    )
    relaxed_detections = extract_detections_from_mask(relaxed_mask, cfg)
    # 厳しめの検出で候補がゼロの場合は、即座に空フレーム扱いせず緩めのマスクから開始する。
    if detections:
        return detections, mask, relaxed_mask
    return relaxed_detections, relaxed_mask, relaxed_mask

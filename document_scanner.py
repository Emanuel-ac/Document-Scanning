import argparse
import csv
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

try:
    from pillow_heif import register_heif_opener
    from PIL import Image as PILImage

    register_heif_opener()
    HEIC_SUPPORT = True
except ImportError:
    HEIC_SUPPORT = False

try:
    from skimage.filters import threshold_sauvola

    SKIMAGE_SUPPORT = True
except ImportError:
    SKIMAGE_SUPPORT = False


SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp", ".heic", ".heif"}
DETECTION_METHODS = ("contour", "threshold", "hough", "corners")
BINARIZATION_METHODS = (
    "adaptive_gaussian",
    "otsu_deshadow",
    "sauvola_deshadow",
    "hybrid_deshadow",
)


@dataclass
class QualityMetrics:
    sharpness: float
    sharpness_label: str
    skew_angle: float
    skew_label: str
    contrast: float
    brightness: float
    shadow_level: float
    binarization_score: float
    detection_confidence: float
    overall_score: float
    overall_label: str


@dataclass
class SceneAnalysis:
    background_complexity: str
    lighting_condition: str
    lighting_variation: float
    background_edge_density: float
    document_area_ratio: float


@dataclass
class ScanResult:
    success: bool
    method: str
    binarization_method: str
    corners: Optional[list]
    rectified: Optional[np.ndarray]
    binary: Optional[np.ndarray]
    metrics: Optional[QualityMetrics]
    scene: Optional[SceneAnalysis]
    binarization_scores: dict[str, float]
    message: str
    accepted: bool = False
    rejection_reason: str = ""
    quality_flags: list[str] = field(default_factory=list)
    requested_detection_mode: str = "auto"


@dataclass
class DetectionCandidate:
    method: str
    corners: np.ndarray
    score: float


@dataclass
class BinarizationCandidate:
    method: str
    image: np.ndarray
    score: float


def read_image(image_path: str) -> Optional[np.ndarray]:
    path = Path(image_path)
    if path.suffix.lower() in {".heic", ".heif"}:
        if not HEIC_SUPPORT:
            print("  [ERROR] HEIC nesuportat. Ruleaza: pip install pillow-heif")
            return None
        pil_img = PILImage.open(path).convert("RGB")
        return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    img = cv2.imread(str(path))
    if img is None:
        print(f"  [ERROR] Nu pot citi imaginea: {path}")
    return img


def odd_size(value: float, minimum: int = 3, maximum: Optional[int] = None) -> int:
    size = int(max(minimum, round(value)))
    if maximum is not None:
        size = min(size, maximum)
    if size % 2 == 0:
        size += 1
    if maximum is not None and size > maximum:
        size = maximum - 1 if maximum % 2 == 0 else maximum
    return max(minimum, size)


def preprocess(img: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    denoised = cv2.bilateralFilter(gray, 9, 75, 75)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(denoised)


def enhance_grayscale(gray: np.ndarray, clip_limit: float = 3.0, tile_size: int = 16) -> np.ndarray:
    tile = max(4, min(32, int(tile_size)))
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile, tile))
    return clahe.apply(gray)


def auto_canny(gray: np.ndarray, sigma: float = 0.33) -> np.ndarray:
    v = np.median(gray)
    lo = int(max(0, (1.0 - sigma) * v))
    hi = int(min(255, (1.0 + sigma) * v))
    return cv2.Canny(gray, lo, hi)


def order_corners(pts: np.ndarray) -> np.ndarray:
    pts = pts.reshape(4, 2).astype(np.float32)
    rect = np.zeros((4, 2), dtype=np.float32)
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


def polygon_area(corners: np.ndarray) -> float:
    return float(cv2.contourArea(corners.astype(np.float32).reshape(-1, 1, 2)))


def angle_at_pt(p1: np.ndarray, vertex: np.ndarray, p2: np.ndarray) -> float:
    a = np.array(p1, dtype=np.float32) - np.array(vertex, dtype=np.float32)
    b = np.array(p2, dtype=np.float32) - np.array(vertex, dtype=np.float32)
    cos = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8)
    return float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))))


def validate_corners(img: np.ndarray, corners: np.ndarray) -> bool:
    if corners is None or len(corners) != 4:
        return False

    h, w = img.shape[:2]
    pts = order_corners(np.array(corners, dtype=np.float32))

    if not cv2.isContourConvex(pts.reshape(-1, 1, 2)):
        return False

    area_ratio = polygon_area(pts) / max(h * w, 1)
    if not (0.08 <= area_ratio <= 0.98):
        return False

    edge_lengths = [
        np.linalg.norm(pts[1] - pts[0]),
        np.linalg.norm(pts[2] - pts[1]),
        np.linalg.norm(pts[3] - pts[2]),
        np.linalg.norm(pts[0] - pts[3]),
    ]
    if min(edge_lengths) < 0.12 * min(h, w):
        return False

    angles = [
        angle_at_pt(pts[3], pts[0], pts[1]),
        angle_at_pt(pts[0], pts[1], pts[2]),
        angle_at_pt(pts[1], pts[2], pts[3]),
        angle_at_pt(pts[2], pts[3], pts[0]),
    ]
    if any(angle < 35 or angle > 145 for angle in angles):
        return False

    top, right, bottom, left = edge_lengths
    if min(top, bottom) / max(top, bottom, 1e-6) < 0.2:
        return False
    if min(left, right) / max(left, right, 1e-6) < 0.2:
        return False
    return True


def score_candidate(img: np.ndarray, gray: np.ndarray, corners: np.ndarray, method: str) -> float:
    h, w = img.shape[:2]
    pts = order_corners(corners)
    area_ratio = polygon_area(pts) / max(h * w, 1)

    angles = np.array(
        [
            angle_at_pt(pts[3], pts[0], pts[1]),
            angle_at_pt(pts[0], pts[1], pts[2]),
            angle_at_pt(pts[1], pts[2], pts[3]),
            angle_at_pt(pts[2], pts[3], pts[0]),
        ]
    )
    angle_score = 1.0 - min(np.mean(np.abs(angles - 90.0)) / 45.0, 1.0)

    mask = np.zeros(gray.shape, dtype=np.uint8)
    cv2.fillConvexPoly(mask, pts.astype(np.int32), 255)
    edges = auto_canny(gray)
    edge_density = float(np.mean(edges[mask > 0] > 0)) if np.any(mask > 0) else 0.0

    border_margin = 0.02 * min(h, w)
    near_border = np.mean(
        [
            pts[:, 0].min() <= border_margin,
            pts[:, 1].min() <= border_margin,
            pts[:, 0].max() >= (w - 1 - border_margin),
            pts[:, 1].max() >= (h - 1 - border_margin),
        ]
    )

    bonus = {"contour": 0.08, "threshold": 0.06, "hough": 0.05, "corners": 0.02}.get(method, 0.0)
    return float(
        0.38 * min(area_ratio / 0.6, 1.0)
        + 0.27 * angle_score
        + 0.20 * min(edge_density / 0.12, 1.0)
        + 0.10 * near_border
        + bonus
    )


def detect_contour(img: np.ndarray, gray: np.ndarray) -> Optional[np.ndarray]:
    h, w = img.shape[:2]
    min_area = (h * w) * 0.05

    edges = auto_canny(gray)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)
    cnts, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)

    for cnt in cnts[:10]:
        if cv2.contourArea(cnt) < min_area:
            break
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
        if len(approx) == 4:
            return order_corners(approx)
    return None


def line_angle(line) -> float:
    x1, y1, x2, y2 = line[0]
    return float(np.degrees(np.arctan2(y2 - y1, x2 - x1)))


def cluster_lines(lines, angle_thresh: float = 20):
    horiz, vert = [], []
    for line in lines:
        ang = abs(line_angle(line)) % 180
        if ang < angle_thresh or ang > (180 - angle_thresh):
            horiz.append(line)
        elif 90 - angle_thresh < ang < 90 + angle_thresh:
            vert.append(line)
    return horiz, vert


def line_to_abc(line):
    x1, y1, x2, y2 = line[0]
    a = y2 - y1
    b = x1 - x2
    c = x2 * y1 - x1 * y2
    return a, b, c


def intersect(l1, l2) -> Optional[tuple]:
    a1, b1, c1 = line_to_abc(l1)
    a2, b2, c2 = line_to_abc(l2)
    det = a1 * b2 - a2 * b1
    if abs(det) < 1e-6:
        return None
    x = (b1 * c2 - b2 * c1) / det
    y = (a2 * c1 - a1 * c2) / det
    return x, y


def detect_hough(img: np.ndarray, gray: np.ndarray) -> Optional[np.ndarray]:
    h, w = img.shape[:2]
    edges = auto_canny(gray)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)

    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=80,
        minLineLength=min(h, w) * 0.2,
        maxLineGap=30,
    )
    if lines is None or len(lines) < 4:
        return None

    horiz, vert = cluster_lines(lines)
    if len(horiz) < 2 or len(vert) < 2:
        horiz = sorted(lines, key=lambda line: abs(line_angle(line)) % 90)[: max(2, len(lines) // 2)]
        vert = sorted(lines, key=lambda line: -abs(abs(line_angle(line)) % 180 - 90))[: max(2, len(lines) // 2)]

    def extreme_lines(group, axis):
        coords = []
        for line in group:
            x1, y1, x2, y2 = line[0]
            coords.append((y1 + y2) / 2 if axis == 0 else (x1 + x2) / 2)
        idx_min = int(np.argmin(coords))
        idx_max = int(np.argmax(coords))
        return group[idx_min], group[idx_max]

    h_top, h_bot = extreme_lines(horiz, 0)
    v_lft, v_rgt = extreme_lines(vert, 1)

    corners = []
    for hl in [h_top, h_bot]:
        for vl in [v_lft, v_rgt]:
            pt = intersect(hl, vl)
            if pt is not None:
                corners.append(pt)

    if len(corners) < 4:
        return None

    corners = np.array(corners, dtype=np.float32)
    corners[:, 0] = np.clip(corners[:, 0], 0, w - 1)
    corners[:, 1] = np.clip(corners[:, 1], 0, h - 1)
    return order_corners(corners)


def detect_corners_harris(img: np.ndarray, gray: np.ndarray) -> Optional[np.ndarray]:
    h, w = img.shape[:2]
    corners = cv2.goodFeaturesToTrack(
        gray,
        maxCorners=200,
        qualityLevel=0.01,
        minDistance=20,
        blockSize=7,
        useHarrisDetector=True,
        k=0.04,
    )
    if corners is None or len(corners) < 4:
        return None

    corners = corners.reshape(-1, 2)
    cx, cy = w / 2, h / 2
    quadrants = {
        "TL": corners[(corners[:, 0] < cx) & (corners[:, 1] < cy)],
        "TR": corners[(corners[:, 0] >= cx) & (corners[:, 1] < cy)],
        "BR": corners[(corners[:, 0] >= cx) & (corners[:, 1] >= cy)],
        "BL": corners[(corners[:, 0] < cx) & (corners[:, 1] >= cy)],
    }
    targets = {"TL": (0, 0), "TR": (w, 0), "BR": (w, h), "BL": (0, h)}

    result = []
    for key, pts in quadrants.items():
        if len(pts) == 0:
            return None
        tx, ty = targets[key]
        dists = np.sqrt((pts[:, 0] - tx) ** 2 + (pts[:, 1] - ty) ** 2)
        result.append(pts[np.argmin(dists)])

    return order_corners(np.array(result, dtype=np.float32))


def detect_threshold_quad(img: np.ndarray, gray: np.ndarray) -> Optional[np.ndarray]:
    h, w = img.shape[:2]
    min_area = (h * w) * 0.08
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, otsu = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))

    for binary in [otsu, cv2.bitwise_not(otsu)]:
        closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)
        cnts, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
        for cnt in cnts[:10]:
            if cv2.contourArea(cnt) < min_area:
                break
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
            if len(approx) == 4:
                return order_corners(approx)
            rect = cv2.minAreaRect(cnt)
            box = cv2.boxPoints(rect)
            return order_corners(box)
    return None


def rectify(img: np.ndarray, corners: np.ndarray) -> np.ndarray:
    corners = corners.astype(np.float32)
    tl, tr, br, bl = corners
    w1 = np.linalg.norm(br - bl)
    w2 = np.linalg.norm(tr - tl)
    h1 = np.linalg.norm(tr - br)
    h2 = np.linalg.norm(tl - bl)

    out_w = int(max(w1, w2))
    out_h = int(max(h1, h2))
    if out_w < 80 or out_h < 80:
        raise ValueError("Degenerate quadrilateral produced an invalid warp.")

    a4_ratio = 297 / 210
    actual_ratio = out_h / max(out_w, 1)
    if 0.7 * a4_ratio < actual_ratio < 1.3 * a4_ratio:
        out_h = int(out_w * a4_ratio)

    dst = np.array(
        [[0, 0], [out_w - 1, 0], [out_w - 1, out_h - 1], [0, out_h - 1]],
        dtype=np.float32,
    )
    matrix = cv2.getPerspectiveTransform(corners, dst)
    return cv2.warpPerspective(img, matrix, (out_w, out_h), flags=cv2.INTER_LANCZOS4)


def compute_shadow_level(gray: np.ndarray) -> float:
    sigma = max(3.0, min(gray.shape[:2]) / 24.0)
    illumination = cv2.GaussianBlur(gray.astype(np.float32), (0, 0), sigmaX=sigma, sigmaY=sigma)
    illumination /= np.mean(illumination) + 1e-6
    return float(np.std(illumination) * 100.0)


def remove_shadows_divide(gray: np.ndarray) -> np.ndarray:
    sigma = max(7.0, min(gray.shape[:2]) / 18.0)
    background = cv2.GaussianBlur(gray, (0, 0), sigmaX=sigma, sigmaY=sigma)
    corrected = cv2.divide(gray, background, scale=255)
    return cv2.normalize(corrected, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)


def remove_shadows_subtract(gray: np.ndarray) -> np.ndarray:
    kernel_size = odd_size(min(gray.shape[:2]) / 10.0, minimum=21, maximum=101)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    background = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
    corrected = 255 - cv2.subtract(background, gray)
    return cv2.normalize(corrected, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)


def cleanup_binary(binary: np.ndarray) -> np.ndarray:
    binary = np.where(binary > 127, 255, 0).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    return binary


def score_binarization(gray: np.ndarray, binary: np.ndarray) -> float:
    binary = np.where(binary > 127, 255, 0).astype(np.uint8)
    ink_mask = binary == 0
    ink_ratio = float(np.mean(ink_mask))

    ideal_ratio = 0.14
    ink_score = max(0.0, 1.0 - abs(ink_ratio - ideal_ratio) / max(ideal_ratio, 1e-6))

    edges = auto_canny(gray)
    if np.any(edges > 0):
        edge_alignment = float(np.mean(ink_mask[edges > 0]))
    else:
        edge_alignment = 0.5

    num_labels, _, stats, _ = cv2.connectedComponentsWithStats(ink_mask.astype(np.uint8), connectivity=8)
    areas = stats[1:, cv2.CC_STAT_AREA] if num_labels > 1 else np.array([], dtype=np.int32)
    max_component = max(20, int(gray.size * 0.0025))
    text_components = int(np.sum((areas >= 5) & (areas <= max_component)))
    expected_components = max(gray.size / 1400.0, 1.0)
    component_ratio = text_components / expected_components
    if component_ratio <= 1.0:
        component_score = component_ratio
    else:
        component_score = max(0.0, 1.0 - (component_ratio - 1.0) / 4.0)

    if np.any(ink_mask) and np.any(~ink_mask):
        fg_mean = float(gray[ink_mask].mean())
        bg_mean = float(gray[~ink_mask].mean())
        separation = np.clip((bg_mean - fg_mean) / 120.0, 0.0, 1.0)
    else:
        separation = 0.0

    score = 100.0 * (
        0.35 * edge_alignment
        + 0.25 * ink_score
        + 0.20 * separation
        + 0.20 * component_score
    )
    return round(float(score), 2)


def build_binarization_candidates(
    img: np.ndarray, enabled_methods: Optional[tuple[str, ...] | list[str] | set[str]] = None
) -> list[BinarizationCandidate]:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img.copy()
    candidates = []
    enabled = set(enabled_methods or BINARIZATION_METHODS)

    block = odd_size(min(gray.shape[:2]) / 18.0, minimum=25, maximum=91)
    sauvola_window = odd_size(min(gray.shape[:2]) / 10.0, minimum=31, maximum=101)

    if "adaptive_gaussian" in enabled:
        adaptive_source = enhance_grayscale(gray, clip_limit=3.0, tile_size=16)
        adaptive = cv2.adaptiveThreshold(
            adaptive_source,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            block,
            15,
        )
        adaptive = cleanup_binary(adaptive)
        candidates.append(
            BinarizationCandidate(
                method="adaptive_gaussian",
                image=adaptive,
                score=score_binarization(gray, adaptive),
            )
        )

    divide_source = None
    if {"otsu_deshadow", "sauvola_deshadow"} & enabled:
        divide_source = enhance_grayscale(remove_shadows_divide(gray), clip_limit=2.8, tile_size=12)

    if "otsu_deshadow" in enabled and divide_source is not None:
        _, otsu = cv2.threshold(
            cv2.GaussianBlur(divide_source, (5, 5), 0),
            0,
            255,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU,
        )
        otsu = cleanup_binary(otsu)
        candidates.append(
            BinarizationCandidate(
                method="otsu_deshadow",
                image=otsu,
                score=score_binarization(gray, otsu),
            )
        )

    if "sauvola_deshadow" in enabled and SKIMAGE_SUPPORT and divide_source is not None:
        threshold_map = threshold_sauvola(divide_source, window_size=sauvola_window, k=0.2)
        sauvola = np.where(divide_source > threshold_map, 255, 0).astype(np.uint8)
        sauvola = cleanup_binary(sauvola)
        candidates.append(
            BinarizationCandidate(
                method="sauvola_deshadow",
                image=sauvola,
                score=score_binarization(gray, sauvola),
            )
        )

    if "hybrid_deshadow" in enabled:
        hybrid_source = enhance_grayscale(remove_shadows_subtract(gray), clip_limit=3.0, tile_size=12)
        local = cv2.adaptiveThreshold(
            hybrid_source,
            255,
            cv2.ADAPTIVE_THRESH_MEAN_C,
            cv2.THRESH_BINARY,
            block,
            10,
        )
        _, global_otsu = cv2.threshold(hybrid_source, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        hybrid = cv2.bitwise_and(local, global_otsu)
        hybrid = cleanup_binary(hybrid)
        candidates.append(
            BinarizationCandidate(
                method="hybrid_deshadow",
                image=hybrid,
                score=score_binarization(gray, hybrid),
            )
        )

    return candidates


def choose_binarization_candidate(
    candidates: list[BinarizationCandidate], mode: str = "auto"
) -> BinarizationCandidate:
    if not candidates:
        raise ValueError("No binarization candidates were produced.")

    if mode != "auto":
        for candidate in candidates:
            if candidate.method == mode:
                return candidate
    return max(candidates, key=lambda candidate: candidate.score)


def compute_sharpness(gray: np.ndarray) -> float:
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def compute_skew(gray: np.ndarray) -> float:
    edges = auto_canny(gray)
    lines = cv2.HoughLinesP(
        edges,
        1,
        np.pi / 180,
        100,
        minLineLength=gray.shape[1] * 0.3,
        maxLineGap=20,
    )
    if lines is None:
        return 0.0

    angles = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        ang = np.degrees(np.arctan2(y2 - y1, x2 - x1))
        if ang < -45:
            ang += 90
        elif ang > 45:
            ang -= 90
        angles.append(ang)
    return float(np.median(angles)) if angles else 0.0


def compute_contrast(gray: np.ndarray) -> float:
    return float(gray.std() / 255.0 * 100.0)


def label_sharpness(value: float) -> str:
    if value > 800:
        return "Excellent"
    if value > 300:
        return "Good"
    if value > 100:
        return "Fair"
    return "Poor"


def label_skew(value: float) -> str:
    value = abs(value)
    if value < 1:
        return "Straight"
    if value < 5:
        return "Slight"
    if value < 15:
        return "Moderate"
    return "Severe"


def label_overall(score: float) -> str:
    if score >= 80:
        return "Excellent"
    if score >= 60:
        return "Good"
    if score >= 40:
        return "Fair"
    return "Poor"


def classify_background(score: float) -> str:
    if score < 0.05:
        return "clean"
    if score < 0.12:
        return "moderate"
    return "cluttered"


def classify_lighting(brightness: float, variation: float) -> str:
    if variation > 9.5:
        return "uneven"
    if brightness < 95:
        return "dim"
    if brightness > 205:
        return "bright"
    return "balanced"


def analyze_scene(img: np.ndarray, corners: np.ndarray) -> SceneAnalysis:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape[:2]

    ordered = order_corners(corners)
    area_ratio = polygon_area(ordered) / max(h * w, 1)

    mask = np.zeros(gray.shape, dtype=np.uint8)
    cv2.fillConvexPoly(mask, ordered.astype(np.int32), 255)
    bg_mask = mask == 0

    if np.mean(bg_mask) < 0.03:
        border = max(8, int(min(h, w) * 0.08))
        bg_mask = np.zeros_like(gray, dtype=bool)
        bg_mask[:border, :] = True
        bg_mask[-border:, :] = True
        bg_mask[:, :border] = True
        bg_mask[:, -border:] = True

    edges = auto_canny(gray)
    bg_edge_density = float(np.mean(edges[bg_mask] > 0)) if np.any(bg_mask) else 0.0
    bg_std = float(gray[bg_mask].std() / 255.0) if np.any(bg_mask) else float(gray.std() / 255.0)
    background_score = 0.7 * bg_edge_density + 0.3 * bg_std

    lighting_variation = compute_shadow_level(gray)
    brightness = float(gray.mean())

    return SceneAnalysis(
        background_complexity=classify_background(background_score),
        lighting_condition=classify_lighting(brightness, lighting_variation),
        lighting_variation=round(lighting_variation, 2),
        background_edge_density=round(bg_edge_density, 4),
        document_area_ratio=round(area_ratio, 3),
    )


def compute_metrics(img: np.ndarray, confidence: float, binarization_score: float) -> QualityMetrics:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
    sharpness = compute_sharpness(gray)
    skew = compute_skew(gray)
    contrast = compute_contrast(gray)
    brightness = float(gray.mean())
    shadow_level = compute_shadow_level(gray)

    sharp_score = min(sharpness / 900.0 * 28.0, 28.0)
    skew_score = max(0.0, 20.0 - abs(skew) * 1.8)
    conf_score = confidence * 15.0
    cont_score = min(contrast / 30.0 * 12.0, 12.0)
    binary_score = min(binarization_score / 100.0 * 15.0, 15.0)
    shadow_score = max(0.0, 10.0 - shadow_level * 0.8)
    overall = sharp_score + skew_score + conf_score + cont_score + binary_score + shadow_score

    return QualityMetrics(
        sharpness=round(sharpness, 2),
        sharpness_label=label_sharpness(sharpness),
        skew_angle=round(skew, 2),
        skew_label=label_skew(skew),
        contrast=round(contrast, 2),
        brightness=round(brightness, 2),
        shadow_level=round(shadow_level, 2),
        binarization_score=round(binarization_score, 2),
        detection_confidence=round(confidence, 3),
        overall_score=round(overall, 1),
        overall_label=label_overall(overall),
    )


def evaluate_quality_gate(
    metrics: QualityMetrics,
    quality_threshold: float = 60.0,
    reject_low_quality: bool = True,
) -> tuple[bool, str, list[str]]:
    flags = []

    if metrics.overall_score < quality_threshold:
        flags.append(f"score<{quality_threshold:.0f}")
    if metrics.detection_confidence < 0.35:
        flags.append("conf<0.35")
    if metrics.sharpness < 120:
        flags.append("blurred")
    if abs(metrics.skew_angle) > 12.0:
        flags.append("skew>12deg")
    if metrics.binarization_score < 45.0:
        flags.append("binary<45")
    if metrics.shadow_level > 55.0:
        flags.append("strong_shadows")

    accepted = (not reject_low_quality) or not flags
    reason = ", ".join(flags)
    return accepted, reason, flags


def estimate_confidence(img: np.ndarray, corners: np.ndarray, method: str) -> float:
    h, w = img.shape[:2]
    area_ratio = polygon_area(corners) / max(h * w, 1)

    if area_ratio < 0.05 or area_ratio > 0.98:
        area_score = 0.3
    elif 0.2 < area_ratio < 0.9:
        area_score = 1.0
    else:
        area_score = 0.7

    angles = [
        angle_at_pt(corners[3], corners[0], corners[1]),
        angle_at_pt(corners[0], corners[1], corners[2]),
        angle_at_pt(corners[1], corners[2], corners[3]),
        angle_at_pt(corners[2], corners[3], corners[0]),
    ]
    angle_score = 1.0 - np.std(angles) / 90.0
    angle_score = max(0.0, min(1.0, angle_score))

    bonus = {"contour": 0.05, "threshold": 0.03, "hough": 0.0, "corners": -0.1}.get(method, 0.0)
    return float(np.clip(0.5 * area_score + 0.4 * angle_score + 0.1 + bonus, 0.0, 1.0))


def _save_debug_image(img: np.ndarray, corners: np.ndarray, path: Path) -> None:
    debug = img.copy()
    pts = corners.astype(np.int32)
    cv2.polylines(debug, [pts], True, (0, 255, 0), 3)
    for idx, pt in enumerate(pts):
        cv2.circle(debug, tuple(pt), 10, (0, 0, 255), -1)
        cv2.putText(
            debug,
            ["TL", "TR", "BR", "BL"][idx],
            (pt[0] + 10, pt[1] + 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 0),
            2,
        )
    cv2.imwrite(str(path), debug)


def _ensure_color(img: np.ndarray) -> np.ndarray:
    if len(img.shape) == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return img


def _make_stage_tile(
    img: np.ndarray,
    title: str,
    tile_size: tuple[int, int] = (290, 380),
    accent_color: tuple[int, int, int] = (210, 216, 224),
) -> np.ndarray:
    tile_h, tile_w = tile_size
    image = _ensure_color(img)
    canvas = np.full((tile_h, tile_w, 3), 250, dtype=np.uint8)

    header_h = 42
    footer_margin = 16
    max_w = tile_w - 24
    max_h = tile_h - header_h - footer_margin
    scale = min(max_w / image.shape[1], max_h / image.shape[0])
    new_w = max(1, int(image.shape[1] * scale))
    new_h = max(1, int(image.shape[0] * scale))
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)

    x0 = (tile_w - new_w) // 2
    y0 = header_h + (max_h - new_h) // 2
    canvas[y0 : y0 + new_h, x0 : x0 + new_w] = resized

    cv2.rectangle(canvas, (0, 0), (tile_w - 1, header_h - 1), (230, 236, 244), thickness=-1)
    cv2.rectangle(canvas, (0, 0), (tile_w - 1, tile_h - 1), accent_color, thickness=3)
    cv2.putText(
        canvas,
        title,
        (14, 27),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (35, 45, 55),
        2,
        cv2.LINE_AA,
    )
    return canvas


def _build_tile_grid(tiles: list[np.ndarray], columns: int = 4) -> np.ndarray:
    if not tiles:
        raise ValueError("Tile grid needs at least one tile.")

    tile_h, tile_w = tiles[0].shape[:2]
    blank = np.full((tile_h, tile_w, 3), 245, dtype=np.uint8)
    rows = []
    for start in range(0, len(tiles), columns):
        row_tiles = tiles[start : start + columns]
        if len(row_tiles) < columns:
            row_tiles = row_tiles + [blank.copy() for _ in range(columns - len(row_tiles))]
        rows.append(np.hstack(row_tiles))
    return np.vstack(rows)


def build_pipeline_preview(
    file_name: str,
    original: np.ndarray,
    corners: np.ndarray,
    rectified: np.ndarray,
    binary: np.ndarray,
    detection_method: str,
    binarization_method: str,
    metrics: QualityMetrics,
    binary_candidates: list[BinarizationCandidate],
    accepted: bool,
    rejection_reason: str,
) -> np.ndarray:
    detected = original.copy()
    pts = corners.astype(np.int32)
    cv2.polylines(detected, [pts], True, (0, 255, 0), 4)
    for label, pt in zip(["TL", "TR", "BR", "BL"], pts):
        cv2.circle(detected, tuple(pt), 9, (0, 0, 255), -1)
        cv2.putText(
            detected,
            label,
            (pt[0] + 10, pt[1] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 0),
            2,
            cv2.LINE_AA,
        )

    rectified_gray = cv2.cvtColor(rectified, cv2.COLOR_BGR2GRAY)
    preprocess_preview = preprocess(original)
    adaptive_source = enhance_grayscale(rectified_gray, clip_limit=3.0, tile_size=16)
    divide_preview = enhance_grayscale(remove_shadows_divide(rectified_gray), clip_limit=2.8, tile_size=12)
    subtract_preview = enhance_grayscale(remove_shadows_subtract(rectified_gray), clip_limit=3.0, tile_size=12)

    candidate_map = {candidate.method: candidate for candidate in binary_candidates}
    tiles = [
        _make_stage_tile(original, "1. Original"),
        _make_stage_tile(detected, f"2. Detectie ({detection_method})"),
        _make_stage_tile(preprocess_preview, "3. Preprocess CLAHE"),
        _make_stage_tile(rectified, "4. Rectified"),
        _make_stage_tile(
            candidate_map["adaptive_gaussian"].image,
            f"5. Adaptive ({candidate_map['adaptive_gaussian'].score:.1f})",
        ),
        _make_stage_tile(divide_preview, "6. De-shadow divide"),
        _make_stage_tile(
            candidate_map["otsu_deshadow"].image,
            f"7. Otsu divide ({candidate_map['otsu_deshadow'].score:.1f})",
        ),
    ]

    if "sauvola_deshadow" in candidate_map:
        tiles.append(
            _make_stage_tile(
                candidate_map["sauvola_deshadow"].image,
                f"8. Sauvola ({candidate_map['sauvola_deshadow'].score:.1f})",
            )
        )

    tiles.extend(
        [
            _make_stage_tile(subtract_preview, "9. De-shadow subtract"),
            _make_stage_tile(
                candidate_map["hybrid_deshadow"].image,
                f"10. Hybrid ({candidate_map['hybrid_deshadow'].score:.1f})",
            ),
            _make_stage_tile(
                binary,
                f"11. Final chosen ({binarization_method})",
                accent_color=(46, 204, 113),
            ),
        ]
    )

    grid = _build_tile_grid(tiles, columns=4)

    header_h = 68
    footer_h = 88
    canvas = np.full((header_h + grid.shape[0] + footer_h, grid.shape[1], 3), 255, dtype=np.uint8)
    canvas[header_h : header_h + grid.shape[0], :, :] = grid

    cv2.rectangle(canvas, (0, 0), (canvas.shape[1] - 1, header_h - 1), (32, 48, 72), thickness=-1)
    cv2.putText(
        canvas,
        f"Document pipeline - {file_name}",
        (18, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.95,
        (245, 247, 250),
        2,
        cv2.LINE_AA,
    )

    status_text = "ACCEPTED" if accepted else "REJECTED"
    status_color = (46, 204, 113) if accepted else (52, 73, 231)
    badge_w = 190 if accepted else 210
    badge_x1 = canvas.shape[1] - badge_w - 18
    badge_y1 = 16
    badge_x2 = canvas.shape[1] - 18
    badge_y2 = 50
    cv2.rectangle(canvas, (badge_x1, badge_y1), (badge_x2, badge_y2), status_color, thickness=-1)
    cv2.putText(
        canvas,
        status_text,
        (badge_x1 + 18, badge_y1 + 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.78,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    footer_y = header_h + grid.shape[0] + 26
    footer_text = (
        f"score={metrics.overall_score:.1f}/100 ({metrics.overall_label})   "
        f"sharp={metrics.sharpness:.1f}   "
        f"skew={metrics.skew_angle:.2f} deg   "
        f"conf={metrics.detection_confidence:.2f}"
    )
    footer_text_2 = (
        f"contrast={metrics.contrast:.1f}%   "
        f"shadow={metrics.shadow_level:.2f}   "
        f"binary_score={metrics.binarization_score:.1f}/100"
    )
    status_line = f"quality={'accepted' if accepted else 'rejected'}"
    if rejection_reason:
        shortened_reason = rejection_reason if len(rejection_reason) <= 110 else f"{rejection_reason[:107]}..."
        status_line += f"   reason={shortened_reason}"
    cv2.putText(canvas, footer_text, (18, footer_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (40, 44, 52), 2, cv2.LINE_AA)
    cv2.putText(
        canvas,
        footer_text_2,
        (18, footer_y + 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (60, 66, 76),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        status_line,
        (18, footer_y + 48),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.62,
        (76, 82, 92),
        2,
        cv2.LINE_AA,
    )
    return canvas


def result_to_summary_row(file_name: str, result: ScanResult) -> dict:
    row = {
        "file": file_name,
        "method": result.method,
        "detection_method": result.method,
        "requested_detection_mode": result.requested_detection_mode,
        "binarization_method": result.binarization_method,
        "detection_success": int(result.method != "fallback"),
        "usable_scan": int(result.accepted),
        "quality_status": "accepted" if result.accepted else "rejected",
        "rejected_input": int(not result.accepted),
        "rejection_reason": result.rejection_reason,
        "quality_flags": "; ".join(result.quality_flags),
    }

    if result.metrics:
        row.update(asdict(result.metrics))
    if result.scene:
        row.update(asdict(result.scene))

    for method_name in BINARIZATION_METHODS:
        row[f"binary_score_{method_name}"] = round(result.binarization_scores.get(method_name, -1.0), 2)
    return row


def resolve_csv_output_dir(output_dir: str, csv_output_dir: Optional[str] = None) -> Path:
    if csv_output_dir:
        return Path(csv_output_dir)

    image_output = Path(output_dir)
    return image_output.parent / f"{image_output.name}_csv"


def write_summary(results_summary: list[dict], output_dir: str) -> Path:
    if not results_summary:
        return Path(output_dir) / "summary.csv"
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    fieldnames = []
    for row in results_summary:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)

    summary_path = out / "summary.csv"
    with open(summary_path, "w", newline="", encoding="utf-8") as csvf:
        writer = csv.DictWriter(csvf, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results_summary)
    return summary_path


def iter_input_files(input_path: Path, recursive: bool = False) -> list[Path]:
    if input_path.is_file():
        return [input_path]

    iterator = input_path.rglob("*") if recursive else input_path.iterdir()
    files = [file for file in iterator if file.is_file() and file.suffix.lower() in SUPPORTED_EXTS]
    return sorted(files)


def scan_document(
    image_path: str,
    output_dir: Optional[str] = None,
    save_debug: bool = False,
    detection_mode: str = "auto",
    binarization_mode: str = "auto",
    save_all_binaries: bool = False,
    save_stage_images: bool = False,
    quality_threshold: float = 60.0,
    reject_low_quality: bool = True,
    candidate_binarization_methods: Optional[tuple[str, ...] | list[str] | set[str]] = None,
) -> ScanResult:
    img = read_image(image_path)
    if img is None:
        return ScanResult(
            False,
            "none",
            "none",
            None,
            None,
            None,
            None,
            None,
            {},
            "Eroare la citirea imaginii.",
            accepted=False,
            rejection_reason="load_failed",
            quality_flags=["load_failed"],
            requested_detection_mode=detection_mode,
        )

    h, w = img.shape[:2]
    scale = 1.0
    if max(h, w) > 2000:
        scale = 2000 / max(h, w)
        small = cv2.resize(img, (int(w * scale), int(h * scale)))
    else:
        small = img.copy()

    gray_small = preprocess(small)
    candidates = []
    detectors = [
        ("contour", detect_contour),
        ("threshold", detect_threshold_quad),
        ("hough", detect_hough),
        ("corners", detect_corners_harris),
    ]
    if detection_mode != "auto":
        detectors = [item for item in detectors if item[0] == detection_mode]
        if not detectors:
            raise ValueError(f"Unsupported detection mode: {detection_mode}")

    for method_name, detector in detectors:
        detected = detector(small, gray_small)
        if detected is None:
            continue
        detected = order_corners(np.array(detected, dtype=np.float32))
        if not validate_corners(small, detected):
            continue
        candidates.append(
            DetectionCandidate(
                method=method_name,
                corners=detected,
                score=score_candidate(small, gray_small, detected, method_name),
            )
        )

    if not candidates:
        method = "fallback"
        corners_full = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype=np.float32)
        confidence = 0.1
    else:
        best = max(candidates, key=lambda candidate: candidate.score)
        method = best.method
        corners_full = (best.corners / scale).astype(np.float32)
        confidence = estimate_confidence(img, corners_full, method)

    try:
        rectified = rectify(img, corners_full)
    except ValueError:
        method = "fallback"
        corners_full = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype=np.float32)
        confidence = 0.1
        rectified = rectify(img, corners_full)

    scene = analyze_scene(img, corners_full)

    binary_candidates = build_binarization_candidates(rectified, enabled_methods=candidate_binarization_methods)
    selected_binary = choose_binarization_candidate(binary_candidates, mode=binarization_mode)
    binarization_scores = {candidate.method: candidate.score for candidate in binary_candidates}

    metrics = compute_metrics(rectified, confidence, selected_binary.score)
    accepted, rejection_reason, quality_flags = evaluate_quality_gate(
        metrics,
        quality_threshold=quality_threshold,
        reject_low_quality=reject_low_quality,
    )

    stem = Path(image_path).stem
    if output_dir:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        pipeline_preview = build_pipeline_preview(
            Path(image_path).name,
            img,
            corners_full,
            rectified,
            selected_binary.image,
            method,
            selected_binary.method,
            metrics,
            binary_candidates,
            accepted,
            rejection_reason,
        )
        cv2.imwrite(str(out / f"{stem}_pipeline.jpg"), pipeline_preview, [cv2.IMWRITE_JPEG_QUALITY, 95])

        if save_stage_images:
            cv2.imwrite(str(out / f"{stem}_rectified.jpg"), rectified, [cv2.IMWRITE_JPEG_QUALITY, 95])
            cv2.imwrite(str(out / f"{stem}_binary.jpg"), selected_binary.image)

        if save_all_binaries:
            for candidate in binary_candidates:
                cv2.imwrite(str(out / f"{stem}_binary_{candidate.method}.jpg"), candidate.image)

        payload = {
            "file": Path(image_path).name,
            "detection_method": method,
            "binarization_method": selected_binary.method,
            "corners": np.round(corners_full, 2).tolist(),
            "requested_detection_mode": detection_mode,
            "detection_success": int(method != "fallback"),
            "usable_scan": int(accepted),
            "quality_status": "accepted" if accepted else "rejected",
            "rejected_input": int(not accepted),
            "rejection_reason": rejection_reason,
            "quality_flags": quality_flags,
            **asdict(metrics),
            **asdict(scene),
            "binarization_scores": {key: round(value, 2) for key, value in binarization_scores.items()},
        }
        with open(out / f"{stem}_metrics.json", "w", encoding="utf-8") as file:
            json.dump(payload, file, indent=2, ensure_ascii=False)
        if save_debug:
            _save_debug_image(img, corners_full, out / f"{stem}_debug.jpg")

    return ScanResult(
        success=True,
        method=method,
        binarization_method=selected_binary.method,
        corners=corners_full.tolist() if corners_full is not None else None,
        rectified=rectified,
        binary=selected_binary.image,
        metrics=metrics,
        scene=scene,
        binarization_scores=binarization_scores,
        message=(
            f"{'ACCEPTED' if accepted else 'REJECTED'} - detectie: {method}, "
            f"binarizare: {selected_binary.method}"
            + (f", motive: {rejection_reason}" if rejection_reason else "")
        ),
        accepted=accepted,
        rejection_reason=rejection_reason,
        quality_flags=quality_flags,
        requested_detection_mode=detection_mode,
    )


def main():
    parser = argparse.ArgumentParser(description="Document Scanner - Pipeline Complet")
    parser.add_argument("input", help="Imagine sau folder cu imagini")
    parser.add_argument("-o", "--output", default="output", help="Folder output")
    parser.add_argument(
        "--csv-output",
        default=None,
        help="Folder separat pentru fisierele CSV; implicit foloseste un folder sibling de forma <output>_csv",
    )
    parser.add_argument("--debug", action="store_true", help="Salveaza imagini debug")
    parser.add_argument("--recursive", action="store_true", help="Cauta imagini si in subfoldere")
    parser.add_argument(
        "--detection",
        default="auto",
        choices=["auto", *DETECTION_METHODS],
        help="Metoda de detectie folosita pentru document sau selectie automata",
    )
    parser.add_argument(
        "--binarization",
        default="auto",
        choices=["auto", *BINARIZATION_METHODS],
        help="Metoda de binarizare folosita pentru scanul final",
    )
    parser.add_argument(
        "--save-all-binaries",
        action="store_true",
        help="Salveaza si variantele intermediare pentru toate metodele de binarizare",
    )
    parser.add_argument(
        "--save-stages-separately",
        action="store_true",
        help="Salveaza si rectified/binary separat; implicit se salveaza doar imaginea compusa de pipeline",
    )
    parser.add_argument(
        "--quality-threshold",
        type=float,
        default=60.0,
        help="Pragul minim al scorului overall pentru a accepta un scan",
    )
    parser.add_argument(
        "--no-reject-low-quality",
        action="store_true",
        help="Nu marca scanurile slabe ca rejectate; salveaza doar scorul si warning-urile",
    )
    args = parser.parse_args()

    inp = Path(args.input)
    files = iter_input_files(inp, recursive=args.recursive)
    csv_output = resolve_csv_output_dir(args.output, args.csv_output)

    if not files:
        print("[ERROR] Nicio imagine gasita.")
        return

    results_summary = []
    for file in files:
        print(f"\n[SCAN] {file.name}")
        result = scan_document(
            str(file),
            args.output,
            args.debug,
            detection_mode=args.detection,
            binarization_mode=args.binarization,
            save_all_binaries=args.save_all_binaries,
            save_stage_images=args.save_stages_separately,
            quality_threshold=args.quality_threshold,
            reject_low_quality=not args.no_reject_low_quality,
        )
        if result.success and result.metrics:
            metrics = result.metrics
            scene = result.scene
            print(f"  Metoda detectie   : {result.method}")
            print(f"  Mod detectie      : {result.requested_detection_mode}")
            print(f"  Binarizare        : {result.binarization_method}")
            print(f"  Sharpness         : {metrics.sharpness:.1f} ({metrics.sharpness_label})")
            print(f"  Skew angle        : {metrics.skew_angle:.2f} deg ({metrics.skew_label})")
            print(f"  Contrast          : {metrics.contrast:.1f}%")
            print(f"  Shadow level      : {metrics.shadow_level:.2f}")
            print(f"  Binary score      : {metrics.binarization_score:.1f}/100")
            print(f"  Confidence        : {metrics.detection_confidence:.2f}")
            print(f"  Overall score     : {metrics.overall_score:.1f}/100 ({metrics.overall_label})")
            print(f"  Quality status    : {'accepted' if result.accepted else 'rejected'}")
            if result.rejection_reason:
                print(f"  Rejection reason  : {result.rejection_reason}")
            if scene:
                print(f"  Background        : {scene.background_complexity}")
                print(f"  Lighting          : {scene.lighting_condition}")
            results_summary.append(result_to_summary_row(file.name, result))
        else:
            print(f"  [FAIL] {result.message}")

    if results_summary:
        summary_path = write_summary(results_summary, str(csv_output))
        print(f"\nImagini salvate in: {args.output}/")
        print(f"CSV salvat in: {summary_path}")
        print("Pentru fiecare fisier se salveaza implicit un singur *_pipeline.jpg")
        print(f"   summary.csv cu {len(results_summary)} inregistrari")


if __name__ == "__main__":
    main()

import os
import uuid
import random
import json
import logging
from dataclasses import dataclass
from typing import Tuple, Dict, Any, Optional, List

import cv2
import numpy as np
from tqdm import tqdm

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

@dataclass
class ComposeOptions:
    """Options for the image synthesis process."""
    # Scale range of the main object relative to the background area
    scale_range: Tuple[float, float] = (0.3, 0.7)
    # Scale range for distractor objects (usually smaller than the main object)
    distractor_scale_range: Tuple[float, float] = (0.1, 0.4)
    # Minimum safety margin from the edge in pixels
    margin: int = 12
    # Feathering radius for edge smoothing
    feather: int = 12
    # Shadow parameters
    add_shadow: bool = True
    shadow_offset: Tuple[int, int] = (6, 6)
    shadow_blur: int = 21
    shadow_alpha: float = 0.35  # Shadow intensity (0.0 to 1.0)
    # Color matching using LAB channel mean/std adjustment
    color_match: bool = True
    # Random seed for reproducibility
    seed: Optional[int] = None
    # Maximum trials to find a non-overlapping placement
    max_place_trials: int = 5000
    # Maximum allowed overlap ratio relative to the new object's area
    max_overlap_ratio: float = 0.15
    # Output file extension ('.png' or '.jpg')
    output_ext: str = ".png"


def _set_seed(seed: Optional[int]):
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)


def _load_image(path: str) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Failed to read image: {path}")
    return img


def _random_scale_factor(bg_w: int, bg_h: int, fg_w: int, fg_h: int, 
                        scale_range: Tuple[float, float], margin: int) -> float:
    """Calculates a random scale factor ensuring the foreground fits within background bounds."""
    max_scale_w = (bg_w - 2 * margin) / max(1, fg_w)
    max_scale_h = (bg_h - 2 * margin) / max(1, fg_h)
    max_scale_fit = max(0.0, min(max_scale_w, max_scale_h))
    
    if max_scale_fit <= 0:
        return min(max_scale_w, max_scale_h)

    low, high = scale_range
    # Area-based scaling for more intuitive visual results
    scale = np.sqrt(random.uniform(low**2, high**2) * (bg_w * bg_h) / max(1, fg_w * fg_h))
    scale = min(scale, max_scale_fit)
    return max(scale, 1e-3)


def _get_affine_matrix(fg_w: int, fg_h: int, scale: float, center_dst: Tuple[float, float]) -> np.ndarray:
    """Generates affine matrix for scaling and translation."""
    cx, cy = fg_w / 2.0, fg_h / 2.0
    px, py = center_dst
    tx = px - scale * cx
    ty = py - scale * cy
    return np.array([[scale, 0.0, tx], [0.0, scale, ty]], dtype=np.float32)


def _transform_points(matrix: np.ndarray, pts: np.ndarray) -> np.ndarray:
    ones = np.ones((pts.shape[0], 1), dtype=np.float32)
    homo = np.hstack([pts.astype(np.float32), ones])
    return (matrix @ homo.T).T


def _get_corners(w: int, h: int) -> np.ndarray:
    return np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype=np.float32)


def _is_inside_bounds(pts: np.ndarray, bg_w: int, bg_h: int, margin: int) -> bool:
    x_ok = (pts[:, 0] >= margin) & (pts[:, 0] < bg_w - margin)
    y_ok = (pts[:, 1] >= margin) & (pts[:, 1] < bg_h - margin)
    return bool(np.all(x_ok & y_ok))


def _build_feather_mask(mask: np.ndarray, feather: int) -> np.ndarray:
    if feather <= 0:
        return (mask.astype(np.float32) / 255.0).clip(0, 1)
    k_size = max(3, int(feather) | 1)
    blur = cv2.GaussianBlur(mask, (k_size, k_size), sigmaX=feather / 2.0, sigmaY=feather / 2.0)
    return (blur.astype(np.float32) / 255.0).clip(0.0, 1.0)


def _apply_shadow(canvas: np.ndarray, mask: np.ndarray, offset: Tuple[int, int], blur: int, alpha: float):
    if alpha <= 0:
        return
    dx, dy = offset
    shadow_mask = np.zeros_like(mask)
    shift_matrix = np.array([[1, 0, dx], [0, 1, dy]], dtype=np.float32)
    cv2.warpAffine(mask, shift_matrix, (mask.shape[1], mask.shape[0]),
                   dst=shadow_mask, flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    
    k_size = max(3, int(blur) | 1)
    shadow_mask = cv2.GaussianBlur(shadow_mask, (k_size, k_size), sigmaX=blur / 2.5, sigmaY=blur / 2.5)
    shadow_factor = (shadow_mask.astype(np.float32) / 255.0) * alpha
    
    for c in range(3):
        canvas[:, :, c] = (canvas[:, :, c].astype(np.float32) * (1.0 - shadow_factor)).clip(0, 255).astype(np.uint8)


def _match_colors_lab(foreground: np.ndarray, background: np.ndarray, mask_alpha: np.ndarray) -> np.ndarray:
    """Matches foreground colors to background using LAB color space statistics."""
    mask = (mask_alpha > 1e-3).astype(np.uint8)
    if mask.sum() < 50:
        return foreground

    f_lab = cv2.cvtColor(foreground, cv2.COLOR_BGR2LAB).astype(np.float32)
    b_lab = cv2.cvtColor(background, cv2.COLOR_BGR2LAB).astype(np.float32)

    m_idx = mask.astype(bool)
    f_pixels = f_lab[m_idx]
    b_pixels = b_lab[m_idx]

    f_mean, f_std = f_pixels.mean(axis=0), f_pixels.std(axis=0) + 1e-5
    b_mean, b_std = b_pixels.mean(axis=0), b_pixels.std(axis=0) + 1e-5

    gain = (b_std / f_std).clip(0.6, 1.8)
    bias = (b_mean - f_mean).clip(-20, 20)

    result_lab = f_lab.copy()
    result_lab[m_idx] = (f_pixels - f_mean) * gain + f_mean + bias
    result_lab = np.clip(result_lab, 0, 255).astype(np.uint8)
    return cv2.cvtColor(result_lab, cv2.COLOR_LAB2BGR)


def synthesize_sample(background_sample: Dict[str, Any],
                      foreground_samples: List[Dict[str, Any]],
                      out_dir: str,
                      options: ComposeOptions = ComposeOptions()) -> Optional[Dict[str, Any]]:
    """
    Composes multiple foreground objects onto a background.
    The first foreground in the list is treated as the 'target' object.
    """
    _set_seed(options.seed)

    bg_img = _load_image(background_sample["image"])
    bg_h, bg_w = bg_img.shape[:2]

    canvas = bg_img.copy()
    occupied_mask = np.zeros((bg_h, bg_w), dtype=np.uint8)
    
    target_transform = None
    target_sample = foreground_samples[0]

    for i, fg_sample in enumerate(foreground_samples):
        is_target = (i == 0)
        
        try:
            fg_img = _load_image(fg_sample["image"])
        except FileNotFoundError:
            logging.warning(f"Foreground image not found: {fg_sample['image']}")
            if is_target: return None
            continue

        fg_h, fg_w = fg_img.shape[:2]
        current_scale_range = options.scale_range if is_target else options.distractor_scale_range
        scale = _random_scale_factor(bg_w, bg_h, fg_w, fg_h, current_scale_range, options.margin)

        fg_corners = _get_corners(fg_w, fg_h)
        placed = False
        m_final = None

        for _ in range(options.max_place_trials):
            cx = random.uniform(0, bg_w - 1)
            cy = random.uniform(0, bg_h - 1)
            m = _get_affine_matrix(fg_w, fg_h, scale, (cx, cy))
            corners_in_bg = _transform_points(m, fg_corners)

            if not _is_inside_bounds(corners_in_bg, bg_w, bg_h, options.margin):
                continue
            
            # Create temporary mask to check overlap
            temp_mask = np.full((fg_h, fg_w), 255, dtype=np.uint8)
            warped_mask = cv2.warpAffine(temp_mask, m, (bg_w, bg_h), flags=cv2.INTER_NEAREST, borderValue=0)
            
            fg_area = np.sum(warped_mask > 0)
            if fg_area == 0: continue
            
            overlap_area = np.sum(np.logical_and(warped_mask > 0, occupied_mask > 0))
            if overlap_area / fg_area < options.max_overlap_ratio:
                placed = True
                m_final = m
                break
        
        if not placed:
            if is_target:
                logging.warning(f"Failed to place target object after {options.max_place_trials} trials.")
                return None
            continue

        # Warping images and masks
        overlay_warped = cv2.warpAffine(fg_img, m_final, (bg_w, bg_h), flags=cv2.INTER_LINEAR)
        mask_raw = np.full((fg_h, fg_w), 255, dtype=np.uint8)
        mask_warped = cv2.warpAffine(mask_raw, m_final, (bg_w, bg_h), flags=cv2.INTER_NEAREST)

        if options.add_shadow:
            _apply_shadow(canvas, mask_warped, options.shadow_offset, options.shadow_blur, options.shadow_alpha)

        alpha_mask = _build_feather_mask(mask_warped, options.feather)

        if options.color_match:
            overlay_warped = _match_colors_lab(overlay_warped, canvas, alpha_mask)
        
        # Alpha blending
        alpha_3d = np.dstack([alpha_mask] * 3)
        canvas = (overlay_warped.astype(np.float32) * alpha_3d + 
                  canvas.astype(np.float32) * (1.0 - alpha_3d)).clip(0, 255).astype(np.uint8)

        occupied_mask[mask_warped > 0] = 255
        if is_target:
            target_transform = m_final

    if target_transform is None:
        return None

    # Map original coordinates to new synthesized image
    orig_x, orig_y = map(float, target_sample["coordinate"])
    new_pt = _transform_points(target_transform, np.array([[orig_x, orig_y]], dtype=np.float32))[0]
    
    final_x = float(max(0, min(bg_w - 1, new_pt[0])))
    final_y = float(max(0, min(bg_h - 1, new_pt[1])))

    # Save output
    os.makedirs(out_dir, exist_ok=True)
    out_filename = f"syn_{uuid.uuid4().hex}{options.output_ext}"
    out_path = os.path.join(out_dir, out_filename)
    
    if options.output_ext.lower() == ".jpg":
        cv2.imwrite(out_path, canvas, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
    else:
        cv2.imwrite(out_path, canvas)

    return {
        "image": out_path,
        "description": target_sample["description"],
        "coordinate": [int(round(final_x)), int(round(final_y))]
    }


def main():
    # --- Configuration ---
    TAG = "dataset_v1"
    NUM_DISTRACTORS = 3
    TOTAL_IMAGES = 50000 
    
    # Path settings
    BG_JSON = "path/to/background.json"
    FG_JSON = "path/to/foreground.json"
    OUTPUT_IMG_DIR = f"./images/images_{TAG}_"
    OUTPUT_JSON = f"./{TAG}_data.json"

    # Load data
    with open(BG_JSON, 'r') as f:
        bg_data = [{"image": img, "description": "", "coordinate": [0, 0]} for img in json.load(f)]
    
    with open(FG_JSON, 'r') as f:
        fg_pool = json.load(f)

    # Synthesis Options
    opts = ComposeOptions(
        scale_range=(0.7, 0.85),
        distractor_scale_range=(0.5, 0.7),
        margin=30,
        feather=5,
        add_shadow=True,
        color_match=True,
        max_overlap_ratio=0.3
    )

    results = []
    used_fgs = set()
    pbar = tqdm(total=TOTAL_IMAGES)

    while len(results) < TOTAL_IMAGES:
        bg = random.choice(bg_data)
        
        if len(fg_pool) < 1 + NUM_DISTRACTORS:
            raise ValueError("Foreground pool is too small.")

        batch_fgs = random.sample(fg_pool, 1 + NUM_DISTRACTORS)
        target_fg = batch_fgs[0]
        
        if target_fg['image'] in used_fgs:
            continue
        
        try:
            sample = synthesize_sample(bg, batch_fgs, OUTPUT_IMG_DIR, opts)
            if sample:
                results.append(sample)
                used_fgs.add(target_fg['image'])
                pbar.update(1)

        except Exception as e:
            logging.error(f"Synthesis failed: {e}")
            continue

        # Save progress periodically
        if len(results) % 500 == 0:
            with open(OUTPUT_JSON, 'w') as f:
                json.dump(results, f, indent=4)
    
    pbar.close()
    with open(OUTPUT_JSON, 'w') as f:
        json.dump(results, f, indent=4)
    logging.info(f"Successfully generated {len(results)} images.")


if __name__ == "__main__":
    main()

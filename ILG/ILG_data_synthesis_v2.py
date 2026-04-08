import os
import uuid
import random
import json
import cv2
import numpy as np
from dataclasses import dataclass, field
from typing import Tuple, Dict, Any, Optional, List
from tqdm import tqdm


@dataclass
class ComposeOptions:
    """Configuration options for the synthesis process."""
    scale_range: Tuple[float, float] = (0.3, 0.7)
    distractor_scale_range: Tuple[float, float] = (0.1, 0.4)
    margin: int = 12
    feather: int = 12
    add_shadow: bool = True
    shadow_offset: Tuple[int, int] = (6, 6)
    shadow_blur: int = 21
    shadow_alpha: float = 0.35
    color_match: bool = True
    seed: Optional[int] = None
    max_place_trials: int = 5000
    max_overlap_ratio: float = 0.15
    output_ext: str = ".png"
    # Background enhancement config
    bg_grid: Tuple[int, int] = (1, 1)
    bg_mode: str = "repeat"  # "repeat" or "random_pool"
    bg_keep_aspect: bool = True
    bg_random_flip: bool = True
    bg_seam_blur: int = 3


# ----------------------------------------------------------------------------
# Internal Utility Functions
# ----------------------------------------------------------------------------

def _set_seed(seed: Optional[int]):
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)


def _load_bgr(path: str) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Failed to read image: {path}")
    return img


def _random_scale_factor(Aw: int, Ah: int, Bw: int, Bh: int, 
                        scale_range: Tuple[float, float], margin: int) -> float:
    max_scale_by_w = (Aw - 2 * margin) / max(1, Bw)
    max_scale_by_h = (Ah - 2 * margin) / max(1, Bh)
    max_scale_fit = max(0.0, min(max_scale_by_w, max_scale_by_h))
    
    if max_scale_fit <= 0:
        return min(max_scale_by_w, max_scale_by_h)
        
    lo, hi = scale_range
    s = np.sqrt(random.uniform(lo**2, hi**2) * (Aw * Ah) / max(1, Bw * Bh))
    return max(1e-3, min(s, max_scale_fit))


def _get_affine_matrix(Bw: int, Bh: int, scale: float, center_dst: Tuple[float, float]) -> np.ndarray:
    cx, cy = Bw / 2.0, Bh / 2.0
    px, py = center_dst
    tx = px - scale * cx
    ty = py - scale * cy
    return np.array([[scale, 0.0, tx], [0.0, scale, ty]], dtype=np.float32)


def _transform_points(M: np.ndarray, pts: np.ndarray) -> np.ndarray:
    ones = np.ones((pts.shape[0], 1), dtype=np.float32)
    homo = np.hstack([pts.astype(np.float32), ones])
    return (M @ homo.T).T


def _corners(w: int, h: int) -> np.ndarray:
    return np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype=np.float32)


def _inside_bounds(pts: np.ndarray, Aw: int, Ah: int, margin: int) -> bool:
    x_ok = (pts[:, 0] >= margin) & (pts[:, 0] < Aw - margin)
    y_ok = (pts[:, 1] >= margin) & (pts[:, 1] < Ah - margin)
    return bool(np.all(x_ok & y_ok))


def _build_feather_mask(mask: np.ndarray, feather: int) -> np.ndarray:
    if feather <= 0:
        return (mask.astype(np.float32) / 255.0).clip(0, 1)
    k = max(3, int(feather) | 1)
    blur = cv2.GaussianBlur(mask, (k, k), sigmaX=feather / 2.0, sigmaY=feather / 2.0)
    return (blur.astype(np.float32) / 255.0).clip(0.0, 1.0)


def _apply_shadow(bg: np.ndarray, mask_warp: np.ndarray, offset: Tuple[int, int], blur: int, alpha: float):
    if alpha <= 0:
        return
    dx, dy = offset
    shadow = np.zeros_like(mask_warp)
    M_shift = np.array([[1, 0, dx], [0, 1, dy]], dtype=np.float32)
    cv2.warpAffine(mask_warp, M_shift, (mask_warp.shape[1], mask_warp.shape[0]),
                   dst=shadow, flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    k = max(3, int(blur) | 1)
    shadow = cv2.GaussianBlur(shadow, (k, k), sigmaX=blur / 2.5, sigmaY=blur / 2.5)
    shadow_f = (shadow.astype(np.float32) / 255.0) * alpha
    for c in range(3):
        bg[:, :, c] = (bg[:, :, c].astype(np.float32) * (1.0 - shadow_f)).clip(0, 255).astype(np.uint8)


def _color_match_lab(overlay: np.ndarray, bg: np.ndarray, mask_alpha: np.ndarray) -> np.ndarray:
    mask = (mask_alpha > 1e-3).astype(np.uint8)
    if mask.sum() < 50:
        return overlay
    olab = cv2.cvtColor(overlay, cv2.COLOR_BGR2LAB).astype(np.float32)
    blab = cv2.cvtColor(bg, cv2.COLOR_BGR2LAB).astype(np.float32)
    m = mask.astype(bool)
    o_pixels, b_pixels = olab[m], blab[m]
    o_mean, o_std = o_pixels.mean(axis=0), o_pixels.std(axis=0) + 1e-5
    b_mean, b_std = b_pixels.mean(axis=0), b_pixels.std(axis=0) + 1e-5
    gain = (b_std / o_std).clip(0.6, 1.8)
    bias = (b_mean - o_mean).clip(-20, 20)
    adj = olab.copy()
    adj[m] = (o_pixels - o_mean) * gain + o_mean + bias
    return cv2.cvtColor(np.clip(adj, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)


# ----------------------------------------------------------------------------
# Background Processing
# ----------------------------------------------------------------------------

def enhance_background(base_img: np.ndarray, 
                       options: ComposeOptions, 
                       pool: Optional[List[str]] = None) -> np.ndarray:
    """Enhances background resolution by tiling images in a grid."""
    rows, cols = max(1, options.bg_grid[0]), max(1, options.bg_grid[1])
    if rows == 1 and cols == 1:
        return base_img

    tile_h, tile_w = base_img.shape[:2]
    canvas = np.zeros((tile_h * rows, tile_w * cols, 3), dtype=np.uint8)

    def get_tile():
        if options.bg_mode == "random_pool" and pool:
            try: return _load_bgr(random.choice(pool))
            except: return base_img
        return base_img

    for r in range(rows):
        for c in range(cols):
            tile = get_tile()
            if options.bg_keep_aspect:
                # Resize cover logic
                h, w = tile.shape[:2]
                scale = max(tile_w / w, tile_h / h)
                nw, nh = int(round(w * scale)), int(round(h * scale))
                resized = cv2.resize(tile, (nw, nh))
                tile = resized[(nh-tile_h)//2 : (nh-tile_h)//2 + tile_h, 
                               (nw-tile_w)//2 : (nw-tile_w)//2 + tile_w]
            else:
                tile = cv2.resize(tile, (tile_w, tile_h))
            
            if options.bg_random_flip:
                if random.random() < 0.3: tile = cv2.flip(tile, 1)
                if random.random() < 0.1: tile = cv2.flip(tile, 0)
            
            canvas[r*tile_h:(r+1)*tile_h, c*tile_w:(c+1)*tile_w] = tile

    if options.bg_seam_blur > 0:
        k = max(3, int(options.bg_seam_blur) | 1)
        mask = np.zeros(canvas.shape[:2], dtype=np.uint8)
        lw = max(1, options.bg_seam_blur // 2)
        for i in range(1, cols): cv2.rectangle(mask, (i*tile_w-lw, 0), (i*tile_w+lw, canvas.shape[0]), 255, -1)
        for i in range(1, rows): cv2.rectangle(mask, (0, i*tile_h-lw), (canvas.shape[1], i*tile_h+lw), 255, -1)
        blurred = cv2.GaussianBlur(canvas, (k, k), 0)
        alpha = (cv2.GaussianBlur(mask, (k, k), 0).astype(np.float32) / 255.0)[..., None]
        canvas = (blurred * alpha + canvas * (1 - alpha)).astype(np.uint8)

    return canvas


# ----------------------------------------------------------------------------
# Core Synthesis
# ----------------------------------------------------------------------------

def synthesize_multiple(background_sample: Dict[str, Any],
                        foreground_samples: List[Dict[str, Any]],
                        out_dir: str,
                        options: ComposeOptions,
                        bg_pool: Optional[List[str]] = None) -> Optional[Dict[str, Any]]:
    """Compose multiple foreground objects onto a background image."""
    _set_seed(options.seed)
    
    try:
        bg_img = _load_bgr(background_sample["image"])
    except Exception as e:
        print(f"Error loading background: {e}")
        return None

    if options.bg_grid[0] > 1 or options.bg_grid[1] > 1:
        bg_img = enhance_background(bg_img, options, bg_pool)

    ah, aw = bg_img.shape[:2]
    composite_img = bg_img.copy()
    occupied_mask = np.zeros((ah, aw), dtype=np.uint8)
    main_transform = None
    main_fg_sample = foreground_samples[0]

    for i, fg_sample in enumerate(foreground_samples):
        is_main = (i == 0)
        try:
            fg_img = _load_bgr(fg_sample["image"])
        except:
            if is_main: return None
            continue

        bh, bw = fg_img.shape[:2]
        s_range = options.scale_range if is_main else options.distractor_scale_range
        scale = _random_scale_factor(aw, ah, bw, bh, s_range, options.margin)
        
        corners_b = _corners(bw, bh)
        m_final = None

        for _ in range(options.max_place_trials):
            cx = random.uniform(0, aw - 1)
            cy = random.uniform(0, ah - 1)
            m = _get_affine_matrix(bw, bh, scale, (cx, cy))
            pts_a = _transform_points(m, corners_b)

            if _inside_bounds(pts_a, aw, ah, options.margin):
                mask_candidate = cv2.warpAffine(np.full((bh, bw), 255, dtype=np.uint8), m, (aw, ah), flags=cv2.INTER_NEAREST)
                area = np.sum(mask_candidate > 0)
                if area > 0:
                    overlap = np.sum(np.logical_and(mask_candidate > 0, occupied_mask > 0))
                    if overlap / area < options.max_overlap_ratio:
                        m_final = m
                        break
        
        if m_final is None:
            if is_main: return None
            continue

        overlay_w = cv2.warpAffine(fg_img, m_final, (aw, ah))
        mask_w = cv2.warpAffine(np.full((bh, bw), 255, dtype=np.uint8), m_final, (aw, ah), flags=cv2.INTER_NEAREST)

        if options.add_shadow:
            _apply_shadow(composite_img, mask_w, options.shadow_offset, options.shadow_blur, options.shadow_alpha)

        alpha_soft = _build_feather_mask(mask_w, options.feather)
        if options.color_match:
            overlay_w = _color_match_lab(overlay_w, composite_img, alpha_soft)
        
        alpha3 = alpha_soft[..., None]
        composite_img = (overlay_w * alpha3 + composite_img * (1.0 - alpha3)).astype(np.uint8)
        occupied_mask[mask_w > 0] = 255
        if is_main: main_transform = m_final

    # Calculate final coordinates for grounding
    orig_coords = np.array([main_fg_sample["coordinate"]], dtype=np.float32)
    new_pt = _transform_points(main_transform, orig_coords)[0]
    final_x = int(np.clip(new_pt[0], 0, aw - 1))
    final_y = int(np.clip(new_pt[1], 0, ah - 1))

    os.makedirs(out_dir, exist_ok=True)
    filename = f"syn_{uuid.uuid4().hex}{options.output_ext}"
    filepath = os.path.join(out_dir, filename)
    cv2.imwrite(filepath, composite_img)

    return {
        "image": filepath,
        "description": main_fg_sample["description"],
        "coordinate": [final_x, final_y]
    }


# ----------------------------------------------------------------------------
# Main Execution
# ----------------------------------------------------------------------------

if __name__ == "__main__":
    # Settings
    DATA_TAG = "PART_1"
    TARGET_COUNT = 50000
    DISTRACTOR_COUNT = 3
    
    # Path configuration
    BG_PATH = "path/to/background.json"
    FG_PATH = "path/to/foreground.json"
    OUTPUT_DIR = f"./images/{DATA_TAG}_"
    JSON_OUT = f"./{DATA_TAG}_ILG.json"

    # Initialize Options
    opt = ComposeOptions(
        scale_range=(0.7, 0.8),
        distractor_scale_range=(0.4, 0.6),
        # bg_grid=(1, 1),  # Increase resolution 1x horizontally
        bg_grid=(1, 2),  # Increase resolution 2x horizontally
        bg_mode="random_pool",
        color_match=True,
        output_ext=".png"
    )

    # Load Source Data
    with open(BG_PATH, 'r') as f:
        bg_list = [{"image": p, "description": "", "coordinate": [0,0]} for p in json.load(f)]
    with open(FG_PATH, 'r') as f:
        fg_list = json.load(f)
    
    bg_pool = [b["image"] for b in bg_list]
    
    generated_data = []
    used_main_fgs = set()
    pbar = tqdm(total=TARGET_COUNT, desc="Synthesizing")

    while len(generated_data) < TARGET_COUNT:
        bg_sample = random.choice(bg_list)
        batch_fgs = random.sample(fg_list, 1 + DISTRACTOR_COUNT)
        
        main_obj = batch_fgs[0]
        if main_obj['image'] in used_main_fgs:
            continue
        
        result = synthesize_multiple(bg_sample, batch_fgs, OUTPUT_DIR, opt, bg_pool)
        
        if result:
            generated_data.append(result)
            used_main_fgs.add(main_obj['image'])
            pbar.update(1)
            
            # Periodic save
            if len(generated_data) % 500 == 0:
                with open(JSON_OUT, 'w') as f:
                    json.dump(generated_data, f, indent=4)

    pbar.close()
    with open(JSON_OUT, 'w') as f:
        json.dump(generated_data, f, indent=4)
    print(f"Generation complete: {len(generated_data)} images saved.")

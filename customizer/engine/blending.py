"""
Realistic Blending
--------------------
Makes the warped, fold-conformed design actually look printed on the
fabric: the base photo's lighting, shadows, and weave texture must remain
visible *through* the design, not be covered by a flat sticker.

Technique: multiply the design by a normalized lighting map derived from the
base photo's luminance in the print area (classic "linear light" garment
mockup approach), then composite with a feathered alpha mask so the edges
blend rather than looking cut out.
"""
import numpy as np
import cv2


def build_lighting_map(base_image_bgr: np.ndarray, print_area: dict, clip=(0.45, 1.55)):
    """
    Normalized per-pixel brightness multiplier for the print area, centered
    at 1.0 (average brightness). Values >1 = highlight, <1 = shadow.
    """
    x, y, w, h = print_area["x"], print_area["y"], print_area["w"], print_area["h"]
    region = base_image_bgr[y:y + h, x:x + w]
    gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY).astype(np.float32)

    mean = gray.mean() or 1.0
    lighting = gray / mean
    lighting = np.clip(lighting, clip[0], clip[1])
    return lighting  # (h, w)


def feather_alpha(alpha: np.ndarray, radius: int = 3) -> np.ndarray:
    if radius <= 0:
        return alpha
    k = radius * 2 + 1
    return cv2.GaussianBlur(alpha, (k, k), 0)


def sharpen_design(design_rgb: np.ndarray, alpha: np.ndarray, amount: float = 0.6, sigma: float = 1.0) -> np.ndarray:
    """
    Unsharp mask, applied only where the design is actually visible (alpha).
    Two perspective/displacement resample passes (see pipeline.py) each add a
    little softening -- this recovers edge crispness, which matters most for
    small print areas and lower-resolution uploaded logos where that
    accumulated blur is otherwise very noticeable (fine strokes/edges wash out).
    """
    blurred = cv2.GaussianBlur(design_rgb, (0, 0), sigmaX=sigma)
    sharpened = design_rgb + (design_rgb - blurred) * amount
    sharpened = np.clip(sharpened, 0, 255)
    # Only apply where the design has real coverage, so we don't sharpen noise
    # in near-transparent fringe pixels.
    mask = (alpha > 0.05)[:, :, None]
    return np.where(mask, sharpened, design_rgb)


def composite_design(
    base_image_bgr: np.ndarray,
    design_rgba: np.ndarray,
    print_area: dict,
    fabric_texture_strength: float = 0.12,
    sharpen_amount: float = 0.6,
) -> np.ndarray:
    """
    Blend a perspective-warped + fold-conformed design (RGBA, matching the
    print area's w/h) onto the base product photo at the print area location.

    Returns the full composited base image (BGR, same size as input).
    """
    x, y, w, h = print_area["x"], print_area["y"], print_area["w"], print_area["h"]
    out = base_image_bgr.copy()

    if design_rgba.shape[0] != h or design_rgba.shape[1] != w:
        design_rgba = cv2.resize(design_rgba, (w, h), interpolation=cv2.INTER_CUBIC)

    design_rgb = design_rgba[:, :, :3].astype(np.float32)
    alpha = (design_rgba[:, :, 3].astype(np.float32) / 255.0)

    if sharpen_amount > 0:
        design_rgb = sharpen_design(design_rgb, alpha, amount=sharpen_amount)

    alpha = feather_alpha(alpha, radius=2)

    lighting = build_lighting_map(base_image_bgr, print_area)  # (h, w)
    lighting_3c = lighting[:, :, None]

    # 1) Multiply blend: design darkens/lightens with the fabric's real lighting
    lit_design = design_rgb * lighting_3c

    # 2) Let a touch of the base fabric's fine texture show through (weave/threads)
    base_region = base_image_bgr[y:y + h, x:x + w].astype(np.float32)
    base_detail = base_region - cv2.GaussianBlur(base_region, (0, 0), sigmaX=3.0)
    lit_design = lit_design + base_detail * fabric_texture_strength

    lit_design = np.clip(lit_design, 0, 255)

    region = out[y:y + h, x:x + w].astype(np.float32)
    blended = region * (1 - alpha[:, :, None]) + lit_design * alpha[:, :, None]
    out[y:y + h, x:x + w] = np.clip(blended, 0, 255).astype(np.uint8)
    return out

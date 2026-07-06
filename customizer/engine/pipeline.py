"""
Rendering Pipeline
--------------------
Two distinct phases, which is the key to the performance requirement:

  PHASE 1 - ANALYSIS (slow, ~100-300ms, runs ONCE per product photo):
      Perspective quad detection + displacement/height map extraction.
      Triggered when an admin uploads/edits a product photo, not per request.
      Results are cached (see models.ProductImageAnalysis).

  PHASE 2 - RENDER (fast, ~10-40ms even at high res, runs per customization
      request): perspective-warp the design into the cached quad, bend it
      using the cached height map, blend it using the cached lighting map.
      This is pure vectorized numpy/OpenCV -- no analysis, no ML inference --
      which is what makes concurrent high-volume rendering practical.
"""
from dataclasses import dataclass
import numpy as np
import cv2

from .perspective import detect_surface_quad
from .displacement import build_displacement_map, apply_fabric_conformation
from .blending import composite_design

# Below this size (in px, on the print area's shorter side), cubic
# interpolation's edge ringing/overshoot becomes visually dominant relative
# to the tiny pixel budget -- e.g. a cap-back embroidery zone that's only
# 40-50px tall. Falling back to linear there trades a little softness for
# avoiding the blocky/haloed look ringing produces at that scale.
SMALL_AREA_PX = 80


def _warp_interp(min_dim: int) -> int:
    return cv2.INTER_CUBIC if min_dim >= SMALL_AREA_PX else cv2.INTER_LINEAR


@dataclass
class ProductImageAnalysisResult:
    quad: np.ndarray            # (4,2) destination corners for perspective warp
    height_map: np.ndarray      # (h,w) fold/wrinkle map, cached per print area
    meta: dict                  # tilt_deg, foreshorten, etc. (useful for admin QA)


def analyze_product_image(base_image_bgr: np.ndarray, print_area: dict, max_tilt_deg: float = 18.0) -> ProductImageAnalysisResult:
    """PHASE 1. Run once per product photo and cache the result.

    `max_tilt_deg` bounds how much automatic tilt detection is allowed to
    shear the design. Flat, large-panel garments (t-shirt front/back) are
    well-behaved with the default. Small curved panels (cap sides, sleeves)
    have busier stitching/seam edges that can fool the gradient-based
    detector into reporting more tilt than is really there -- when that
    happens, detect_surface_quad saturates at this clamp, which is a signal
    to use a lower, more conservative value for that product type rather
    than trusting the raw estimate.
    """
    quad, meta = detect_surface_quad(base_image_bgr, print_area, max_tilt_deg=max_tilt_deg)
    height_map = build_displacement_map(base_image_bgr, print_area)
    return ProductImageAnalysisResult(quad=quad, height_map=height_map, meta=meta)


def _rotate_rgba_no_crop(rgba: np.ndarray, angle_deg: float, interp: int) -> np.ndarray:
    """
    Rotate an RGBA image around its own center by angle_deg, expanding the
    output canvas so corners aren't clipped (standard "rotate without
    cropping" trick: new canvas = bounding box of the rotated rect, with the
    rotation matrix's translation adjusted to re-center into it).
    """
    h, w = rgba.shape[:2]
    if abs(angle_deg) < 0.01:
        return rgba

    center = (w / 2.0, h / 2.0)
    M = cv2.getRotationMatrix2D(center, angle_deg, 1.0)

    cos, sin = abs(M[0, 0]), abs(M[0, 1])
    new_w = int(np.ceil(h * sin + w * cos))
    new_h = int(np.ceil(h * cos + w * sin))
    M[0, 2] += (new_w / 2.0) - center[0]
    M[1, 2] += (new_h / 2.0) - center[1]

    return cv2.warpAffine(
        rgba, M, (new_w, new_h),
        flags=interp,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0, 0),
    )


def _place_on_canvas(source_rgba: np.ndarray, canvas_h: int, canvas_w: int, center_x: float, center_y: float) -> np.ndarray:
    """
    Paste `source_rgba` onto a fresh transparent (canvas_h, canvas_w) canvas,
    centered at (center_x, center_y) in canvas coordinates. Safely clips if
    the source is larger than the canvas or would extend past its edges --
    e.g. when the user scales a design up past the print area, or drags it
    toward an edge -- rather than raising an index error.
    """
    canvas = np.zeros((canvas_h, canvas_w, 4), dtype=np.uint8)
    sh, sw = source_rgba.shape[:2]

    dst_x0 = int(round(center_x - sw / 2.0))
    dst_y0 = int(round(center_y - sh / 2.0))
    dst_x1, dst_y1 = dst_x0 + sw, dst_y0 + sh

    # Intersect the destination rect with the canvas bounds
    clip_x0, clip_y0 = max(0, dst_x0), max(0, dst_y0)
    clip_x1, clip_y1 = min(canvas_w, dst_x1), min(canvas_h, dst_y1)
    if clip_x0 >= clip_x1 or clip_y0 >= clip_y1:
        return canvas  # placed entirely off-canvas -- nothing visible, not an error

    src_x0, src_y0 = clip_x0 - dst_x0, clip_y0 - dst_y0
    src_x1, src_y1 = src_x0 + (clip_x1 - clip_x0), src_y0 + (clip_y1 - clip_y0)

    canvas[clip_y0:clip_y1, clip_x0:clip_x1] = source_rgba[src_y0:src_y1, src_x0:src_x1]
    return canvas


def render_customization(
    base_image_bgr: np.ndarray,
    design_rgba: np.ndarray,
    print_area: dict,
    analysis: ProductImageAnalysisResult,
    fold_strength: float = 10.0,
    scale: float = 1.0,
    rotation_deg: float = 0.0,
    offset_x: float = 0.0,
    offset_y: float = 0.0,
) -> np.ndarray:
    """
    PHASE 2. Fast per-request render using a cached analysis result.

    `scale`, `rotation_deg`, `offset_x`, `offset_y` come from the interactive
    preview editor (move/resize/rotate before "Generate Final Mockup") and
    default to exactly the old auto-centered, auto-fit behavior when left
    unspecified -- existing callers (and the original demo page) are
    unaffected.

      scale:        multiplier on top of the automatic "fit inside the print
                    area" size. 1.0 = fills the box as before; >1 = bigger,
                    <1 = smaller. Clamped to [0.1, 4.0].
      rotation_deg: additional rotation, applied before the perspective warp.
      offset_x/y:   fraction of the print area's width/height to shift the
                    design's center away from the box's center, in [-0.5, 0.5].
                    0, 0 = centered (the old default).

    Steps map directly to the spec:
      1. Perspective Alignment  -> warpPerspective into analysis.quad
      2. Fabric Conformation    -> remap using analysis.height_map
      3. Realistic Blending     -> composite_design()
    """
    x, y, w, h = print_area["x"], print_area["y"], print_area["w"], print_area["h"]
    is_default_placement = (scale == 1.0 and rotation_deg == 0.0 and offset_x == 0.0 and offset_y == 0.0)

    scale = float(np.clip(scale, 0.1, 4.0))
    offset_x = float(np.clip(offset_x, -0.5, 0.5))
    offset_y = float(np.clip(offset_y, -0.5, 0.5))

    # --- 0. Fit the uploaded design into the print area, preserving its
    # aspect ratio, before doing anything else. Without this step the design
    # is stretched to whatever shape the print area happens to be (e.g. a
    # square logo squashed into a wide, short cap panel) and the perspective
    # transform below samples the wrong region of the source image, since it
    # assumes the source frame is already (w, h). Both bugs together are
    # what caused designs to render squished into a corner of the print area.
    design_h, design_w = design_rgba.shape[:2]
    base_fit_scale = min(w / design_w, h / design_h)
    effective_scale = base_fit_scale * scale
    # Safety cap: even at scale=4.0 on a tiny box, don't resize to something
    # absurdly large -- clip to 3x the print area's longer side.
    max_dim = 3 * max(w, h)
    fit_w = max(1, min(max_dim, int(round(design_w * effective_scale))))
    fit_h = max(1, min(max_dim, int(round(design_h * effective_scale))))

    fitted = cv2.resize(
        design_rgba, (fit_w, fit_h),
        interpolation=_warp_interp(min(fit_w, fit_h)) if effective_scale > 1.0 else cv2.INTER_AREA,
    )

    if rotation_deg:
        fitted = _rotate_rgba_no_crop(fitted, rotation_deg, _warp_interp(min(fit_w, fit_h)))

    center_x = (w / 2.0) + offset_x * w
    center_y = (h / 2.0) + offset_y * h
    design_rgba = _place_on_canvas(fitted, h, w, center_x, center_y)

    # --- 1. Perspective Alignment ---
    src_pts = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32)
    # quad is in full-image coordinates; make it relative to the print area
    # origin since we warp the design into a w x h canvas first.
    quad_local = analysis.quad.copy()
    quad_local[:, 0] -= x
    quad_local[:, 1] -= y

    M = cv2.getPerspectiveTransform(src_pts, quad_local)
    warped = cv2.warpPerspective(
        design_rgba, M, (w, h),
        flags=_warp_interp(min(w, h)),
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0, 0),
    )

    # --- 2. Fabric Conformation ---
    bent = apply_fabric_conformation(warped, analysis.height_map, strength=fold_strength)

    # --- 2b. Guaranteed centering (default placement only) ---
    # Perspective tilt and fold-bending are centroid-neutral by construction
    # (see perspective.py / displacement.py), but photo-specific edge cases
    # can still leave a small residual drift. When the caller didn't ask for
    # a specific manual placement, we measure the actual result and correct
    # it, guaranteeing sub-pixel centering rather than hoping the heuristics
    # cancel out. When the user *did* deliberately move/scale/rotate the
    # design in the preview editor, we respect that placement instead of
    # silently re-centering it back -- that would undo their edit.
    if is_default_placement:
        bent = _recenter_to_box(bent)

    # --- 3. Realistic Blending ---
    # Sharpening helps recover crispness lost to resampling on normal-sized
    # print areas, but on very small ones (embroidery-scale) there are so
    # few pixels that the same unsharp-mask amount over-amplifies edges into
    # a blocky/haloed look instead of perceived detail -- scale it down.
    sharpen_amount = 0.6 if min(w, h) >= SMALL_AREA_PX else 0.15
    result = composite_design(base_image_bgr, bent, print_area, sharpen_amount=sharpen_amount)
    return result


def _recenter_to_box(rgba: np.ndarray, alpha_threshold: int = 15) -> np.ndarray:
    """
    Shift `rgba` (design already warped into a canvas the size of the print
    area) so the centroid of its visible (alpha > threshold) pixels sits
    exactly at the canvas center. No-op if the design is fully transparent.
    """
    h, w = rgba.shape[:2]
    alpha = rgba[:, :, 3]
    ys, xs = np.where(alpha > alpha_threshold)
    if xs.size == 0:
        return rgba

    cx, cy = xs.mean(), ys.mean()
    dx, dy = (w / 2.0) - cx, (h / 2.0) - cy

    if abs(dx) < 0.5 and abs(dy) < 0.5:
        return rgba  # already centered within half a pixel

    M = np.array([[1, 0, dx], [0, 1, dy]], dtype=np.float32)
    return cv2.warpAffine(
        rgba, M, (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0, 0),
    )


def load_rgba(path: str) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(path)
    if img.shape[2] == 3:
        alpha = np.full(img.shape[:2], 255, dtype=np.uint8)
        img = np.dstack([img, alpha])
    return img


def load_bgr(path: str) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(path)
    return img

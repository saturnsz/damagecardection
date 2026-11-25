"""
Image Quality Gatekeeper Module
================================

Runs BEFORE YOLO inference to check image quality and optionally enhance
dark images. Blurry images are rejected outright; dark images are enhanced
with CLAHE; images that pass both checks proceed unchanged.

Dependencies: cv2, numpy (no other packages required).
"""

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Default configuration (IQ-6)
# ---------------------------------------------------------------------------

DEFAULT_CONFIG: dict = {
    "blur_threshold": 100,        # Laplacian variance below this → blurry
    "dark_threshold": 80,         # Mean L* below this → dark
    "clahe_clip_limit": 2.0,      # CLAHE clip limit
    "clahe_tile_grid_size": (8, 8),  # CLAHE tile grid size
    "resize_width": 500,          # Width (px) used for blur measurement only
}


# ---------------------------------------------------------------------------
# Helper: validate that an image array is usable
# ---------------------------------------------------------------------------

def _validate_image(image_bgr: np.ndarray) -> None:
    """Raise ``ValueError`` if *image_bgr* is None, empty, or not a valid array."""
    if image_bgr is None:
        raise ValueError("Image is None.")
    if not isinstance(image_bgr, np.ndarray):
        raise ValueError(f"Expected numpy.ndarray, got {type(image_bgr).__name__}.")
    if image_bgr.size == 0:
        raise ValueError("Image array is empty (zero elements).")
    if image_bgr.ndim < 2:
        raise ValueError(f"Image must have at least 2 dimensions, got {image_bgr.ndim}.")


# ---------------------------------------------------------------------------
# (IQ-1 / IQ-2)  Blur detection
# ---------------------------------------------------------------------------

def check_blur(image_bgr: np.ndarray, threshold: float = 100,
               resize_width: int = 500) -> dict:
    """Check whether an image is blurry using Laplacian variance.

    The image is first resized to *resize_width* (maintaining aspect ratio) so
    that the metric is resolution-independent (IQ-1).  The resized image is
    converted to grayscale and the variance of its Laplacian is computed
    (IQ-2).  A variance below *threshold* indicates blur.

    Parameters
    ----------
    image_bgr : np.ndarray
        Input image in BGR colour order.
    threshold : float, optional
        Laplacian-variance threshold (default 100).
    resize_width : int, optional
        Width in pixels to resize to before measuring (default 500).

    Returns
    -------
    dict
        ``is_blurry`` (bool) – True if the image is considered blurry.
        ``laplacian_var`` (float) – The computed Laplacian variance.
    """
    _validate_image(image_bgr)

    # Resize to fixed reference width, keeping aspect ratio (IQ-1)
    h, w = image_bgr.shape[:2]
    if w > 0 and resize_width > 0:
        scale = resize_width / w
        new_h = max(1, int(h * scale))
        resized = cv2.resize(image_bgr, (resize_width, new_h),
                             interpolation=cv2.INTER_AREA)
    else:
        resized = image_bgr

    # Grayscale → Laplacian variance (IQ-2)
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY) if resized.ndim == 3 else resized
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    laplacian_var = float(laplacian.var())

    return {
        "is_blurry": laplacian_var < threshold,
        "laplacian_var": laplacian_var,
    }


# ---------------------------------------------------------------------------
# (IQ-3)  Brightness detection
# ---------------------------------------------------------------------------

def check_brightness(image_bgr: np.ndarray,
                     dark_threshold: float = 80) -> dict:
    """Check whether an image is too dark.

    Converts to CIE-LAB colour space and computes the mean of the L* channel
    (IQ-3).  An L* mean below *dark_threshold* flags the image as dark.

    Parameters
    ----------
    image_bgr : np.ndarray
        Input image in BGR colour order.
    dark_threshold : float, optional
        L* mean below this value is considered dark (default 80).

    Returns
    -------
    dict
        ``is_dark`` (bool) – True if the image is considered dark.
        ``mean_brightness`` (float) – The mean L* value.
    """
    _validate_image(image_bgr)

    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
    l_channel = lab[:, :, 0].astype(np.float64)
    mean_brightness = float(l_channel.mean())

    return {
        "is_dark": mean_brightness < dark_threshold,
        "mean_brightness": mean_brightness,
    }


# ---------------------------------------------------------------------------
# (IQ-5)  CLAHE enhancement
# ---------------------------------------------------------------------------

def apply_clahe(image_bgr: np.ndarray,
                clip_limit: float = 2.0,
                tile_grid_size: tuple = (8, 8)) -> np.ndarray:
    """Enhance a dark image using CLAHE on the L* channel only.

    Converts to LAB, applies CLAHE to the L* channel, and converts back to
    BGR so that colours are preserved (IQ-5).

    Parameters
    ----------
    image_bgr : np.ndarray
        Input image in BGR colour order.
    clip_limit : float, optional
        CLAHE contrast-limit parameter (default 2.0).
    tile_grid_size : tuple[int, int], optional
        CLAHE tile-grid size (default ``(8, 8)``).

    Returns
    -------
    np.ndarray
        Enhanced image in BGR colour order.
    """
    _validate_image(image_bgr)

    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    l_enhanced = clahe.apply(l_channel)

    lab_enhanced = cv2.merge([l_enhanced, a_channel, b_channel])
    enhanced_bgr = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)
    return enhanced_bgr


# ---------------------------------------------------------------------------
# (IQ-4 / IQ-7 / IQ-8 / IQ-9)  Main gate
# ---------------------------------------------------------------------------

def run_gate(image_bgr: np.ndarray, config: dict = None) -> dict:
    """Run the full quality gate on an image.

    Pipeline
    --------
    1. Check blur → if blurry, **reject** immediately (``rejected_blur``).
    2. Check brightness → if dark, **enhance** with CLAHE (``enhanced``).
    3. Otherwise → pass through unchanged (``ok``).

    Parameters
    ----------
    image_bgr : np.ndarray
        Input image in BGR colour order.
    config : dict, optional
        Override any key from :data:`DEFAULT_CONFIG`.

    Returns
    -------
    dict
        ``status`` (str) – One of ``"ok"``, ``"enhanced"``, or
        ``"rejected_blur"`` (IQ-7).

        ``image`` (np.ndarray) – The (possibly enhanced) BGR image to pass to
        YOLO.

        ``original_image`` (np.ndarray) – The original unmodified BGR image
        (IQ-8).

        ``blur_score`` (float) – Laplacian variance.

        ``brightness_score`` (float) – Mean L* brightness.

        ``message`` (str) – Human-readable message for the UI (IQ-9).
    """
    _validate_image(image_bgr)

    # Merge caller config with defaults (IQ-6)
    cfg = {**DEFAULT_CONFIG, **(config or {})}

    blur_threshold = cfg["blur_threshold"]
    dark_threshold = cfg["dark_threshold"]
    clahe_clip = cfg["clahe_clip_limit"]
    clahe_grid = tuple(cfg["clahe_tile_grid_size"])
    resize_width = cfg["resize_width"]

    # Keep an unmodified reference (IQ-8)
    original_image = image_bgr.copy()

    # --- Step 1: Blur check ------------------------------------------------
    blur_result = check_blur(image_bgr, threshold=blur_threshold,
                             resize_width=resize_width)
    blur_score = blur_result["laplacian_var"]

    if blur_result["is_blurry"]:
        # Also compute brightness for the report, even though we reject
        brightness_result = check_brightness(image_bgr,
                                             dark_threshold=dark_threshold)
        return {
            "status": "rejected_blur",
            "image": image_bgr,
            "original_image": original_image,
            "blur_score": blur_score,
            "brightness_score": brightness_result["mean_brightness"],
            "message": (
                f"Image rejected: too blurry (Laplacian variance "
                f"{blur_score:.1f} < threshold {blur_threshold})."
            ),
        }

    # --- Step 2: Brightness check ------------------------------------------
    brightness_result = check_brightness(image_bgr,
                                         dark_threshold=dark_threshold)
    brightness_score = brightness_result["mean_brightness"]

    if brightness_result["is_dark"]:
        enhanced = apply_clahe(image_bgr, clip_limit=clahe_clip,
                               tile_grid_size=clahe_grid)
        return {
            "status": "enhanced",
            "image": enhanced,
            "original_image": original_image,
            "blur_score": blur_score,
            "brightness_score": brightness_score,
            "message": (
                f"Image was dark (mean L* {brightness_score:.1f} < "
                f"{dark_threshold}); CLAHE enhancement applied."
            ),
        }

    # --- Step 3: All good (IQ-4) -------------------------------------------
    return {
        "status": "ok",
        "image": image_bgr,
        "original_image": original_image,
        "blur_score": blur_score,
        "brightness_score": brightness_score,
        "message": (
            f"Image quality OK (blur score {blur_score:.1f}, "
            f"brightness {brightness_score:.1f})."
        ),
    }

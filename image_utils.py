import io
import logging
from PIL import Image

logger = logging.getLogger(__name__)

# Max long-edge size before sending to API (avoids huge base64 payloads)
MAX_IMAGE_SIDE = 2048


def prepare_image(image_path: str) -> str:
    """
    If the image is very large, resize it in-place (overwrite temp file) so
    base64 payload stays reasonable. Returns the (possibly same) path.
    """
    try:
        with Image.open(image_path) as img:
            w, h = img.size
            if max(w, h) <= MAX_IMAGE_SIDE:
                return image_path  # nothing to do

            # Downscale keeping aspect ratio
            scale = MAX_IMAGE_SIDE / max(w, h)
            new_size = (int(w * scale), int(h * scale))
            img = img.convert("RGB")
            img = img.resize(new_size, Image.LANCZOS)
            img.save(image_path, format="JPEG", quality=88)
            logger.info("Resized image from %dx%d to %dx%d", w, h, *new_size)
    except Exception as e:
        logger.warning("Could not resize image: %s", e)

    return image_path


def is_valid_bbox(bbox: dict | None) -> bool:
    """
    Return True if the bbox is valid and covers a meaningful sub-region
    (not the whole image, not tiny).
    """
    if not bbox:
        return False

    x_min = bbox.get("x_min", 0.0)
    y_min = bbox.get("y_min", 0.0)
    x_max = bbox.get("x_max", 1.0)
    y_max = bbox.get("y_max", 1.0)

    w = x_max - x_min
    h = y_max - y_min

    if w < 0.03 or h < 0.03:          # too small
        return False
    if w > 0.97 and h > 0.97:          # whole image → skip crop
        return False

    return True


def crop_region(image_path: str, bbox: dict) -> io.BytesIO | None:
    """
    Crop the image to the normalized bounding box region.
    Returns a BytesIO JPEG image or None on failure.
    """
    try:
        with Image.open(image_path) as img:
            if img.mode not in ("RGB", "L"):
                img = img.convert("RGB")

            w, h = img.size
            left   = max(0, int(bbox["x_min"] * w))
            top    = max(0, int(bbox["y_min"] * h))
            right  = min(w, int(bbox["x_max"] * w))
            bottom = min(h, int(bbox["y_max"] * h))

            if right - left < 20 or bottom - top < 20:
                logger.warning("Crop too small: %dx%d", right - left, bottom - top)
                return None

            cropped = img.crop((left, top, right, bottom))
            buf = io.BytesIO()
            cropped.save(buf, format="JPEG", quality=88)
            buf.seek(0)
            return buf

    except Exception as e:
        logger.error("crop_region failed: %s", e)
        return None

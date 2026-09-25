import cv2
import numpy as np

# OpenCV stores hue on a 0-179 scale, so red sits at both ends of the range.
HUE_MAX = 179
HUE_TOLERANCE = 10

PRESET_COLORS = {
    "red": (255, 0, 0),
    "orange": (255, 128, 0),
    "yellow": (255, 255, 0),
    "green": (0, 255, 0),
    "cyan": (0, 255, 255),
    "blue": (0, 0, 255),
    "purple": (128, 0, 255),
}


def parse_color(value):
    """Accept a preset name ("yellow") or an "R,G,B" string ("255,255,0")."""
    if value.lower() in PRESET_COLORS:
        return PRESET_COLORS[value.lower()]
    parts = [int(p) for p in value.split(",")]
    if len(parts) != 3 or not all(0 <= p <= 255 for p in parts):
        raise ValueError(f"expected a preset name or R,G,B in 0-255, got {value!r}")
    return tuple(parts)


def get_hsv_ranges(rgb):
    """Return a list of (lower, upper) HSV bounds that cover the given RGB color.

    When the hue window crosses 0 or 179 (reds), it is split into two ranges
    so both ends of the hue circle are matched.
    """
    pixel = np.uint8([[rgb]])
    hue = int(cv2.cvtColor(pixel, cv2.COLOR_RGB2HSV)[0][0][0])

    low, high = hue - HUE_TOLERANCE, hue + HUE_TOLERANCE
    if low < 0:
        spans = [(0, high), (HUE_MAX + 1 + low, HUE_MAX)]
    elif high > HUE_MAX:
        spans = [(low, HUE_MAX), (0, high - HUE_MAX - 1)]
    else:
        spans = [(low, high)]

    return [
        (np.array([lo, 100, 100], dtype=np.uint8), np.array([hi, 255, 255], dtype=np.uint8))
        for lo, hi in spans
    ]


def color_mask(hsv_frame, ranges):
    mask = np.zeros(hsv_frame.shape[:2], dtype=np.uint8)
    for lower, upper in ranges:
        mask |= cv2.inRange(hsv_frame, lower, upper)
    return mask

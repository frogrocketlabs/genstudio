from typing import Union

import numpy as np

from genstudio.layout import Hiccup, JSExpr, JSRef, LayoutItem


def bitmap(
    pixels: Union[list, np.ndarray, JSExpr],
    width: int | None = None,
    height: int | None = None,
) -> LayoutItem:
    """
    Renders raw pixel data from an array.

    Args:
        pixels: Image data in one of these formats:
               - Raw pixel data in RGB or RGBA format (flat array of bytes)
               - 2D numpy array (grayscale image)
               - 3D numpy array with shape (height, width, channels)
        width: Width of the image in pixels. Required for flat arrays,
               inferred from array shape for numpy arrays.
        height: Height of the image in pixels. Required for flat arrays,
                inferred from array shape for numpy arrays.

    Returns:
        A PlotSpec object representing the bitmap mark.

    Example:
        >>> # Create 2x2 red square from raw bytes
        >>> pixels = bytes([255,0,0] * 4) # RGB format
        >>> bitmap(pixels, width=2, height=2)

        >>> # Create from numpy array
        >>> img = np.zeros((100, 100, 3)) # 100x100 black image
        >>> bitmap(img)
    """
    if isinstance(pixels, np.ndarray):
        if pixels.ndim == 2:
            # Convert grayscale to RGB
            pixels = np.stack([pixels] * 3, axis=-1)

        if pixels.ndim == 3:
            height, width = pixels.shape[:2]
            # Flatten array to 1D array in row-major order
            pixels = pixels.astype(np.uint8).flatten()
    return Hiccup(
        [JSRef("Bitmap"), {"pixels": pixels, "width": width, "height": height}]
    )

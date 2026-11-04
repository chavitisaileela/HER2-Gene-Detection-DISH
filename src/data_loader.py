"""
data_loader.py
Module for loading and validating brightfield DISH breast histopathology images.
Handles standard image formats (PNG, JPG, BMP) and high-resolution TIFF images.
"""

import os
import cv2
import numpy as np

try:
    import tifffile
    HAS_TIFFFILE = True
except ImportError:
    HAS_TIFFFILE = False


class DISHDataLoader:
    """Loads and inspects brightfield microscopy DISH images."""

    @staticmethod
    def load_image(file_path: str) -> np.ndarray:
        """
        Loads an image from disk and returns an 8-bit BGR numpy array.
        
        Args:
            file_path: Path to the image file (.tif, .tiff, .png, .jpg).
            
        Returns:
            np.ndarray: BGR image array with shape (Height, Width, 3).
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Image file not found at: {file_path}")

        ext = os.path.splitext(file_path)[1].lower()

        # Handle TIFF using tifffile for robust multi-strip/bigtiff support
        if ext in [".tif", ".tiff"] and HAS_TIFFFILE:
            try:
                img_data = tifffile.imread(file_path)
                # If grayscale, convert to 3 channels
                if img_data.ndim == 2:
                    img_bgr = cv2.cvtColor(img_data, cv2.COLOR_GRAY2BGR)
                elif img_data.ndim == 3:
                    # Check channels: if RGBA, convert to BGR; if RGB, convert to BGR
                    if img_data.shape[2] == 4:
                        img_bgr = cv2.cvtColor(img_data, cv2.COLOR_RGBA2BGR)
                    elif img_data.shape[2] == 3:
                        img_bgr = cv2.cvtColor(img_data, cv2.COLOR_RGB2BGR)
                    else:
                        img_bgr = img_data[:, :, :3]
                else:
                    raise ValueError(f"Unsupported TIFF dimensionality: {img_data.ndim}")
                
                # Ensure uint8
                if img_bgr.dtype != np.uint8:
                    img_bgr = cv2.normalize(img_bgr, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                return img_bgr
            except Exception as e:
                print(f"[Warning] tifffile read failed ({e}), falling back to cv2.imread.")

        # Standard OpenCV load
        img = cv2.imread(file_path, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError(f"Failed to read image from {file_path}. File may be corrupted or unsupported.")
            
        return img

    @staticmethod
    def inspect_image(img: np.ndarray) -> dict:
        """
        Inspects image attributes (dimensions, channels, mean intensity).
        
        Returns:
            dict containing width, height, channels, mean brightness.
        """
        h, w = img.shape[:2]
        c = img.shape[2] if img.ndim > 2 else 1
        mean_val = float(np.mean(img))
        
        return {
            "height": h,
            "width": w,
            "channels": c,
            "mean_intensity": round(mean_val, 2),
            "is_standard_dish_size": (h == 1500 and w == 1500)
        }

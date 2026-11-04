"""
signal_detector.py
HER2 (Black) and CEP17 (Red) Chromogenic Signal Detection Module.
Identifies and quantifies intracellular chromogenic signal dots within each segmented
nucleus according to the dual-color detection methodology in Micaraseth et al. (2026).
"""

from typing import List, Dict, Any, Tuple
import cv2
import numpy as np


class SignalDetector:
    """
    Detects red (CEP17 centromere) and black (HER2 gene) signal dots
    inside the boundaries of individual cell nuclei.
    """

    def __init__(
        self,
        min_dot_radius: int = 1,
        max_dot_radius: int = 10,
        black_intensity_thresh: int = 120,
        red_ratio_thresh: float = 1.25,
    ):
        self.min_dot_radius = min_dot_radius
        self.max_dot_radius = max_dot_radius
        self.black_intensity_thresh = black_intensity_thresh
        self.red_ratio_thresh = red_ratio_thresh

    def detect_signals_in_crop(
        self, cell_crop_bgr: np.ndarray, cell_mask: np.ndarray
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Detects HER2 (black) and CEP17 (red) dots inside a single cropped nucleus.

        Args:
            cell_crop_bgr: Cropped BGR image of the nucleus (H, W, 3).
            cell_mask: Binary mask of the nucleus boundary (H, W), values in {0, 255}.

        Returns:
            Tuple of:
                - her2_signals: List of dicts [{"local_pos": (x, y), "radius": r, "confidence": c}]
                - cep17_signals: List of dicts [{"local_pos": (x, y), "radius": r, "confidence": c}]
        """
        ch, cw = cell_crop_bgr.shape[:2]
        if ch < 5 or cw < 5:
            return [], []

        b = cell_crop_bgr[:, :, 0].astype(np.float32)
        g = cell_crop_bgr[:, :, 1].astype(np.float32)
        r = cell_crop_bgr[:, :, 2].astype(np.float32)

        # -------------------------------------------------------------
        # 1. CEP17 Detection (Red Signals)
        # -------------------------------------------------------------
        # Red chromogen (Fast Red) has prominent Red dominance over Green and Blue
        red_dominance = r - np.maximum(g, b)
        red_score = np.where((cell_mask > 0) & (r > 90) & (r / (g + 1e-4) > self.red_ratio_thresh), red_dominance, 0)
        red_score = np.clip(red_score, 0, 255).astype(np.uint8)

        # Morphological White Top-Hat to isolate small bright red spots
        kernel_red = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        red_tophat = cv2.morphologyEx(red_score, cv2.MORPH_TOPHAT, kernel_red)

        # Threshold to get candidate red peaks
        _, red_bin = cv2.threshold(red_tophat, 15, 255, cv2.THRESH_BINARY)
        # Ensure mask confinement
        red_bin = cv2.bitwise_and(red_bin, cell_mask)

        cep17_signals = []
        red_contours, _ = cv2.findContours(red_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in red_contours:
            area = cv2.contourArea(cnt)
            perimeter = cv2.arcLength(cnt, True)
            if area > (np.pi * (self.max_dot_radius ** 2) * 1.5) or area < 2:
                continue

            (cx, cy), radius = cv2.minEnclosingCircle(cnt)
            cx, cy = int(cx), int(cy)
            radius = max(2, int(radius))

            if 0 <= cx < cw and 0 <= cy < ch and cell_mask[cy, cx] > 0:
                cep17_signals.append({
                    "local_pos": (cx, cy),
                    "radius": radius,
                    "color": "red",
                    "confidence": round(float(min(1.0, (area + 2) / 15.0)), 2),
                })

        # -------------------------------------------------------------
        # 2. HER2 Detection (Black Signals)
        # -------------------------------------------------------------
        # Black chromogen (silver acetate precipitation) has low intensity in all channels:
        # We calculate darkness and isolate local intensity minima
        max_rgb = np.maximum(np.maximum(r, g), b)
        darkness = np.clip(255.0 - max_rgb, 0, 255).astype(np.uint8)

        # Black Top-Hat on grayscale to extract dots distinctly darker than background nuclear chromatin
        gray = cv2.cvtColor(cell_crop_bgr, cv2.COLOR_BGR2GRAY)
        kernel_black = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        black_tophat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel_black)

        # Combine darkness and local dip
        combined_black = np.where((cell_mask > 0) & (max_rgb < self.black_intensity_thresh), black_tophat.astype(np.float32) * 1.5, 0)
        combined_black = np.clip(combined_black, 0, 255).astype(np.uint8)

        _, black_bin = cv2.threshold(combined_black, 18, 255, cv2.THRESH_BINARY)
        black_bin = cv2.bitwise_and(black_bin, cell_mask)

        # Also remove any pixel already identified as a red signal
        for red_sig in cep17_signals:
            rx, ry = red_sig["local_pos"]
            rr = red_sig["radius"] + 2
            cv2.circle(black_bin, (rx, ry), rr, 0, -1)

        her2_signals = []
        black_contours, _ = cv2.findContours(black_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in black_contours:
            area = cv2.contourArea(cnt)
            if area > (np.pi * (self.max_dot_radius ** 2) * 2.0) or area < 2:
                continue

            (cx, cy), radius = cv2.minEnclosingCircle(cnt)
            cx, cy = int(cx), int(cy)
            radius = max(2, int(radius))

            if 0 <= cx < cw and 0 <= cy < ch and cell_mask[cy, cx] > 0:
                her2_signals.append({
                    "local_pos": (cx, cy),
                    "radius": radius,
                    "color": "black",
                    "confidence": round(float(min(1.0, (area + 2) / 15.0)), 2),
                })

        return her2_signals, cep17_signals

    def quantify_slide(
        self, img_bgr: np.ndarray, cells: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Quantifies HER2 and CEP17 signals across all segmented cells in the image.

        Args:
            img_bgr: Full image array (H, W, 3).
            cells: List of cell dicts from CellSegmenter.

        Returns:
            Updated cells list with:
                - 'her2_signals': List with global_pos & local_pos
                - 'cep17_signals': List with global_pos & local_pos
                - 'her2_count': int
                - 'cep17_count': int
                - 'cell_ratio': float or None
        """
        for cell in cells:
            x, y, w, h = cell["bbox"]
            cell_crop = img_bgr[y : y + h, x : x + w]
            cell_mask = cell["mask"]

            her2_local, cep17_local = self.detect_signals_in_crop(cell_crop, cell_mask)

            # Map to global slide coordinates
            her2_global = []
            for sig in her2_local:
                lx, ly = sig["local_pos"]
                her2_global.append({
                    "global_pos": (x + lx, y + ly),
                    "local_pos": (lx, ly),
                    "radius": sig["radius"],
                    "confidence": sig["confidence"],
                })

            cep17_global = []
            for sig in cep17_local:
                lx, ly = sig["local_pos"]
                cep17_global.append({
                    "global_pos": (x + lx, y + ly),
                    "local_pos": (lx, ly),
                    "radius": sig["radius"],
                    "confidence": sig["confidence"],
                })

            cell["her2_signals"] = her2_global
            cell["cep17_signals"] = cep17_global
            cell["her2_count"] = len(her2_global)
            cell["cep17_count"] = len(cep17_global)
            cell["cell_ratio"] = (
                round(len(her2_global) / len(cep17_global), 2)
                if len(cep17_global) > 0
                else None
            )

        return cells

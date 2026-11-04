"""
cell_segmentation.py
Cell Nucleus Segmentation and Top-Candidate Selection Module.
Implements the clinical cell selection protocol from Micaraseth et al. (2026),
filtering out debris, border-clipped nuclei, and poorly stained regions, then
ranking cells by morphological clarity to select the Top-N diagnostically relevant nuclei.
"""

from typing import List, Dict, Any, Tuple
import cv2
import numpy as np


class CellSegmenter:
    """
    Segments individual nuclei from brightfield DISH histopathology images,
    applies pathologist inclusion criteria (non-overlapping, clear morphology),
    and ranks them by confidence score.
    """

    def __init__(
        self,
        min_area: int = 600,
        max_area: int = 15000,
        min_circularity: float = 0.40,
        max_aspect_ratio: float = 2.4,
        border_margin: int = 20,
    ):
        self.min_area = min_area
        self.max_area = max_area
        self.min_circularity = min_circularity
        self.max_aspect_ratio = max_aspect_ratio
        self.border_margin = border_margin

    def segment_nuclei(self, img_bgr: np.ndarray, top_k: int = 100) -> List[Dict[str, Any]]:
        """
        Segments nuclei from a DISH image and returns the top-ranked cells.

        Args:
            img_bgr: Input BGR image (1500x1500 or arbitrary size).
            top_k: Number of highest-confidence cells to retain (default: 100, as in paper Table 8).

        Returns:
            List of dictionaries, each containing:
                - cell_id: Unique integer index
                - bbox: (x, y, w, h)
                - center: (cx, cy)
                - area: int
                - contour: contour points
                - mask: binary mask (same size as bbox crop)
                - confidence: float score in [0, 1]
        """
        h, w = img_bgr.shape[:2]

        # 1. Color deconvolution / hematoxylin contrast enhancement
        # Hematoxylin stains nucleic acids pale blue/purple.
        # Background is near white (high R, G, B).
        # We calculate an inverted optical contrast map emphasizing hematoxylin:
        b_chan = img_bgr[:, :, 0].astype(np.float32)
        g_chan = img_bgr[:, :, 1].astype(np.float32)
        r_chan = img_bgr[:, :, 2].astype(np.float32)

        # Contrast index: Nuclei have darker values and relatively stronger blue component than red
        nucleus_signal = (255.0 - (0.35 * b_chan + 0.35 * g_chan + 0.30 * r_chan))
        nucleus_signal = np.clip(nucleus_signal, 0, 255).astype(np.uint8)

        # Smooth to reduce intracellular signal noise during cell boundary detection
        blurred = cv2.GaussianBlur(nucleus_signal, (11, 11), 2.0)

        # 2. Otsu thresholding + Adaptive Morphological separation
        _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Morphological opening to detach slightly touching nuclei
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        opened = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)

        # 3. Find candidate contours
        contours, _ = cv2.findContours(opened, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        candidate_cells = []

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < self.min_area or area > self.max_area:
                continue

            x, y, cw, ch = cv2.boundingRect(cnt)

            # Exclusion Criterion 1: Exclude cells touching image borders
            if (
                x <= self.border_margin
                or y <= self.border_margin
                or (x + cw) >= (w - self.border_margin)
                or (y + ch) >= (h - self.border_margin)
            ):
                continue

            # Exclusion Criterion 2: Aspect ratio check
            aspect_ratio = max(cw, ch) / (min(cw, ch) + 1e-5)
            if aspect_ratio > self.max_aspect_ratio:
                continue

            # Exclusion Criterion 3: Circularity check (4 * pi * Area / Perimeter^2)
            perimeter = cv2.arcLength(cnt, True)
            if perimeter <= 0:
                continue
            circularity = (4.0 * np.pi * area) / (perimeter * perimeter)
            if circularity < self.min_circularity:
                continue

            # Compute centroid
            moments = cv2.moments(cnt)
            if moments["m00"] != 0:
                cx = int(moments["m10"] / moments["m00"])
                cy = int(moments["m01"] / moments["m00"])
            else:
                cx, cy = x + cw // 2, y + ch // 2

            # Morphological Confidence Score:
            # Rewards optimal nuclear diameter (~60-100px) and roundness, penalizes boundary irregularity
            ideal_area = 4500.0
            area_score = np.exp(-0.5 * ((area - ideal_area) / 3000.0) ** 2)
            circularity_score = min(1.0, circularity)
            confidence = round(float(0.6 * circularity_score + 0.4 * area_score), 4)

            # Local cell mask relative to bbox
            cell_mask = np.zeros((ch, cw), dtype=np.uint8)
            shifted_cnt = cnt - np.array([x, y])
            cv2.drawContours(cell_mask, [shifted_cnt], -1, 255, -1)

            candidate_cells.append({
                "bbox": (x, y, cw, ch),
                "center": (cx, cy),
                "area": int(area),
                "contour": cnt,
                "mask": cell_mask,
                "confidence": confidence,
            })

        # Rank cells by confidence descending
        candidate_cells.sort(key=lambda c: c["confidence"], reverse=True)

        # Select Top-K cells
        selected_cells = candidate_cells[:top_k]

        # Assign 1-based sequential IDs
        for idx, cell in enumerate(selected_cells, 1):
            cell["cell_id"] = idx

        return selected_cells

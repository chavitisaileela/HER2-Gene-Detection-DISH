"""
visualizer.py
Visualization and Diagnostic Report Generation Module.
Recreates the visualizations from Micaraseth et al. (Intelligence-Based Medicine 2026):
- Figure 4: Annotated slide with green bounding boxes and per-cell signal counts, plus cropped nucleus gallery.
- Figure 7: 4-stage pipeline visualization:
    (a) Nucleus segmentation
    (b) HER2 signal annotation
    (c) CEP17 signal annotation
    (d) Final merged detection output
"""

import os
from typing import List, Dict, Any
import cv2
import numpy as np


class DISHVisualizer:
    """Generates visual overlays and clinical reports for DISH analysis."""

    @staticmethod
    def draw_annotated_slide(
        img_bgr: np.ndarray, cells: List[Dict[str, Any]], draw_signals: bool = True
    ) -> np.ndarray:
        """
        Draws green bounding boxes around detected cells and annotates them
        with HER2/CEP17 counts (matching Figure 4 of the paper).
        """
        annotated = img_bgr.copy()

        for cell in cells:
            x, y, w, h = cell["bbox"]
            her2_cnt = cell.get("her2_count", 0)
            cep17_cnt = cell.get("cep17_count", 0)

            # Draw green bounding box for selected cell (as shown in Figure 4)
            cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 200, 0), 2)

            # Draw small label above cell with counts: H: <count>, C: <count>
            label = f"H:{her2_cnt} C:{cep17_cnt}"
            (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            cv2.rectangle(annotated, (x, max(0, y - lh - 6)), (x + lw + 4, y), (0, 180, 0), -1)
            cv2.putText(
                annotated,
                label,
                (x + 2, max(lh + 2, y - 3)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

            # Draw circles around detected intracellular signals if requested
            if draw_signals:
                for sig in cell.get("her2_signals", []):
                    gx, gy = sig["global_pos"]
                    cv2.circle(annotated, (gx, gy), sig["radius"] + 2, (255, 0, 0), 1)  # Blue circle around black HER2
                for sig in cell.get("cep17_signals", []):
                    gx, gy = sig["global_pos"]
                    cv2.circle(annotated, (gx, gy), sig["radius"] + 2, (0, 0, 255), 1)  # Red circle around red CEP17

        return annotated

    @staticmethod
    def create_cropped_cell_gallery(
        img_bgr: np.ndarray, cells: List[Dict[str, Any]], max_cells: int = 15, thumb_size: int = 120
    ) -> np.ndarray:
        """
        Generates a horizontal strip gallery of cropped individual cells
        with their detected signals (replicating the 'Crop Images' strip in Figure 4).
        """
        if not cells:
            return np.ones((thumb_size, thumb_size, 3), dtype=np.uint8) * 255

        display_cells = cells[:max_cells]
        thumbs = []

        for cell in display_cells:
            x, y, w, h = cell["bbox"]
            crop = img_bgr[y : y + h, x : x + w].copy()

            # Draw internal signal markers
            for sig in cell.get("her2_signals", []):
                lx, ly = sig["local_pos"]
                cv2.circle(crop, (lx, ly), sig["radius"] + 2, (255, 0, 0), 1)
            for sig in cell.get("cep17_signals", []):
                lx, ly = sig["local_pos"]
                cv2.circle(crop, (lx, ly), sig["radius"] + 2, (0, 0, 255), 1)

            # Resize with padding to fixed thumb_size
            padded = np.ones((thumb_size, thumb_size, 3), dtype=np.uint8) * 240
            scale = min((thumb_size - 24) / max(w, 1), (thumb_size - 24) / max(h, 1))
            nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
            resized = cv2.resize(crop, (nw, nh), interpolation=cv2.INTER_AREA)

            ox = (thumb_size - nw) // 2
            oy = (thumb_size - nh - 18) // 2
            padded[oy : oy + nh, ox : ox + nw] = resized

            # Border and tag
            cv2.rectangle(padded, (0, 0), (thumb_size - 1, thumb_size - 1), (180, 180, 180), 1)
            tag = f"C#{cell['cell_id']} H:{cell.get('her2_count', 0)}"
            cv2.putText(padded, tag, (6, thumb_size - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (30, 30, 30), 1)
            thumbs.append(padded)

        # Concatenate horizontally
        gallery = np.hstack(thumbs)
        return gallery

    @staticmethod
    def create_figure7_stages(
        img_bgr: np.ndarray, cells: List[Dict[str, Any]], crop_box: tuple = None
    ) -> np.ndarray:
        """
        Creates a 4-panel comparison matching Figure 7 in the paper:
        (a) Segmentation results
        (b) HER2 annotation (black signals)
        (c) CEP17 annotation (red signals)
        (d) Final detection output
        """
        h, w = img_bgr.shape[:2]
        if crop_box is None:
            # Focus on a representative 500x500 central ROI
            cw, ch = min(550, w), min(550, h)
            cx, cy = w // 2, h // 2
            x0 = max(0, cx - cw // 2)
            y0 = max(0, cy - ch // 2)
            crop_box = (x0, y0, cw, ch)

        x0, y0, cw, ch = crop_box
        base_roi = img_bgr[y0 : y0 + ch, x0 : x0 + cw].copy()

        # Panel A: Nucleus Segmentation (green contours / semi-transparent masks)
        panel_a = base_roi.copy()
        mask_overlay = base_roi.copy()
        for cell in cells:
            cx, cy = cell["center"]
            if x0 <= cx < x0 + cw and y0 <= cy < y0 + ch:
                shifted_cnt = cell["contour"] - np.array([x0, y0])
                cv2.drawContours(mask_overlay, [shifted_cnt], -1, (0, 230, 0), -1)
                cv2.drawContours(panel_a, [shifted_cnt], -1, (0, 180, 0), 2)
        cv2.addWeighted(mask_overlay, 0.35, panel_a, 0.65, 0, panel_a)
        cv2.putText(panel_a, "(a) Nucleus Segmentation", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 100, 0), 2)

        # Panel B: HER2 Black Signal Annotation
        panel_b = base_roi.copy()
        for cell in cells:
            for sig in cell.get("her2_signals", []):
                gx, gy = sig["global_pos"]
                if x0 <= gx < x0 + cw and y0 <= gy < y0 + ch:
                    lx, ly = gx - x0, gy - y0
                    cv2.circle(panel_b, (lx, ly), sig["radius"] + 3, (255, 50, 0), 2)
        cv2.putText(panel_b, "(b) HER2 Annotation (Black)", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (180, 0, 0), 2)

        # Panel C: CEP17 Red Signal Annotation
        panel_c = base_roi.copy()
        for cell in cells:
            for sig in cell.get("cep17_signals", []):
                gx, gy = sig["global_pos"]
                if x0 <= gx < x0 + cw and y0 <= gy < y0 + ch:
                    lx, ly = gx - x0, gy - y0
                    cv2.circle(panel_c, (lx, ly), sig["radius"] + 3, (0, 0, 255), 2)
        cv2.putText(panel_c, "(c) CEP17 Annotation (Red)", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 200), 2)

        # Panel D: Merged Detection Output
        panel_d = base_roi.copy()
        for cell in cells:
            x, y, bw, bh = cell["bbox"]
            if x0 <= (x + bw // 2) < x0 + cw and y0 <= (y + bh // 2) < y0 + ch:
                cv2.rectangle(panel_d, (x - x0, y - y0), (x - x0 + bw, y - y0 + bh), (0, 200, 0), 1)
                for sig in cell.get("her2_signals", []):
                    gx, gy = sig["global_pos"]
                    cv2.circle(panel_d, (gx - x0, gy - y0), sig["radius"] + 2, (255, 0, 0), 1)
                for sig in cell.get("cep17_signals", []):
                    gx, gy = sig["global_pos"]
                    cv2.circle(panel_d, (gx - x0, gy - y0), sig["radius"] + 2, (0, 0, 255), 1)
        cv2.putText(panel_d, "(d) Final Merged Detection", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 120, 0), 2)

        # Combine into 2x2 grid
        top_row = np.hstack([panel_a, panel_b])
        bot_row = np.hstack([panel_c, panel_d])
        figure_7 = np.vstack([top_row, bot_row])
        return figure_7

"""
ratio_analyzer.py
Clinical Interpretation and Ratio Computation Module.
Applies official ASCO/CAP and Table 2 clinical criteria from Micaraseth et al. (2026):
- HER2/CEP17 Ratio
- Average HER2 Copy Number per Cell
- Amplification Status Classification (Positive, Negative, or Equivocal/Borderline)
"""

from typing import List, Dict, Any
import numpy as np


class RatioAnalyzer:
    """
    Computes clinical diagnostic metrics for HER2 gene amplification in DISH slides.
    """

    @staticmethod
    def analyze(cells: List[Dict[str, Any]], patient_id: str = "Unknown") -> Dict[str, Any]:
        """
        Analyzes the quantified cells and produces clinical diagnostic conclusions.

        Args:
            cells: List of quantified cell dictionaries with 'her2_count' and 'cep17_count'.
            patient_id: Identifier of the patient case (e.g. 'c-001').

        Returns:
            Dict containing:
                - patient_id: str
                - total_cells: int
                - total_her2: int
                - total_cep17: int
                - her2_cep17_ratio: float
                - avg_her2_per_cell: float
                - avg_cep17_per_cell: float
                - status: 'Positive', 'Negative', or 'Equivocal (Borderline)'
                - clinical_interpretation: str
                - cell_breakdown: List of per-cell metrics
        """
        total_cells = len(cells)
        if total_cells == 0:
            return {
                "patient_id": patient_id,
                "total_cells": 0,
                "total_her2": 0,
                "total_cep17": 0,
                "her2_cep17_ratio": 0.0,
                "avg_her2_per_cell": 0.0,
                "avg_cep17_per_cell": 0.0,
                "status": "Inconclusive",
                "clinical_interpretation": "No diagnostically acceptable nuclei detected.",
                "cell_breakdown": [],
            }

        total_her2 = sum(c.get("her2_count", 0) for c in cells)
        total_cep17 = sum(c.get("cep17_count", 0) for c in cells)

        # HER2/CEP17 Ratio calculation
        if total_cep17 > 0:
            ratio = round(float(total_her2) / float(total_cep17), 2)
        else:
            ratio = float("inf")

        avg_her2 = round(float(total_her2) / float(total_cells), 2)
        avg_cep17 = round(float(total_cep17) / float(total_cells), 2)

        # Clinical Classification based on Table 2 & ASCO/CAP guidelines:
        # Positive: Ratio >= 2.0 AND Avg HER2 >= 4.0
        # Negative: Ratio < 2.0 AND Avg HER2 < 4.0
        # Equivocal: Borderline criteria (requires expanded review)
        if ratio >= 2.0 and avg_her2 >= 4.0:
            status = "POSITIVE"
            interpretation = "HER2 gene amplification detected. Eligible for HER2-targeted therapy (e.g., Trastuzumab)."
        elif ratio < 2.0 and avg_her2 < 4.0:
            status = "NEGATIVE"
            interpretation = "No significant HER2 gene amplification detected."
        elif ratio >= 2.0 and avg_her2 < 4.0:
            status = "EQUIVOCAL (Ratio >= 2.0, Low Avg Copy Number)"
            interpretation = "Borderline result: Ratio is elevated but average HER2 count is < 4.0. Additional review recommended."
        else: # ratio < 2.0 and avg_her2 >= 4.0
            status = "EQUIVOCAL (Ratio < 2.0, High HER2 Copy Number)"
            interpretation = "Borderline result: Average HER2 copy number is elevated (>= 4.0) with co-amplified CEP17."

        # Per-cell breakdown summary
        cell_breakdown = []
        for c in cells:
            cell_breakdown.append({
                "cell_id": c["cell_id"],
                "center": c["center"],
                "her2_count": c.get("her2_count", 0),
                "cep17_count": c.get("cep17_count", 0),
                "ratio": c.get("cell_ratio"),
                "confidence": c.get("confidence", 0.0),
            })

        return {
            "patient_id": patient_id,
            "total_cells": total_cells,
            "total_her2": total_her2,
            "total_cep17": total_cep17,
            "her2_cep17_ratio": ratio,
            "avg_her2_per_cell": avg_her2,
            "avg_cep17_per_cell": avg_cep17,
            "status": status,
            "clinical_interpretation": interpretation,
            "cell_breakdown": cell_breakdown,
        }

"""
main.py
End-to-End Automated HER2 Gene Copy Number Detection System.
Implements the deep learning / image processing pipeline described in:
Micaraseth et al., 'Automated detection of HER2 gene copy number in breast cancer
using deep learning techniques', Intelligence-Based Medicine 13 (2026) 100333.
"""

import os
import sys
import json
import argparse
import cv2

# Add src to path
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(current_dir, "src")
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from data_loader import DISHDataLoader
from cell_segmentation import CellSegmenter
from signal_detector import SignalDetector
from ratio_analyzer import RatioAnalyzer
from visualizer import DISHVisualizer


def run_pipeline(
    image_path: str,
    patient_id: str = "c-001",
    top_cells: int = 100,
    output_dir: str = "outputs",
) -> dict:
    """
    Executes the full automated DISH analysis pipeline:
    1. Loads and inspects DISH brightfield image.
    2. Segments nuclei and filters top diagnostic cells.
    3. Quantifies black HER2 and red CEP17 chromogenic signals.
    4. Computes HER2/CEP17 ratio and clinical status.
    5. Generates Figure 4 & Figure 7 visual outputs and reports.
    """
    print("=" * 70)
    print("      HER2 GENE COPY NUMBER AUTOMATED QUANTIFICATION SYSTEM      ")
    print("     Based on Micaraseth et al. (Intelligence-Based Medicine 2026)   ")
    print("=" * 70)
    print(f"[*] Patient ID     : {patient_id}")
    print(f"[*] Input Image    : {image_path}")

    # 1. Load Image
    print("\n[Step 1/5] Loading and inspecting image...")
    img = DISHDataLoader.load_image(image_path)
    metadata = DISHDataLoader.inspect_image(img)
    print(f"    - Dimensions   : {metadata['width']} x {metadata['height']} px")
    print(f"    - Channels     : {metadata['channels']}")
    print(f"    - Mean Intensity: {metadata['mean_intensity']}")

    # 2. Cell Segmentation (YOLOv11-seg protocol)
    print(f"\n[Step 2/5] Segmenting nuclei and selecting Top-{top_cells} cells...")
    segmenter = CellSegmenter()
    cells = segmenter.segment_nuclei(img, top_k=top_cells)
    print(f"    - Valid Diagnostically Relevant Nuclei Found: {len(cells)}")

    # 3. Dual-Color Signal Quantification (HER2 & CEP17)
    print("\n[Step 3/5] Quantifying intracellular chromogenic signals...")
    detector = SignalDetector()
    cells = detector.quantify_slide(img, cells)
    print(f"    - Finished signal detection across {len(cells)} cells.")

    # 4. Clinical Ratio and Guideline Evaluation
    print("\n[Step 4/5] Computing HER2/CEP17 ratio and clinical diagnosis...")
    report = RatioAnalyzer.analyze(cells, patient_id=patient_id)

    # 5. Visual Output Generation
    print("\n[Step 5/5] Generating visual diagnostic outputs (Figures 4 & 7)...")
    os.makedirs(output_dir, exist_ok=True)

    # (a) Annotated full slide (Figure 4)
    annotated_img = DISHVisualizer.draw_annotated_slide(img, cells)
    annotated_path = os.path.join(output_dir, f"{patient_id}_annotated_slide.png")
    cv2.imwrite(annotated_path, annotated_img)
    print(f"    - Saved Annotated Slide   : {annotated_path}")

    # (b) Individual cropped cell gallery (Figure 4 Top)
    gallery_img = DISHVisualizer.create_cropped_cell_gallery(img, cells, max_cells=12)
    gallery_path = os.path.join(output_dir, f"{patient_id}_cell_gallery.png")
    cv2.imwrite(gallery_path, gallery_img)
    print(f"    - Saved Cropped Gallery   : {gallery_path}")

    # (c) 4-Stage Figure 7 pipeline visual
    figure7_img = DISHVisualizer.create_figure7_stages(img, cells)
    fig7_path = os.path.join(output_dir, f"{patient_id}_figure7_stages.png")
    cv2.imwrite(fig7_path, figure7_img)
    print(f"    - Saved Figure 7 Stages   : {fig7_path}")

    # (d) Save JSON clinical report
    json_path = os.path.join(output_dir, f"{patient_id}_clinical_report.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=4)
    print(f"    - Saved Clinical JSON     : {json_path}")

    # Terminal summary display
    print("\n" + "=" * 70)
    print("                    CLINICAL DIAGNOSTIC REPORT                   ")
    print("=" * 70)
    print(f" Patient ID            : {report['patient_id']}")
    print(f" Total Analyzed Cells  : {report['total_cells']}")
    print(f" Sum HER2 (Black Dots) : {report['total_her2']}")
    print(f" Sum CEP17 (Red Dots)  : {report['total_cep17']}")
    print(f" HER2 / CEP17 Ratio    : {report['her2_cep17_ratio']}")
    print(f" Avg HER2 Copy / Cell  : {report['avg_her2_per_cell']}")
    print(f" Diagnostic Status     : >>> {report['status']} <<<")
    print(f" Interpretation        : {report['clinical_interpretation']}")
    print("=" * 70)

    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="HER2 DISH Signal Quantification")
    default_img = os.path.join(current_dir, "data", "sample_positive_patient_c001.png")
    parser.add_argument("--image", type=str, default=default_img, help="Path to input DISH image")
    parser.add_argument("--patient_id", type=str, default="c-001", help="Patient Identifier")
    parser.add_argument("--top_cells", type=int, default=100, help="Maximum number of top cells to analyze")
    parser.add_argument("--output_dir", type=str, default=os.path.join(current_dir, "outputs"), help="Output directory")

    args = parser.parse_args()
    run_pipeline(
        image_path=args.image,
        patient_id=args.patient_id,
        top_cells=args.top_cells,
        output_dir=args.output_dir,
    )

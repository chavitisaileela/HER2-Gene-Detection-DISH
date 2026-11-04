# Dual-Signal Histopathology Analytics for Automated HER2/CEP17 Gene Amplification Scoring in Breast Cancer

**CSIR-Fourth Paradigm Institute (CSIR-4PI) Internship Project**  
*Author:* Chaviti Sai Leela  
*Repository:* [https://github.com/chavitisaileela/HER2-Gene-Detection-DISH](https://github.com/chavitisaileela/HER2-Gene-Detection-DISH)

---

## 1. Project Overview & Clinical Objective

Accurate evaluation of **HER2** (*Human Epidermal Growth Factor Receptor 2*) gene amplification is a critical diagnostic milestone in breast cancer management, directly determining patient eligibility for targeted therapies such as **Trastuzumab (Herceptin)**. 

This project implements an automated digital pathology system designed to analyze brightfield **Dual In Situ Hybridization (DISH)** whole-slide microscopy images:
* **Black Signals**: Chromogenic silver precipitation marking the **HER2 oncogene** on Chromosome 17.
* **Red Signals**: Fast Red chromogen marking the **CEP17 centromeric region** of Chromosome 17 (reference probe).
* **Nuclei**: Counterstained with pale blue/purple hematoxylin.

### Clinical Diagnostic Guidelines (ASCO / College of American Pathologists & Clinical Guidelines):
* **HER2 Positive (Amplification Detected):**
  $$\text{HER2 / CEP17 Ratio} \ge 2.0 \quad \text{AND} \quad \text{Average HER2 Copy Number} \ge 4.0 \text{ per cell}$$
  *Actionable Decision:* Patient is eligible for targeted anti-HER2 therapy.
* **HER2 Negative (No Significant Amplification):**
  $$\text{HER2 / CEP17 Ratio} < 2.0 \quad \text{AND} \quad \text{Average HER2 Copy Number} < 4.0 \text{ per cell}$$
* **Model-Based Top-100 Cell Selection Strategy:** Eliminates inter-observer variability by replacing subjective manual cell counting with automated morphological filtering and confidence-ranked nucleus selection.

---

## 2. System Architecture & Methodology

The analytical pipeline comprises five modular stages:
1. **Multimodal Image Ingestion (`src/data_loader.py`)**: High-resolution loading and validation of brightfield microscopy crops and tiled TIFF/PNG slides ($1500 \times 1500$ px).
2. **Nucleus Segmentation & Quality Filtering (`src/cell_segmentation.py`)**: Identifies cell nuclei, excludes border-clipped cells and debris, enforces circularity and aspect ratio thresholds, and selects the Top-N diagnostically eligible cells.
3. **Dual-Color Chromogenic Quantification (`src/signal_detector.py`)**: Applies spatial top-hat morphological transforms and chromogen color deconvolution to quantify black (HER2) and red (CEP17) dots strictly inside nuclear boundaries.
4. **Clinical Ratio & Guideline Scoring (`src/ratio_analyzer.py`)**: Calculates total probe counts, $\text{HER2/CEP17}$ ratio, average HER2 copy number per cell, and assigns automated diagnostic classifications.
5. **Multi-Scale Visualization Engine (`src/visualizer.py`)**: Produces whole-slide views with green bounding boxes and per-cell counts, alongside 4-stage pipeline diagnostics and cropped nucleus galleries.

---

## 3. Directory Layout

```text
HER2-Gene-Detection-DISH/
│
├── data/
│   ├── generate_sample_dish.py            # Synthetic DISH slide synthesis module
│   ├── sample_positive_patient_c001.png   # HER2-positive amplified case (Ratio: 3.97)
│   ├── sample_negative_patient_c002.png   # HER2-negative non-amplified case (Ratio: 1.46)
│   └── sample_borderline_patient_c003.png # Borderline equivocal case (Ratio: 1.94)
│
├── outputs/                               # Validated outputs and reports
│   ├── csir-c001_annotated_slide.png      # Slide view with bounding boxes and counts
│   ├── csir-c001_cell_gallery.png         # High-confidence cropped nucleus gallery
│   ├── csir-c001_figure7_stages.png       # 4-stage pipeline analysis
│   └── csir-c001_clinical_report.json     # Detailed numerical clinical report
│
├── src/
│   ├── __init__.py
│   ├── data_loader.py                     # Image loading and validation
│   ├── cell_segmentation.py               # Nuclei segmentation & Top-100 ranking
│   ├── signal_detector.py                 # Intracellular black & red dot quantification
│   ├── ratio_analyzer.py                  # Clinical ASCO/CAP ratio computation
│   └── visualizer.py                      # Multi-panel visualization generation
│
├── templates/
│   └── index.html                         # Clinical web application interface
│
├── app.py                                 # Interactive web application
├── main.py                                # Automated CLI pipeline runner
├── requirements.txt                       # Python dependencies
└── README.md                              # Project documentation
```

---

## 4. Execution Instructions

### 1. Command-Line Batch Pipeline
Run automated analysis directly on test slides or clinical cases:
```powershell
# Analyze HER2 Positive Case
python main.py --image "data/sample_positive_patient_c001.png" --patient_id "csir-c001"

# Analyze HER2 Negative Case
python main.py --image "data/sample_negative_patient_c002.png" --patient_id "csir-c002"
```

### 2. Interactive Clinical Web Application
Launch the graphical browser interface:
```powershell
python app.py
```
Open `http://127.0.0.1:5000` in your browser to inspect slides, upload custom DISH images, and generate reports.

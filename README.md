# Automated Detection of HER2 Gene Copy Number in Breast Cancer Using Deep Learning Techniques

Implementation of the clinical breast cancer diagnostic system described in:
> **Terisara Micaraseth, Shanop Shuangshoti, Sakdina Prommaouan, Somruetai Shuangshoti, Rizwan Ullah, Gridsada Phanomchoeng.**  
> *"Automated detection of HER2 gene copy number in breast cancer using deep learning techniques."*  
> **Intelligence-Based Medicine** 13 (2026) 100333. [https://doi.org/10.1016/j.ibmed.2025.100333](https://doi.org/10.1016/j.ibmed.2025.100333)

---

## 1. Clinical Background & Objective

In breast cancer diagnostics, assessing **HER2** (Human Epidermal Growth Factor Receptor 2) gene amplification is decisive for identifying eligibility for targeted anti-HER2 therapies (such as Trastuzumab/Herceptin).

This system analyzes brightfield **Dual In Situ Hybridization (DISH)** histopathology images:
* **Black Dots**: HER2 gene probe signals (silver acetate reaction).
* **Red Dots**: CEP17 chromosome 17 centromeric probe signals (Fast Red reaction).

### Official Clinical Diagnostic Guidelines (ASCO/CAP & Paper Table 2):
* **Positive (Amplification Detected)**:
  $$\text{HER2 / CEP17 Ratio} \ge 2.0 \quad \text{AND} \quad \text{Average HER2 Copy Number} \ge 4.0 \text{ per cell}$$
* **Negative (No Amplification)**:
  $$\text{HER2 / CEP17 Ratio} < 2.0 \quad \text{AND} \quad \text{Average HER2 Copy Number} < 4.0 \text{ per cell}$$
* **Model-Based Top-100 Cell Selection**: Instead of subjective manual counting (20–60 cells), the pipeline automatically ranks nuclei by morphological quality and evaluates the top 100 diagnostically eligible cells.

---

## 2. System Architecture

The pipeline consists of 5 modular components:
1. **Data Loader (`src/data_loader.py`)**: Loads high-resolution TIFF, PNG, or JPG brightfield microscopy tiles ($1500 \times 1500$ px).
2. **Cell Segmentation (`src/cell_segmentation.py`)**: Identifies cell nuclei, applies pathologist inclusion criteria (excluding overlapping cells, debris, and edge artifacts), and ranks nuclei by confidence.
3. **Signal Detection (`src/signal_detector.py`)**: Quantifies intracellular black HER2 and red CEP17 chromogenic dots strictly within the nuclear perimeter.
4. **Clinical Ratio Analyzer (`src/ratio_analyzer.py`)**: Computes total signal counts, ratio, average copy number per cell, and classifies amplification status.
5. **Visualization (`src/visualizer.py`)**: Generates annotated whole-slide views (green bounding boxes with counts, Fig. 4) and 4-stage pipeline outputs (Fig. 7).

---

## 3. Project Directory Structure

```text
HER2_DISH_Quantification/
│
├── data/
│   ├── generate_sample_dish.py            # Generates realistic DISH benchmark slides
│   ├── sample_positive_patient_c001.png   # HER2-positive amplified case
│   ├── sample_negative_patient_c002.png   # HER2-negative case
│   └── sample_borderline_patient_c003.png # Borderline case
│
├── outputs/                               # Saved visual outputs and JSON reports
│   ├── c-001_annotated_slide.png          # Full slide view with bounding boxes (Fig. 4)
│   ├── c-001_cell_gallery.png             # Cropped nuclei gallery (Fig. 4 Top)
│   ├── c-001_figure7_stages.png           # 4-stage breakdown (Fig. 7)
│   └── c-001_clinical_report.json         # Complete numerical diagnostic report
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
│   └── index.html                         # Clinical web interface template
│
├── app.py                                 # Clinical web application (Fig. 4 UI)
├── main.py                                # Command-line pipeline runner
├── requirements.txt                       # Project dependencies
└── README.md                              # Project documentation
```

---

## 4. Execution Instructions (Windows VS Code)

### Step 1: Open VS Code Terminal
Open your terminal in VS Code (`Ctrl + ~`) and navigate to the project directory:
```powershell
cd C:\Users\chavi\.gemini\antigravity\scratch\HER2_DISH_Quantification
```

### Step 2: Run Command-Line Analysis
Execute the pipeline on any sample slide or your own clinical slide:

```powershell
# Run on HER2 Positive Case
python main.py --image "data/sample_positive_patient_c001.png" --patient_id "c-001"

# Run on HER2 Negative Case
python main.py --image "data/sample_negative_patient_c002.png" --patient_id "c-002"
```

### Step 3: Launch Interactive Clinical Web Application
Launch the standalone web application replicating Figure 4 from the paper:
```powershell
python app.py
```
Then open your web browser at:
```
http://127.0.0.1:5000
```
From the web interface:
* Enter any **Patient ID** (e.g. `c-001`).
* Upload an image file or choose one of the preloaded sample slides.
* Click **Run Analysis** to view the real-time clinical summary report, nucleus gallery, and annotated slide.

"""
app.py
Standalone Web Application for HER2 DISH Image Analysis.
Recreates the clinical user interface detailed in Section 3.6 and Figure 4 of:
Micaraseth et al., 'Automated detection of HER2 gene copy number in breast cancer
using deep learning techniques', Intelligence-Based Medicine 13 (2026) 100333.
"""

import os
import sys
from flask import Flask, render_template, request, send_from_directory, redirect, url_for
from werkzeug.utils import secure_filename

# Ensure src is importable
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(current_dir, "src")
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from main import run_pipeline

app = Flask(__name__, template_folder="templates")

DATA_DIR = os.path.join(current_dir, "data")
OUTPUT_DIR = os.path.join(current_dir, "outputs")
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html", report=None)


@app.route("/analyze", methods=["POST"])
def analyze():
    patient_id = request.form.get("patient_id", "c-001").strip()
    if not patient_id:
        patient_id = "c-001"

    uploaded_file = request.files.get("slide_file")
    sample_select = request.form.get("sample_select")

    # Determine input image path
    if uploaded_file and uploaded_file.filename != "":
        filename = secure_filename(uploaded_file.filename)
        save_path = os.path.join(DATA_DIR, f"{patient_id}_{filename}")
        uploaded_file.save(save_path)
        img_path = save_path
    elif sample_select:
        img_path = os.path.join(DATA_DIR, sample_select)
    else:
        # Default fallback to sample positive slide
        img_path = os.path.join(DATA_DIR, "sample_positive_patient_c001.png")

    report = run_pipeline(
        image_path=img_path,
        patient_id=patient_id,
        top_cells=100,
        output_dir=OUTPUT_DIR,
    )

    return render_template("index.html", report=report)


@app.route("/outputs/<path:filename>")
def serve_output(filename):
    return send_from_directory(OUTPUT_DIR, filename)


@app.route("/data/<path:filename>")
def serve_data(filename):
    return send_from_directory(DATA_DIR, filename)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    host = os.environ.get("HOST", "0.0.0.0")
    print(f"[*] Starting Clinical Web Server at http://{host}:{port}")
    app.run(host=host, port=port, debug=False)

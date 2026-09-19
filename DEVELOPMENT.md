# Developer Guide & Local Setup

This document provides setup and execution instructions for developing, testing, and running the Brain Tumor Detection project.

---

## 1. Prerequisites & Environment Setup

- **Python Version:** Python 3.10 or 3.11 is recommended.
- **Operating System:** Linux, macOS, or Windows.

### Recommended: Creating a Virtual Environment

Using standard Python `venv`:

```bash
# Create a virtual environment named .venv
python -m venv .venv

# Activate the virtual environment:
# On Windows (PowerShell):
.venv\Scripts\Activate.ps1
# On Linux / macOS:
source .venv/bin/activate
```

---

## 2. Dependency Installation

Install runtime dependencies:

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

For development and running unit tests:

```bash
pip install -e ".[dev]"
```

---

## 3. Repository Structure & Artifacts

- **Model Artifact Location:**  
  The trained Keras model is expected at `models/brain_tumor_detector.keras`.  
  The application automatically resolves this path relative to the project root.
- **Original Notebook Location:**  
  The historical reference notebook is archived at `notebooks/archive/original_brain_tumor_detection.ipynb`.

---

## 4. Running the Streamlit Web Application

Launch the interactive Streamlit interface from the repository root:

```bash
streamlit run app/streamlit_app.py
```

The application will open in your default browser at `http://localhost:8501`.

---

## 5. Running the Test Suite

Execute the unit test suite with `pytest`:

```bash
pytest
```

To run with verbose output:

```bash
pytest -v
```

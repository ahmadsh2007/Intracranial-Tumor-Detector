# ICTD: Intracranial Tumor Detector

[![License: CC BY-NC 4.0](https://img.shields.io/badge/License-CC%20BY--NC%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc/4.0/)
[![Python](https://img.shields.io/badge/Python-3.10-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-Keras-orange.svg)](https://www.tensorflow.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B.svg)](https://streamlit.io/)

A two-stage convolutional neural network (CNN) pipeline that detects brain tumors in MRI scans and classifies their type, with a Streamlit app that lets clinicians upload scans and view predictions.

🏆 **1st Overall, BT Young Scientist International Competition (2025)** · 🥇 **1st Overall, Jordan Young Scientist (JoYS) (2024)**

## Results (V1.2)

Measured on held-out test images that were never used in training:

| Stage | Task | Test images | Correct | Accuracy |
|-------|------|-------------|---------|----------|
| 1 | Tumor vs. no tumor | 1,311 | 1,286 | **98.1%** |
| 2 | Tumor type (glioma / meningioma / pituitary) | 906 | 838 | **92.5%** |

Per-class accuracy by stage:

- **Stage 1:** tumor 98.2%, no tumor 97.8%
- **Stage 2:** pituitary 98.0%, meningioma 91.5%, glioma 88.0%

> The 2024 JoYS submission (V1.0) reported just over 65%. V1.2 improves on it with the two-stage design and retrained models.

## Why

Diagnostic errors are among the most common kinds of medical malpractice, and cancer is one of the conditions most often misdiagnosed. In Jordan, delays in getting MRI scans make the problem worse. ICTD is a decision-support tool for spotting tumors sooner and more consistently. It is **not** a replacement for a radiologist.

## How it works

1. **Dataset**: a public Kaggle brain-MRI dataset merged with images from a local hospital.
2. **Preprocessing**: resizing to 256×256, normalizing pixel values to [0, 1], and augmentation. The helper scripts are in `File and Image Utilities/`.
3. **Stage 1 model**: a CNN that classifies a scan as *tumor* or *no tumor*.
4. **Stage 2 model**: a CNN that classifies the tumor type of positive scans as *glioma*, *meningioma* or *pituitary*.
5. **App**: a Streamlit interface for uploading scans, viewing predictions and analysing training metrics (`GUI/GUI_V1.2.py`).

## Repository layout

| Path | Contents |
|------|----------|
| `V1.2/` | Current models: `MainCatClassification.ipynb` (stage 1) and `TumorClassification.ipynb` (stage 2) |
| `V1.1/` | Single multi-class model experiments |
| `V1.0/` | Original per-tumor binary models (JoYS 2024) |
| `GUI/` | Streamlit app and training-metric CSVs |
| `5_Classes/` | Inference and testing scripts |
| `File and Image Utilities/` | Dataset merging, splitting, resizing and renaming tools |
| `JoYS.pdf` | Full JoYS 2024 research report |

## Running the app

```bash
pip install -r GUI/requirements.txt
streamlit run GUI/GUI_V1.2.py
```

Trained model files are not committed to the repo, so train them with the V1.2 notebooks first. You may also need to update the file paths at the top of `GUI_V1.2.py`.

## Team

Built by the **MSJS team** (Yarmouk University Model School). **Ahmad Shatnawi** was lead software developer: model development, preprocessing pipeline and app.

## License

Licensed under the [Creative Commons Attribution-NonCommercial 4.0 International License](./LICENSE).
Copyright (c) 2025 Ahmad Shatnawi, MSJS Team

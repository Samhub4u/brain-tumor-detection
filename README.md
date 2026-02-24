# 🧠 Brain Tumor Detection — AI-Powered MRI Analysis

A deep learning web application that detects brain tumors from MRI scans using a fine-tuned **ResNet50** model with **Grad-CAM** visual explanations.

Built as part of a thesis project on medical image classification using transfer learning.

![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15-orange?logo=tensorflow)
![Flask](https://img.shields.io/badge/Flask-3.1-lightgrey?logo=flask)
![License](https://img.shields.io/badge/License-MIT-green)

---

## Features

- **Binary Classification** — Detects *Tumor* vs *Normal* from brain MRI images
- **Grad-CAM Heatmaps** — Visual explanation highlighting regions that influenced the prediction
- **Interactive Image Viewer** — Zoom, pan, and rotate MRI scans
- **Prediction History** — Archives all scans with timestamps and confidence scores
- **Doctor Feedback** — Allows clinicians to confirm or correct AI predictions
- **Responsive UI** — Futuristic glassmorphism medical theme

## Tech Stack

| Component | Technology |
|-----------|-----------|
| **Backend** | Flask (Python) |
| **ML Model** | ResNet50 (Transfer Learning) |
| **Framework** | TensorFlow 2.15 / Keras 3.x |
| **Explainability** | Grad-CAM |
| **Frontend** | HTML5, CSS3 (Glassmorphism), Vanilla JS |
| **Image Processing** | Pillow, OpenCV |

## Project Structure

```
brain-tumor-detection/
├── app.py                  # Flask application & routes
├── requirements.txt        # Python dependencies
├── model/
│   └── model_20layers.best.h5  # Fine-tuned ResNet50 (Git LFS)
├── static/
│   ├── css/style.css       # Glassmorphism theme
│   ├── js/main.js          # Image viewer & interactions
│   └── uploads/            # Uploaded MRI scans
├── templates/
│   ├── base.html           # Base layout with navbar/footer
│   ├── index.html          # Home — upload form
│   ├── result.html         # Prediction results & Grad-CAM
│   ├── history.html        # Scan archive
│   ├── about.html          # Model & project info
│   ├── contact.html        # Support page
│   └── medical_info.html   # Brain tumor information
└── utils/
    ├── model_loader.py     # Model loading & prediction logic
    └── gradcam.py          # Grad-CAM heatmap generation
```

## Quick Start

### Prerequisites

- Python 3.10
- Git LFS (for the model file)

### Installation

```bash
# Clone the repository
git clone https://github.com/Samhub4u/brain-tumor-detection.git
cd brain-tumor-detection

# Pull the model file (Git LFS)
git lfs pull

# Create virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS/Linux

# Install dependencies
pip install -r requirements.txt

# Run the app
python app.py
```

Open **http://localhost:5000** in your browser.

## How It Works

1. **Upload** a brain MRI scan (JPG, PNG, BMP)
2. The image is preprocessed and resized to 224×224 pixels
3. The **ResNet50** model classifies it as *Tumor* or *Normal*
4. **Grad-CAM** generates a heatmap showing which regions influenced the decision
5. Results are displayed with confidence scores and visual explanations

## Model Details

| Property | Value |
|----------|-------|
| Architecture | ResNet50 + GlobalAveragePooling + Dense(512) + Sigmoid |
| Training | Transfer learning — top 20 layers fine-tuned |
| Input Shape | 224 × 224 × 3 |
| Output | Binary (sigmoid) — 0 = Normal, 1 = Tumor |
| Preprocessing | Rescale to [0, 1] (divide by 255) |
| Optimizer | Adam (lr=1e-5) |
| Loss | Binary Crossentropy |

## Screenshots

Upload an MRI scan and get instant AI-powered analysis with Grad-CAM explanations.

## Disclaimer

> **This application is for educational and research purposes only.** It is not intended to replace professional medical diagnosis. Always consult qualified healthcare professionals for medical decisions.

## License

This project is part of a thesis research work. Feel free to use it for educational purposes.

## Author

**Samhub4u** — [GitHub](https://github.com/Samhub4u)

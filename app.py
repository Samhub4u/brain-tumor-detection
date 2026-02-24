"""
Brain Tumor Detection Web Application
ResNet50 Transfer Learning — Flask App

Usage:
    1. Place your trained model (.h5) in the 'model/' directory
    2. pip install -r requirements.txt
    3. python app.py
"""

import os
import json
import uuid
import glob
from datetime import datetime

from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
import numpy as np
from PIL import Image

from utils.model_loader import load_model_from_path, predict, get_model, get_class_names
from utils.gradcam import get_gradcam_heatmap, overlay_gradcam

# ─── App Configuration ───────────────────────────────────────────────
app = Flask(__name__)
app.secret_key = 'brain-tumor-detection-2026'

# Use absolute paths based on app root so it works regardless of cwd
_APP_DIR = os.path.dirname(os.path.abspath(__file__))
app.config['UPLOAD_FOLDER'] = os.path.join(_APP_DIR, 'static', 'uploads')
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB max

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp', 'tif', 'tiff'}
HISTORY_FILE = os.path.join(_APP_DIR, 'static', 'history.json')


# ─── Helper Functions ─────────────────────────────────────────────────
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def load_history():
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, 'r') as f:
            return json.load(f)
    return []


def save_history(history):
    with open(HISTORY_FILE, 'w') as f:
        json.dump(history, f, indent=2)


def find_model_file():
    """Search for .h5 or .keras model files in the model/ directory."""
    model_dir = os.path.join(os.path.dirname(__file__), 'model')
    for ext in ['*.h5', '*.hdf5', '*.keras']:
        files = glob.glob(os.path.join(model_dir, ext))
        if files:
            return files[0]
    return None


# ─── Load Model at Startup ───────────────────────────────────────────
model_path = find_model_file()
model_loaded = False

if model_path:
    try:
        load_model_from_path(model_path)
        model_loaded = True
        print(f"[OK] Model loaded: {model_path}")
    except Exception as e:
        import traceback
        print(f"[ERROR] Failed to load model: {e}")
        traceback.print_exc()
else:
    print("[WARNING] No model file found in 'model/' directory.")
    print("  Place your .h5 model there and restart the app.")

print(f"[DEBUG] model_loaded = {model_loaded}, model_path = {model_path}")


# ─── Routes ───────────────────────────────────────────────────────────

@app.route('/')
def index():
    """Home page — upload form."""
    return render_template('index.html', model_loaded=model_loaded)

@app.route('/medical-info')
def medical_info():
    """Medical information page about brain tumors."""
    return render_template('medical_info.html')

@app.route('/contact')
def contact():
    """Contact and support page."""
    return render_template('contact.html')

@app.route('/predict', methods=['POST'])
def predict_route():
    """Handle image upload and return prediction."""
    if not model_loaded:
        flash('No model loaded. Please place your .h5 model in the model/ folder and restart.', 'error')
        return redirect(url_for('index'))

    if 'file' not in request.files:
        flash('No file selected.', 'error')
        return redirect(url_for('index'))

    file = request.files['file']
    if file.filename == '':
        flash('No file selected.', 'error')
        return redirect(url_for('index'))

    if not allowed_file(file.filename):
        flash('Invalid file type. Please upload a PNG, JPG, or BMP image.', 'error')
        return redirect(url_for('index'))

    # Save uploaded file
    filename = f"{uuid.uuid4().hex[:12]}_{file.filename}"
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    try:
        # Run prediction
        result = predict(filepath)

        # Generate Grad-CAM
        gradcam_filename = None
        try:
            model = get_model()
            heatmap = get_gradcam_heatmap(model, result['img_array'])
            if heatmap.max() > 0:
                overlay = overlay_gradcam(result['original_img'], heatmap, alpha=0.45)
                gradcam_filename = f"gradcam_{filename}"
                gradcam_path = os.path.join(app.config['UPLOAD_FOLDER'], gradcam_filename)
                Image.fromarray(overlay).save(gradcam_path)
        except Exception as e:
            print(f"Grad-CAM error: {e}")

        # Determine severity for UI coloring
        predicted = result['predicted_class']
        if predicted.lower() in ['normal', 'no tumor']:
            severity = 'normal'
        else:
            severity = 'tumor'

        # Save to history
        history = load_history()
        record = {
            'id': uuid.uuid4().hex[:8],
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'filename': file.filename,
            'saved_as': filename,
            'predicted_class': predicted,
            'confidence': result['confidence'],
            'probabilities': result['all_probabilities'],
            'gradcam': gradcam_filename,
            'severity': severity
        }
        history.insert(0, record)
        save_history(history)

        return render_template(
            'result.html',
            result=record,
            model_loaded=model_loaded
        )

    except Exception as e:
        flash(f'Prediction failed: {str(e)}', 'error')
        return redirect(url_for('index'))


@app.route('/history', methods=['GET'])
def history():
    """Show prediction history."""
    records = load_history()
    return render_template('history.html', records=records, model_loaded=model_loaded)


@app.route('/feedback', methods=['POST'])
def save_feedback():
    data = request.json
    filename = data.get('filename')
    feedback_type = data.get('feedback') # 'correct' or 'incorrect'
    actual_label = data.get('label', '')

    history = load_history()
    updated = False
    
    for record in history:
        if record['saved_as'] == filename:
            record['feedback'] = {
                'type': feedback_type,
                'corrected_label': actual_label,
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            updated = True
            break
            
    if updated:
        save_history(history)
        return jsonify({'status': 'success'})
    
    return jsonify({'status': 'error', 'message': 'Record not found'}), 404


@app.route('/history/clear', methods=['POST'])
def clear_history():
    """Clear all history."""
    save_history([])
    
    # Only delete files if they exist
    upload_dir = app.config['UPLOAD_FOLDER']
    if os.path.exists(upload_dir):
        for f in os.listdir(upload_dir):
            file_path = os.path.join(upload_dir, f)
            if f != '.gitkeep' and os.path.isfile(file_path):
                try:
                    os.remove(file_path)
                except:
                    pass
    
    flash('History cleared.', 'info')
    return redirect(url_for('history'))


@app.route('/about')
def about():
    """About page with model details."""
    class_names = get_class_names() if model_loaded else []
    return render_template('about.html', model_loaded=model_loaded, class_names=class_names)


# ─── Run ──────────────────────────────────────────────────────────────
if __name__ == '__main__':
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    print("\n" + "=" * 55)
    print("  Brain Tumor Detection — Web Application")
    print("  http://localhost:5000")
    print("=" * 55 + "\n")
    app.run(debug=True, host='0.0.0.0', port=5000)

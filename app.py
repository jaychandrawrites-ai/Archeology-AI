from flask import Flask, request, redirect, render_template_string, flash
import os
import pickle
import numpy as np
import cv2
import pandas as pd
from werkzeug.utils import secure_filename
import uuid
import logging

# ------------------ Configuration ------------------
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.secret_key = 'replace-with-a-secret-key'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

logging.basicConfig(level=logging.INFO)

# ------------------ HTML TEMPLATE (Embedded) ------------------
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Image Classifier App</title>
    <style>
        body { font-family: Arial; background: #f2f2f2; display: flex; justify-content: center; align-items: center; height: 100vh; margin:0; }
        .container { background: #fff; padding:30px; border-radius:10px; box-shadow:0 0 15px rgba(0,0,0,0.2); text-align:center; width:420px; }
        input[type="file"] { margin:15px 0; }
        button { padding:10px 20px; border:none; background:#007bff; color:white; border-radius:5px; cursor:pointer; }
        button:hover { background:#0056b3; }
        ul { text-align:left; margin-top:20px; }
        .flash { color:red; margin-bottom:10px; }
    </style>
</head>
<body>
    <div class="container">
        <h1>Image Classification App</h1>

        {% with messages = get_flashed_messages() %}
            {% if messages %}
                <div class="flash">
                    {% for msg in messages %}
                        <p>{{ msg }}</p>
                    {% endfor %}
                </div>
            {% endif %}
        {% endwith %}

        <form action="/upload" method="POST" enctype="multipart/form-data">
            <input type="file" name="image" accept="image/*" required>
            <button type="submit">Classify Image</button>
        </form>

        {% if result %}
            <h2>Classification Result:</h2>
            <ul>
                {% for key, value in result.items() %}
                    <li><strong>{{ key }}:</strong> {{ value }}</li>
                {% endfor %}
            </ul>
        {% endif %}
    </div>
</body>
</html>
"""

# ------------------ Load Model and CSV ------------------
import zipfile

def unzip_database(zip_path="database.zip", extract_to="."):
    """
    Safely unzip the model/database if a zip file exists.
    Only extracts expected files.
    """
    if not os.path.exists(zip_path):
        logging.info("No ZIP database found — skipping unzip.")
        return

    logging.info(f"Found {zip_path}. Extracting...")

MODEL_PATH = "model.pkl"
CSV_PATH = "open-context-12473-records.csv"

# Load model
MODEL_PATH = "model.pkl"
CSV_PATH = "open-context-12473-records.csv"

# Load model
try:
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    logging.info("Model loaded successfully.")
except Exception as e:
    raise RuntimeError(f"❌ Failed to load model.pkl: {e}")

# Load CSV to dictionary (adjust if needed)
try:
    df = pd.read_csv(CSV_PATH)
    records = df.set_index(df.columns[0]).to_dict(orient="index")
    logging.info("CSV loaded successfully.")
except Exception as e:
    raise RuntimeError(f"❌ Failed to load CSV: {e}")
    
# ------------------ Helper Functions ------------------
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def process_image(file_path):
    """
    Process image: grayscale → dilation → FFT LP filter →
    inverse FFT → resize to 23×23 → normalize → flatten.
    """
    img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError("Invalid or corrupted image")

    # Dilation
    kernel = np.ones((3, 3), np.uint8)
    dilated = cv2.dilate(img, kernel, iterations=1)

    # FFT
    f = np.fft.fft2(dilated)
    fshift = np.fft.fftshift(f)

    rows, cols = fshift.shape
    crow, ccol = rows // 2, cols // 2

    # Low-pass filter mask
    mask = np.zeros((rows, cols), np.uint8)
    r = 30  # radius
    x, y = np.ogrid[:rows, :cols]
    mask_area = (x - crow)**2 + (y - ccol)**2 <= r*r
    mask[mask_area] = 1

    fshift *= mask

    # Inverse FFT
    f_ishift = np.fft.ifftshift(fshift)
    img_back = np.abs(np.fft.ifft2(f_ishift))

    # 🔥 Resize to 23×23
    img_resized = cv2.resize(img_back, (23, 23))

    # Normalize + flatten
    img_norm = img_resized / 255.0
    return img_norm.reshape(1, -1)


# ------------------ Routes ------------------
@app.route('/', methods=['GET'])
def index():
    return render_template_string(HTML_TEMPLATE)


@app.route('/upload', methods=['POST'])
def upload_file():
    if 'image' not in request.files:
        flash("No file uploaded.")
        return redirect('/')

    file = request.files['image']
    if file.filename == "":
        flash("No file selected.")
        return redirect('/')

    if not allowed_file(file.filename):
        flash("Invalid file format. Use PNG/JPG/JPEG.")
        return redirect('/')

    filename = secure_filename(file.filename)
    unique_name = f"{uuid.uuid4().hex}_{filename}"
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_name)
    file.save(filepath)

    try:
        img_input = process_image(filepath)
    except Exception as e:
        flash(f"Image processing error: {e}")
        return redirect('/')

    try:
        prediction = model.predict(img_input)[0]
    except Exception as e:
        flash(f"Model prediction error: {e}")
        return redirect('/')

    result_info = records.get(prediction, {"predicted_class": prediction})

    return render_template_string(HTML_TEMPLATE, result=result_info)


# ------------------ Run App ------------------
if __name__ == '__main__':
    app.run(debug=True)
    

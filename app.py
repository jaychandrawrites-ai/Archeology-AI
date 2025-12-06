from flask import Flask, request, redirect, render_template_string, flash
import os
import pickle
import numpy as np
import cv2
import pandas as pd
from werkzeug.utils import secure_filename
import uuid
import logging

# ----------------------------------
# App Configuration
# ----------------------------------
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.secret_key = 'replace-with-a-secret-key'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

logging.basicConfig(level=logging.INFO)


# ----------------------------------
# Load Model + CSV WITH SAFE UNZIP
# ----------------------------------
import zipfile

MODEL_PATH = "model.pkl"
CSV_PATH = "open-context-12473-records.csv"
ZIP_PATH = "database.zip"

# ---- Load Model (NOT zipped) ----
try:
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    logging.info("Model loaded successfully.")
except Exception as e:
    raise RuntimeError(f"❌ Failed to load model.pkl: {e}")


# ---- Safe unzip function ----
def safe_unzip_csv(zip_path, target_csv):
    """
    Extract ONLY 'open-context-12473-records.csv' safely.
    Prevents extraction of other files or malicious paths.
    """
    if not os.path.exists(zip_path):
        logging.warning("No ZIP found. Using existing CSV if present.")
        return

    logging.info(f"ZIP found: {zip_path}")

    try:
        with zipfile.ZipFile(zip_path, 'r') as z:
            # Make sure the expected CSV exists inside
            names = z.namelist()
            if target_csv not in [os.path.basename(n) for n in names]:
                raise RuntimeError(f"CSV '{target_csv}' not found inside ZIP.")

            for member in names:
                filename = os.path.basename(member)

                # skip folders
                if not filename:
                    continue

                # ONLY allow extraction of the target CSV
                if filename != target_csv:
                    logging.warning(f"Skipping unexpected file: {filename}")
                    continue

                # secure path
                target_path = os.path.abspath(target_csv)
                with z.open(member) as src, open(target_path, "wb") as dst:
                    dst.write(src.read())

                logging.info(f"Extracted CSV: {target_path}")

    except Exception as e:
        raise RuntimeError(f"❌ Error during CSV unzip: {e}")


# ---- Extract CSV if missing ----
if not os.path.exists(CSV_PATH):
    logging.info(f"{CSV_PATH} not found → extracting from ZIP...")
    safe_unzip_csv(ZIP_PATH, "open-context-12473-records.csv")
else:
    logging.info("CSV already exists — skipping unzip.")


# ---- Load CSV ----
try:
    df = pd.read_csv(CSV_PATH)
    records = df.set_index(df.columns[0]).to_dict(orient="index")
    logging.info("CSV loaded successfully.")
except Exception as e:
    raise RuntimeError(f"❌ Failed to load CSV: {e}")

# ----------------------------------
# Utility Functions
# ----------------------------------
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def process_image(file_path):
    """Full preprocessing pipeline identical to your original."""
    img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError("Invalid or corrupted image")

    kernel = np.ones((3, 3), np.uint8)
    dilated = cv2.dilate(img, kernel, iterations=1)

    f = np.fft.fft2(dilated)
    fshift = np.fft.fftshift(f)

    rows, cols = fshift.shape
    crow, ccol = rows // 2, cols // 2

    mask = np.zeros((rows, cols), np.uint8)
    r = 30
    x, y = np.ogrid[:rows, :cols]
    mask_area = (x - crow)**2 + (y - ccol)**2 <= r*r
    mask[mask_area] = 1

    fshift *= mask
    f_ishift = np.fft.ifftshift(fshift)
    img_back = np.abs(np.fft.ifft2(f_ishift))

    img_resized = cv2.resize(img_back, (23, 23))
    img_norm = img_resized / 255.0
    return img_norm.reshape(1, -1)


# ----------------------------------
# HTML Template (Modern Redesign)
# ----------------------------------
HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Image Classifier</title>
<style>
    body {
        margin: 0;
        padding: 0;
        font-family: 'Segoe UI', sans-serif;
        background: linear-gradient(135deg, #4b79a1, #283e51);
        height: 100vh;
        display: flex;
        justify-content: center;
        align-items: center;
    }

    .card {
        width: 450px;
        background: rgba(255, 255, 255, 0.15);
        padding: 30px;
        border-radius: 16px;
        backdrop-filter: blur(15px);
        box-shadow: 0 8px 25px rgba(0,0,0,0.2);
        color: white;
        text-align: center;
        animation: fadeIn 0.7s ease-in-out;
    }

    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }

    h1 {
        font-weight: 300;
        margin-bottom: 20px;
        letter-spacing: 1px;
    }

    input[type="file"] {
        padding: 12px;
        background: rgba(255,255,255,0.2);
        border-radius: 8px;
        border: 1px solid rgba(255,255,255,0.3);
        color: white;
        width: 100%;
        margin-top: 10px;
    }

    button {
        margin-top: 20px;
        padding: 12px 20px;
        border: none;
        background: #00d4ff;
        color: #00344d;
        font-weight: bold;
        border-radius: 8px;
        cursor: pointer;
        transition: 0.2s;
        width: 100%;
        font-size: 16px;
    }

    button:hover {
        background: #00a8cc;
    }

    .result-box {
        background: rgba(255,255,255,0.2);
        padding: 15px;
        border-radius: 10px;
        text-align: left;
        margin-top: 20px;
    }

    .flash {
        color: #ffbbbb;
        margin-bottom: 10px;
    }

    .credits {
        margin-top: 20px;
        font-size: 14px;
        opacity: 0.8;
    }
</style>
</head>
<body>

<div class="card">
    <h1>Image Classification</h1>

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
    <div class="result-box">
        <h2>Result:</h2>
        <ul>
            {% for key, value in result.items() %}
            <li><strong>{{ key }}:</strong> {{ value }}</li>
            {% endfor %}
        </ul>
    </div>
    {% endif %}

    <div class="credits">
        Created with ❤️ by <strong>Jay Chandra</strong> & <strong>Ishan Nathan</strong>
    </div>
</div>

</body>
</html>
"""


# ----------------------------------
# Routes
# ----------------------------------
@app.route('/')
def index():
    return render_template_string(HTML)


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
        flash("Invalid format. Use PNG/JPG/JPEG.")
        return redirect('/')

    filename = secure_filename(file.filename)
    unique_name = f"{uuid.uuid4().hex}_{filename}"
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_name)
    file.save(filepath)

    try:
        img_input = process_image(filepath)
    except Exception as e:
        flash(f"Image processing failed: {e}")
        return redirect('/')

    try:
        prediction = model.predict(img_input)[0]
    except Exception as e:
        flash(f"Model prediction failed: {e}")
        return redirect('/')

    result_info = records.get(prediction, {"predicted_class": prediction})
    return render_template_string(HTML, result=result_info)


# ----------------------------------
# Run App
# ----------------------------------
if __name__ == '__main__':
    app.run(debug=True)

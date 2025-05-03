import os
import cv2
import pytesseract
import re
from PIL import Image
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'

# Optional: Set the Tesseract path if needed (for Windows)
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Preprocess image for OCR (grayscale + adaptive thresholding)
def preprocess_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("❌ Image not found or failed to load.")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Invert and apply adaptive thresholding
    processed = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        15, 8
    )
    return processed

# OCR to extract math expression
def extract_text_from_image(image):
    pil_image = Image.fromarray(image)
    if pil_image.mode != 'RGB':
        pil_image = pil_image.convert('RGB')

    config = r'--oem 3 --psm 7 -c tessedit_char_whitelist=0123456789+-*/().'
    text = pytesseract.image_to_string(pil_image, config=config)
    return text.strip()

# Extract and evaluate mathematical expression
def extract_and_evaluate_expression(text):
    match = re.search(r'[\d+\-*/(). ]+', text)
    if match:
        expr = match.group().strip()
        try:
            result = eval(expr)
            return expr, result
        except Exception as e:
            return expr, f"❌ Error evaluating: {e}"
    return None, "❌ No valid expression found."

# Flask routes
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'image' not in request.files:
            return render_template('index.html', error="❌ No file uploaded")

        file = request.files['image']
        if file.filename == '':
            return render_template('index.html', error="❌ No file selected")

        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        try:
            preprocessed = preprocess_image(filepath)
            text = extract_text_from_image(preprocessed)
            expression, result = extract_and_evaluate_expression(text)
            return render_template('index.html', expr=expression, result=result, image_url=filepath)
        except Exception as e:
            return render_template('index.html', error=str(e))

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)

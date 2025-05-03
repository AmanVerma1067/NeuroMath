import os
import cv2
import pytesseract
import re
from PIL import Image
from flask import Flask, request, render_template, redirect, url_for
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'

# Optional: set Tesseract path (Windows)
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

def preprocess_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("❌ Image not found or failed to load.")
    return image  # Skip preprocessing for now

def extract_text_from_image(preprocessed_image):
    # Convert OpenCV image (NumPy array) to PIL Image
    pil_image = Image.fromarray(preprocessed_image)

    if pil_image.mode != 'RGB':
        pil_image = pil_image.convert('RGB')

    # Use OCR config optimized for math expressions
    custom_config = r'--oem 1 --psm 7 -c tessedit_char_whitelist=0123456789+-*/().'
    text = pytesseract.image_to_string(pil_image, config=custom_config)
    return text

def extract_and_evaluate_expression(text):
    print("Raw OCR Text:", repr(text))  # Debug output

    match = re.search(r'[\d+\-*/(). ]+', text)
    if match:
        expr = match.group().strip()
        try:
            result = eval(expr)
            return expr, result
        except Exception as e:
            return expr, f"Error in evaluation: {e}"
    return None, "No valid expression found."

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'image' not in request.files:
            return render_template('index.html', error="No file uploaded")

        file = request.files['image']
        if file.filename == '':
            return render_template('index.html', error="No file selected")

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

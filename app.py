from flask import Flask, render_template, request, jsonify
import os
import cv2
import numpy as np
import torch
import torch.nn as nn
from werkzeug.utils import secure_filename
import re
import matplotlib.pyplot as plt
import time

app = Flask(__name__)

# Configuration
UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load the trained model
# Update the model definition in app.py
class EnhancedMathModel(nn.Module):
    def __init__(self):
        super(EnhancedMathModel, self).__init__()
        # Shared feature extractor
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        
        # Character classification head
        self.char_head = nn.Sequential(
            nn.Linear(64*7*7, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 14)  # 10 digits + 4 operators
        )
        
        # Expression evaluation head (added to match saved model)
        self.expr_head = nn.Sequential(
            nn.Linear(64*7*7, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.char_head(x)  # Only use char_head for inference

model = EnhancedMathModel()
model.load_state_dict(torch.load('models/enhanced_math_model.pth', map_location='cpu'))
model.eval()

# Class mapping (matches your training)
CHAR_MAP = {
    0: '0', 1: '1', 2: '2', 3: '3', 4: '4',
    5: '5', 6: '6', 7: '7', 8: '8', 9: '9',
    10: '+', 11: '/', 12: '*', 13: '-'  # Matches your folder names (add->+, div->/, etc.)
}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_image(image):
    """Convert image to model input format"""
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.resize(image, (28, 28))
    image = image.astype(np.float32) / 255.0
    image = np.expand_dims(image, axis=0)  # Add channel dim
    image = np.expand_dims(image, axis=0)  # Add batch dim
    return torch.from_numpy(image)

def segment_characters(image):
    """Improved character segmentation with visualization"""
    # Convert to grayscale and threshold
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Find contours and sort left-to-right
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=lambda c: cv2.boundingRect(c)[0])
    
    characters = []
    for i, cnt in enumerate(contours):
        x, y, w, h = cv2.boundingRect(cnt)
        
        # Skip small noise (adjust thresholds as needed)
        if w < 10 or h < 10:
            continue
            
        # Extract character with padding
        margin = 5
        char_img = gray[max(0, y-margin):min(gray.shape[0], y+h+margin), 
                       max(0, x-margin):min(gray.shape[1], x+w+margin)]
        
        # Add to results
        characters.append({
            'image': char_img,
            'position': (x, y, w, h)
        })
    
    return characters

def safe_eval(expression):
    """Safely evaluate mathematical expressions with validation"""
    # Remove any potentially dangerous characters
    allowed_chars = {'0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '+', '-', '*', '/'}
    cleaned = ''.join(c for c in expression if c in allowed_chars)
    
    # Validate expression structure
    if not re.fullmatch(r'^\d+[\+\-\*/]\d+$', cleaned):
        return "Invalid expression format"
    
    try:
        result = eval(cleaned)
        # Handle division by zero
        if not np.isfinite(result):
            return "Math error (division by zero?)"
        return str(round(result, 2))
    except:
        return "Evaluation error"

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Check if file was uploaded
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'})
            
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'})
        
        if file and allowed_file(file.filename):
            try:
                # Save uploaded file
                filename = secure_filename(file.filename)
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(filepath)
                
                # Read and verify image
                image = cv2.imread(filepath)
                if image is None:
                    return jsonify({'error': 'Could not read image file'})
                
                # Segment characters
                characters = segment_characters(image)
                if not characters:
                    return jsonify({'error': 'No characters detected'})
                
                # Recognize each character
                expression_parts = []
                for char in characters:
                    char_img = char['image']
                    input_tensor = preprocess_image(char_img)
                    
                    with torch.no_grad():
                        output = model(input_tensor)
                        _, pred = torch.max(output, 1)
                        expression_parts.append(CHAR_MAP[pred.item()])
                
                # Combine into final expression
                expression = ''.join(expression_parts)
                result = safe_eval(expression)
                
                # Generate visualization (optional)
                viz_path = None
                if len(characters) > 0:
                    viz = image.copy()
                    for char in characters:
                        x, y, w, h = char['position']
                        cv2.rectangle(viz, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    viz_filename = f"viz_{filename}"
                    viz_path = os.path.join(app.config['UPLOAD_FOLDER'], viz_filename)
                    cv2.imwrite(viz_path, viz)
                
                return jsonify({
                    'success': True,
                    'expression': expression,
                    'result': result,
                    'image_url': f"uploads/{filename}",
                    'viz_url': f"uploads/{viz_filename}" if viz_path else None
                })
                
            except Exception as e:
                return jsonify({
                    'error': f'Processing error: {str(e)}'
                })
    
    return render_template('index.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
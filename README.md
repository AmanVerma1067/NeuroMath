# ✍️ GlyphCalc: Handwritten Expression Calculator

**GlyphCalc** is a Python-based application that uses computer vision and OCR to read and solve handwritten mathematical expressions. Simply upload or scan an image containing handwritten equations, and GlyphCalc will recognize and evaluate them with high accuracy.

---

## 🚀 Features

- 📷 Detect and extract handwritten text using OpenCV and Tesseract OCR  
- 🔍 Image preprocessing for better accuracy  
- 🧠 TensorFlow integration for digit/character classification (if needed)  
- ➕ Supports basic arithmetic: `+`, `-`, `*`, `/`  
- 🖼️ GUI support via Flask (if applicable) or simple CLI  
- 📊 Visual analytics using matplotlib  
- 🗃️ CSV logging with Pandas (optional)

---

## 🛠️ Tech Stack

| Tool/Library   | Purpose                               |
|----------------|----------------------------------------|
| `OpenCV`       | Image preprocessing and contour detection |
| `PyTesseract`  | OCR engine for character recognition    |
| `TensorFlow`   | Optional custom model for character prediction |
| `PIL (Pillow)` | Image format conversion and manipulation |
| `NumPy`        | Numerical operations and matrix handling |
| `Pandas`       | Logging detected expressions/results    |
| `Matplotlib`   | Optional plotting and result visualization |

---

## 📂 Folder Structure
ML!/
│
├── app.py # Main application script
├── templates/ # HTML templates (if using Flask)
├── static/uploads/ # Uploaded images
├── models/ # (Optional) Saved ML models
├── README.md
└── requirements.txt # List of dependencies


📌 Todo
 Add support for multi-line expressions

 Improve accuracy using a custom trained CNN

 Web deployment on Heroku or Render

 Scientific function support (sin, log, etc.)

🤝 Contributors
You – OCR integration, app logic
AmanVerma1067 – Repo maintainer and reviewer

📜 License
MIT License – free to use, modify, and share.

💡 Inspiration
This project was built to explore the intersection of OCR and AI for real-world automation, starting with something simple yet useful — solving handwritten math!

![Python](https://img.shields.io/badge/Python-3.11-blue)
![Gradio](https://img.shields.io/badge/Gradio-WebApp-orange)
![Tesseract](https://img.shields.io/badge/Tesseract-OCR-green)
![OpenCV](https://img.shields.io/badge/OpenCV-ComputerVision-red)

# Smart OCR Extractor (Tesseract + Gradio)

A simple **PDF OCR web app** built with **Gradio + Tesseract**.  
Upload a PDF, apply preprocessing (grayscale / resize / denoise / deskew / threshold), preview the processed result, then extract text from all pages.

## Features
- ✅ Upload **PDF**
- ✅ Convert PDF pages to images (via **Poppler + pdf2image**)
- ✅ Preprocessing pipeline:
  - Grayscale
  - Resize (2x)
  - Denoise (median blur)
  - Deskew (auto-rotation using minAreaRect)
  - Adaptive Threshold
- ✅ Preview:
  - Original first page
  - Processed first page
- ✅ OCR languages:
  - `eng`
  - `ara`
  - `eng+ara`
- ✅ Extract text from **all PDF pages**

## Demo UI
- Upload PDF
- Choose OCR language
- Select preprocessing steps
- Click **Apply Changes** to preview processed page
- Click **Extract Text** to OCR all pages


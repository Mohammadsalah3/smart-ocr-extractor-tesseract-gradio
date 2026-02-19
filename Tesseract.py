import gradio as gr
import pytesseract
from PIL import Image
from pdf2image import convert_from_path
import cv2
import numpy as np

POPPLER_PATH = r"C:\Program Files\poppler-25.12.0\Library\bin"

# Preprocessing
def preprocess_image(pil_image, steps):
    img = np.array(pil_image)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    if "Grayscale" in steps:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    if "Resize" in steps:
        img = cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

    if "Denoise" in steps:
        img = cv2.medianBlur(img, 3)

    if "Deskew" in steps:
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img

        thresh = cv2.threshold(
            gray, 0, 255,
            cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        )[1]

        coords = np.column_stack(np.where(thresh > 0))

        if len(coords) > 0:
            angle = cv2.minAreaRect(coords)[-1]

            if angle < -45:
                angle = 90 + angle

            angle = -angle

            (h, w) = img.shape[:2]
            center = (w // 2, h // 2)

            M = cv2.getRotationMatrix2D(center, angle, 1.0)

            img = cv2.warpAffine(
                img,
                M,
                (w, h),
                flags=cv2.INTER_CUBIC,
                borderMode=cv2.BORDER_REPLICATE
            )

    if "Threshold" in steps:
        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        img = cv2.adaptiveThreshold(
            img,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            11,
            2
        )

    return Image.fromarray(img)


# Handle Upload
def handle_upload(file):
    if file is None:
        return None, None

    filename = file.name.lower()

    if filename.endswith(".pdf"):
        pages = convert_from_path(
            file.name,
            dpi=300,
            poppler_path=POPPLER_PATH
        )
        return pages[0], pages

    else:
        img = Image.open(file.name).convert("RGB")
        return img, None


# Apply Changes 
def apply_changes(original_image, pdf_pages, steps):
    if original_image is None:
        return None

    if pdf_pages:
        return preprocess_image(pdf_pages[0], steps)
    else:
        return preprocess_image(original_image, steps)


# Extract 
def extract_text(original_image, pdf_pages, language, steps):
    if original_image is None:
        return "Upload a file first."

    custom_config = r'--oem 3 --psm 6'

    if pdf_pages:
        full_text = ""
        for page in pdf_pages:
            processed = preprocess_image(page, steps)
            text = pytesseract.image_to_string(
                processed,
                lang=language,
                config=custom_config
            )
            full_text += text + "\n"
        return full_text.strip()
    else:
        processed = preprocess_image(original_image, steps)
        return pytesseract.image_to_string(
            processed,
            lang=language,
            config=custom_config
        ).strip()


# UI
def main():
    with gr.Blocks(title="Smart OCR Extractor (Tesseract)", theme=gr.themes.Soft()) as demo:

        gr.Markdown("# Smart OCR Extractor (Tesseract)")

        pdf_state = gr.State(None)

        with gr.Row():

            with gr.Column(scale=1):

                upload_box = gr.File(
                    file_types=[".pdf"],
                    label="Upload PDF here"
                )

                language = gr.Dropdown(
                    choices=["eng", "ara", "eng+ara"],
                    value="eng",
                    label="OCR Language"
                )

                preprocessing_steps = gr.CheckboxGroup(
                    choices=["Grayscale", "Resize", "Denoise", "Deskew", "Threshold"],
                    value=["Grayscale", "Resize"],
                    label="Preprocessing"
                )

                apply_btn = gr.Button("Apply Changes")
                extract_btn = gr.Button("Extract Text")

            with gr.Column(scale=1):
                original_preview = gr.Image(label="Original Preview")
                processed_preview = gr.Image(label="Processed Preview")

            with gr.Column(scale=1):
                output_text = gr.Textbox(label="Extracted Text", lines=25)

        # Upload
        upload_box.change(
            handle_upload,
            inputs=upload_box,
            outputs=[original_preview, pdf_state]
        )

        # Apply
        apply_btn.click(
            apply_changes,
            inputs=[original_preview, pdf_state, preprocessing_steps],
            outputs=processed_preview
        )

        # Extract
        extract_btn.click(
            extract_text,
            inputs=[original_preview, pdf_state, language, preprocessing_steps],
            outputs=output_text
        )

    demo.launch(inbrowser=True)


if __name__ == "__main__":
    main()
import os
import json
import fitz  # PyMuPDF
from PIL import Image, ImageDraw, ImageFont
from typing import Dict, List, Tuple

def extract_text_info(page: fitz.Page) -> List[Dict]:
    """
    Extract text blocks and their coordinates from a PDF page.
    """
    text_blocks = []
    for block in page.get_text("dict")["blocks"]:
        if "lines" in block:
            for line in block["lines"]:
                for span in line["spans"]:
                    text_blocks.append({
                        "text": span["text"],
                        "bbox": span["bbox"],  # (x0, y0, x1, y1)
                        "font_size": span["size"],
                        "font_name": span["font"]
                    })
    return text_blocks

def convert_pdf_to_images(pdf_path: str, output_dir: str, bg_width: int = 800, bg_height: int = 1280) -> None:
    """
    Convert PDF pages to images with text, placing them on a blank background.
    Also extracts and saves text information with coordinates.
    """
    os.makedirs(output_dir, exist_ok=True)
    text_dir = os.path.join(output_dir, "text_data")
    os.makedirs(text_dir, exist_ok=True)

    pdf_document = fitz.open(pdf_path)
    all_pages_text = {}

    for page_num in range(len(pdf_document)):
        page = pdf_document[page_num]

        # Extract text information
        text_blocks = extract_text_info(page)
        all_pages_text[f"page_{page_num + 1:03d}"] = text_blocks

        # Convert page to image
        pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

        # Calculate scaling to fit width while maintaining aspect ratio
        scale = bg_width / img.width
        new_height = int(img.height * scale)
        img = img.resize((bg_width, new_height), Image.Resampling.LANCZOS)

        # Create and setup background
        background = Image.new('RGB', (bg_width, bg_height), 'white')
        y_position = (bg_height - new_height) // 2
        background.paste(img, (0, y_position))

        # Add page number
        draw = ImageDraw.Draw(background)
        page_text = f"Page {page_num + 1}"
        text_bbox = draw.textbbox((0, 0), page_text)
        text_width = text_bbox[2] - text_bbox[0]
        text_x = bg_width - text_width - 10
        text_y = bg_height - 30
        draw.text((text_x, text_y), page_text, fill='black')

        # Save the image
        output_path = os.path.join(output_dir, f'page_{page_num + 1:03d}.png')
        background.save(output_path, 'PNG')

    # Save text information to JSON
    text_output_path = os.path.join(text_dir, 'text_coordinates.json')
    with open(text_output_path, 'w', encoding='utf-8') as f:
        json.dump(all_pages_text, f, ensure_ascii=False, indent=2)

    pdf_document.close()

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 3:
        print("Usage: python pdf_to_images.py <pdf_path> <output_dir> [width] [height]")
        sys.exit(1)

    pdf_path = sys.argv[1]
    output_dir = sys.argv[2]
    bg_width = int(sys.argv[3]) if len(sys.argv) > 3 else 800
    bg_height = int(sys.argv[4]) if len(sys.argv) > 4 else 1280

    convert_pdf_to_images(pdf_path, output_dir, bg_width, bg_height)

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

def sort_images(images: List[Dict]) -> List[Dict]:
    """
    Sort images by position (top-to-bottom, left-to-right).
    """
    y_threshold = 20  # Consider images within 20 pixels Y distance as same row
    sorted_by_y = sorted(images, key=lambda x: x["bbox"][1])  # Sort by y0

    # Group images by rows based on Y position
    rows = []
    current_row = []
    last_y = None

    for img in sorted_by_y:
        if last_y is None or abs(img["bbox"][1] - last_y) <= y_threshold:
            current_row.append(img)
        else:
            # Sort current row by X coordinate
            current_row.sort(key=lambda x: x["bbox"][0])
            rows.append(current_row)
            current_row = [img]
        last_y = img["bbox"][1]

    if current_row:
        current_row.sort(key=lambda x: x["bbox"][0])
        rows.append(current_row)

    # Flatten rows into final sorted list
    return [img for row in rows for img in row]

def draw_frame(draw: ImageDraw.Draw, bbox: Tuple[float, float, float, float], number: int) -> None:
    """
    Draw a frame around an image and add a number.
    """
    x0, y0, x1, y1 = bbox
    # Draw rectangle frame
    draw.rectangle([x0, y0, x1, y1], outline='red', width=2)
    # Add number
    font_size = 20
    draw.text((x0, y0-font_size-5), str(number), fill='red', font_size=font_size)

def convert_pdf_to_images(pdf_path: str, output_dir: str, bg_width: int = 800, bg_height: int = 1280) -> None:
    """
    Convert PDF pages to images with text, placing them on a blank background.
    Also extracts and saves text information with coordinates.
    """
    os.makedirs(output_dir, exist_ok=True)

    pdf_document = fitz.open(pdf_path)
    all_pages_text = {}

    for page_num in range(len(pdf_document)):
        page = pdf_document[page_num]

        # Extract text information
        text_blocks = extract_text_info(page)
        all_pages_text[f"page_{page_num + 1:03d}"] = text_blocks

        # Extract image information
        image_list = []
        for img in page.get_images():
            try:
                xref = img[0]
                bbox = page.get_image_bbox(xref)
                if bbox and all(isinstance(coord, (int, float)) for coord in bbox):
                    image_list.append({"xref": xref, "bbox": bbox})
            except (ValueError, TypeError, AttributeError) as e:
                print(f"Warning: Could not process image on page {page_num + 1}: {e}")

        # Sort images by position
        sorted_images = sort_images(image_list)

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

        # Draw frames and numbers for images
        draw = ImageDraw.Draw(background)
        for idx, img_info in enumerate(sorted_images, 1):
            bbox = img_info["bbox"]
            # Scale bbox coordinates
            scaled_bbox = (
                bbox[0] * scale,
                bbox[1] * scale + y_position,
                bbox[2] * scale,
                bbox[3] * scale + y_position
            )
            draw_frame(draw, scaled_bbox, idx)

        # Add page number
        page_text = f"Page {page_num + 1}"
        text_bbox = draw.textbbox((0, 0), page_text)
        text_width = text_bbox[2] - text_bbox[0]
        text_x = bg_width - text_width - 10
        text_y = bg_height - 30
        draw.text((text_x, text_y), page_text, fill='black')

        # Save the image
        output_path = os.path.join(output_dir, f'page_{page_num + 1:03d}.png')
        background.save(output_path, 'PNG')

    # Save text information to JSON in output directory
    text_output_path = os.path.join(output_dir, 'text_data.json')
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

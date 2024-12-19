import os
import fitz  # PyMuPDF
from PIL import Image, ImageDraw, ImageFont
import io
import sys
import argparse

def place_on_background(content_image, width=800, height=1280):
    content_width, content_height = content_image.size
    background = Image.new('RGB', (width, height), 'white')
    scale_factor = width / content_width
    new_height = int(content_height * scale_factor)
    content_image = content_image.resize((width, new_height), Image.Resampling.LANCZOS)
    y_position = (height - new_height) // 2
    result = background.copy()
    result.paste(content_image, (0, y_position))
    return result

def extract_text_and_images(pdf_path, output_dir, width=800, height=1280):
    os.makedirs(output_dir, exist_ok=True)
    pdf_document = fitz.open(pdf_path)
    for page_num, page in enumerate(pdf_document, 1):
        rect = page.rect
        pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
        content_img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        draw = ImageDraw.Draw(content_img)
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 24)
        except:
            font = ImageFont.load_default()
        draw.text(
            (50, pix.height - 50),
            f"Page {page_num}",
            fill='black',
            font=font
        )
        final_img = place_on_background(content_img, width, height)
        output_path = os.path.join(output_dir, f"page_{page_num:03d}.png")
        final_img.save(output_path, "PNG")
        print(f"Created image: {output_path}")
    pdf_document.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Convert PDF pages to images with background')
    parser.add_argument('pdf_path', help='Path to the input PDF file')
    parser.add_argument('output_dir', help='Directory to save output images')
    parser.add_argument('--width', type=int, default=800, help='Background width (default: 800)')
    parser.add_argument('--height', type=int, default=1280, help='Background height (default: 1280)')

    args = parser.parse_args()

    if not os.path.exists(args.pdf_path):
        print(f"Error: PDF file not found: {args.pdf_path}")
        sys.exit(1)

    extract_text_and_images(args.pdf_path, args.output_dir, args.width, args.height)

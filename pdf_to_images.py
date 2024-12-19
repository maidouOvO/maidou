import os
import fitz  # PyMuPDF
from PIL import Image, ImageDraw, ImageFont

def convert_pdf_to_images(pdf_path, output_dir, bg_width=800, bg_height=1280):
    """
    Convert PDF pages to images with text, placing them on a blank background.

    Args:
        pdf_path (str): Path to the PDF file
        output_dir (str): Directory to save the output images
        bg_width (int): Width of the background (default: 800)
        bg_height (int): Height of the background (default: 1280)
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Open the PDF
    pdf_document = fitz.open(pdf_path)

    # Process each page
    for page_num in range(len(pdf_document)):
        # Get the page
        page = pdf_document[page_num]

        # Convert page to image
        pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))

        # Create PIL Image from pixmap
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

        # Calculate scaling to fit width while maintaining aspect ratio
        scale = bg_width / img.width
        new_height = int(img.height * scale)

        # Resize image
        img = img.resize((bg_width, new_height), Image.Resampling.LANCZOS)

        # Create blank background
        background = Image.new('RGB', (bg_width, bg_height), 'white')

        # Calculate vertical position to center the image
        y_position = (bg_height - new_height) // 2

        # Paste the image onto the background
        background.paste(img, (0, y_position))

        # Add page number
        draw = ImageDraw.Draw(background)
        page_text = f"Page {page_num + 1}"
        # Position text at bottom right
        text_bbox = draw.textbbox((0, 0), page_text)
        text_width = text_bbox[2] - text_bbox[0]
        text_x = bg_width - text_width - 10
        text_y = bg_height - 30
        draw.text((text_x, text_y), page_text, fill='black')

        # Save the image
        output_path = os.path.join(output_dir, f'page_{page_num + 1:03d}.png')
        background.save(output_path, 'PNG')

    pdf_document.close()

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 3:
        print("Usage: python pdf_to_images.py <pdf_path> <output_dir> [width] [height]")
        sys.exit(1)

    pdf_path = sys.argv[1]
    output_dir = sys.argv[2]

    # Optional width and height parameters
    bg_width = int(sys.argv[3]) if len(sys.argv) > 3 else 800
    bg_height = int(sys.argv[4]) if len(sys.argv) > 4 else 1280

    convert_pdf_to_images(pdf_path, output_dir, bg_width, bg_height)

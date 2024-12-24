"""Utility functions for image processing and manipulation."""
from typing import Tuple, Optional
import io
from PIL import Image
from reportlab.pdfgen import canvas
from .config import BackgroundConfig

def create_background(config: BackgroundConfig) -> Image.Image:
    """Create a white background image with specified dimensions."""
    return Image.new('RGB', config.resolution, 'white')

def resize_image_to_fit(image: Image.Image, max_width: int, max_height: int) -> Image.Image:
    """Resize image to fit within dimensions while maintaining aspect ratio."""
    width_ratio = max_width / image.width
    height_ratio = max_height / image.height
    ratio = min(width_ratio, height_ratio)
    
    new_width = int(image.width * ratio)
    new_height = int(image.height * ratio)
    
    return image.resize((new_width, new_height), Image.Resampling.LANCZOS)

def center_image_on_background(image: Image.Image, background: Image.Image) -> Image.Image:
    """Center an image on the background."""
    bg_width, bg_height = background.size
    img_width, img_height = image.size
    
    x = (bg_width - img_width) // 2
    y = (bg_height - img_height) // 2
    
    result = background.copy()
    result.paste(image, (x, y))
    return result

def merge_images_vertically(image1: Image.Image, image2: Image.Image) -> Image.Image:
    """Merge two images vertically with a small gap between them."""
    gap = 20  # pixels between images
    total_height = image1.height + image2.height + gap
    merged = Image.new('RGB', (max(image1.width, image2.width), total_height), 'white')
    
    # Paste first image at top
    x1 = (merged.width - image1.width) // 2
    merged.paste(image1, (x1, 0))
    
    # Paste second image below with gap
    x2 = (merged.width - image2.width) // 2
    merged.paste(image2, (x2, image1.height + gap))
    
    return merged

def pdf_page_to_image(pdf_page, dpi: int = 200) -> Optional[Image.Image]:
    """Convert a PDF page to PIL Image."""
    try:
        # Convert PDF page to PNG image in memory
        with io.BytesIO() as buffer:
            # Create a canvas with the same dimensions as the PDF page
            width = int(float(pdf_page.mediabox.width))
            height = int(float(pdf_page.mediabox.height))
            c = canvas.Canvas(buffer, pagesize=(width, height))
            
            # Draw the PDF page onto the canvas
            c.drawString(0, 0, " ")  # Ensure at least one operation
            c.save()
            
            # Convert to image
            buffer.seek(0)
            return Image.open(buffer)
    except Exception as e:
        print(f"Error converting PDF page to image: {e}")
        return None

"""PDF processing module."""
import os
import logging
from datetime import datetime
from typing import Tuple, Optional

from PyPDF2 import PdfReader, PdfWriter
from reportlab.pdfgen import canvas
from PIL import Image
import io

from src.pdf_processor.config import BackgroundConfig
from src.pdf_processor.image_utils import resize_image_to_fit

logger = logging.getLogger(__name__)

class PDFProcessor:
    """PDF processing class for handling various PDF modifications."""

    def __init__(self, config: BackgroundConfig):
        """Initialize processor with background configuration."""
        self.config = config
        self.processing_start_time = None

    def add_blank_page(self, pdf_path: str, position: str = 'start') -> str:
        """Add a blank page to the PDF at specified position."""
        logger.info(f"Adding blank page at {position} to {pdf_path}")
        
        # Create blank page using reportlab
        packet = io.BytesIO()
        can = canvas.Canvas(packet)
        can.setPageSize((self.config.width, self.config.height))
        can.showPage()
        can.save()
        packet.seek(0)
        
        # Create PDF with blank page
        blank_pdf = PdfReader(packet)
        existing_pdf = PdfReader(pdf_path)
        output = PdfWriter()
        
        # Add pages in correct order
        if position == 'start':
            output.add_page(blank_pdf.pages[0])
            for page in existing_pdf.pages:
                output.add_page(page)
        else:  # end
            for page in existing_pdf.pages:
                output.add_page(page)
            output.add_page(blank_pdf.pages[0])
        
        # Save modified PDF
        output_path = self._generate_output_path(pdf_path, 'blank')
        with open(output_path, 'wb') as output_file:
            output.write(output_file)
        
        return output_path

    def process_single_pages(self, pdf_path: str) -> str:
        """Process each page individually with background."""
        logger.info(f"Processing single pages for {pdf_path}")
        self.processing_start_time = datetime.now()
        
        reader = PdfReader(pdf_path)
        writer = PdfWriter()
        
        for page_num in range(len(reader.pages)):
            page = reader.pages[page_num]
            # Convert PDF page to image
            page_image = self._convert_page_to_image(page)
            # Resize image to fit background
            resized_image = resize_image_to_fit(page_image, self.config.width, self.config.height)
            # Create new PDF page with background
            new_page = self._create_page_with_background(resized_image)
            writer.add_page(new_page)
        
        output_path = self._generate_output_path(pdf_path, 'single')
        with open(output_path, 'wb') as output_file:
            writer.write(output_file)
        
        return output_path

    def process_merged_pages(self, pdf_path: str) -> str:
        """Merge adjacent pages and process with background."""
        logger.info(f"Processing merged pages for {pdf_path}")
        self.processing_start_time = datetime.now()
        
        reader = PdfReader(pdf_path)
        writer = PdfWriter()
        
        # Process pairs of pages
        for i in range(0, len(reader.pages), 2):
            if i + 1 < len(reader.pages):
                # Convert both pages to images
                image1 = self._convert_page_to_image(reader.pages[i])
                image2 = self._convert_page_to_image(reader.pages[i + 1])
                # Merge images
                merged_image = self._merge_images(image1, image2)
                # Resize merged image
                resized_image = resize_image_to_fit(merged_image, self.config.width, self.config.height)
                # Create new page with background
                new_page = self._create_page_with_background(resized_image)
                writer.add_page(new_page)
            else:
                # Handle odd number of pages
                image = self._convert_page_to_image(reader.pages[i])
                resized_image = resize_image_to_fit(image, self.config.width, self.config.height)
                new_page = self._create_page_with_background(resized_image)
                writer.add_page(new_page)
        
        output_path = self._generate_output_path(pdf_path, 'merged')
        with open(output_path, 'wb') as output_file:
            writer.write(output_file)
        
        return output_path

    def _convert_page_to_image(self, page) -> Image.Image:
        """Convert PDF page to PIL Image."""
        # Implementation would depend on specific PDF library capabilities
        # This is a placeholder for the actual conversion logic
        return Image.new('RGB', (self.config.width, self.config.height), 'white')

    def _create_page_with_background(self, image: Image.Image) -> PdfReader.pages: # type: ignore
        """Create a new PDF page with the image on background."""
        packet = io.BytesIO()
        can = canvas.Canvas(packet, pagesize=(self.config.width, self.config.height))
        
        # Draw background (white by default)
        can.setFillColorRGB(1, 1, 1)  # white
        can.rect(0, 0, self.config.width, self.config.height, fill=1)
        
        # Convert PIL image to bytes
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format='PNG')
        img_byte_arr.seek(0)
        
        # Draw image centered on page
        x = (self.config.width - image.width) / 2
        y = (self.config.height - image.height) / 2
        can.drawImage(img_byte_arr, x, y, image.width, image.height)
        
        can.save()
        packet.seek(0)
        
        return PdfReader(packet).pages[0]

    def _merge_images(self, image1: Image.Image, image2: Image.Image) -> Image.Image:
        """Merge two images side by side."""
        total_width = image1.width + image2.width
        max_height = max(image1.height, image2.height)
        
        merged_image = Image.new('RGB', (total_width, max_height), 'white')
        merged_image.paste(image1, (0, 0))
        merged_image.paste(image2, (image1.width, 0))
        
        return merged_image

    def _generate_output_path(self, input_path: str, suffix: str) -> str:
        """Generate unique output path for processed PDF."""
        directory = os.path.dirname(input_path)
        filename = os.path.basename(input_path)
        name, ext = os.path.splitext(filename)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        return os.path.join(directory, f"{name}_{suffix}_{timestamp}{ext}")

    def generate_processing_report(self, process_type: str,
                                background_resolution: Tuple[int, int],
                                blank_page: Optional[str] = None) -> dict:
        """Generate a report of the processing details."""
        end_time = datetime.now()
        processing_time = (end_time - self.processing_start_time).total_seconds() if self.processing_start_time else 0
        
        return {
            'process_type': process_type,
            'background_resolution': background_resolution,
            'blank_page_added': blank_page,
            'processing_time_seconds': processing_time,
            'timestamp': end_time.isoformat()
        }

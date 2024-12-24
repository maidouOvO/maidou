import os
import io
from pathlib import Path
from typing import Optional, List, Dict, Tuple
import logging
from datetime import datetime

import PyPDF2
from PIL import Image
from reportlab.pdfgen import canvas

from .config import BackgroundConfig
from .image_utils import (
    create_background,
    resize_image_to_fit,
    center_image_on_background,
    merge_images_vertically,
    pdf_page_to_image
)

class PDFProcessor:
    """PDF preprocessing tool for background addition and page merging."""
    
    def __init__(self, config: Optional[BackgroundConfig] = None):
        self.config = config or BackgroundConfig()
        self._setup_logging()
    
    def add_blank_page(self, input_path: str, position: str = "end", output_path: Optional[str] = None) -> str:
        """Add a blank page to the beginning or end of the PDF.
        
        Args:
            input_path: Path to the input PDF file
            position: Where to add the blank page ("start" or "end")
            output_path: Optional path for the output file
            
        Returns:
            Path to the processed PDF file
        """
        try:
            if not output_path:
                output_path = self._generate_output_path(input_path, f"blank_{position}")
            
            self.logger.info(f"Adding blank page at {position} of PDF: {input_path}")
            
            # Create a blank page using reportlab
            packet = io.BytesIO()
            can = canvas.Canvas(packet, pagesize=(612, 792))  # Standard letter size
            can.showPage()  # Required to actually create the page
            can.save()
            packet.seek(0)
            blank_page = PyPDF2.PdfReader(packet).pages[0]
            
            # Process the original PDF
            with open(input_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                pdf_writer = PyPDF2.PdfWriter()
                
                # Add blank page at start if specified
                if position.lower() == "start":
                    pdf_writer.add_page(blank_page)
                
                # Add all pages from original PDF
                for page in pdf_reader.pages:
                    pdf_writer.add_page(page)
                
                # Add blank page at end if specified
                if position.lower() == "end":
                    pdf_writer.add_page(blank_page)
                
                # Save the processed PDF
                with open(output_path, 'wb') as output_file:
                    pdf_writer.write(output_file)
            
            self.logger.info(f"Blank page added successfully. Output saved to: {output_path}")
            return output_path
            
        except Exception as e:
            self.logger.error(f"Error adding blank page: {str(e)}")
            raise
    
    def _setup_logging(self):
        """Initialize logging configuration."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
    
    def process_single_pages(self, input_path: str, output_path: Optional[str] = None) -> str:
        """Process Type 1: Place each page on preset background."""
        try:
            if not output_path:
                output_path = self._generate_output_path(input_path, "single")
            
            self.logger.info(f"Processing PDF: {input_path}")
            self.logger.info(f"Output path: {output_path}")
            
            # Open PDF file
            with open(input_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                total_pages = len(pdf_reader.pages)
                self.logger.info(f"Total pages to process: {total_pages}")
                
                # Create a new PDF for processed pages
                pdf_writer = PyPDF2.PdfWriter()
                
                # Process each page
                for page_num in range(total_pages):
                    self.logger.info(f"Processing page {page_num + 1}/{total_pages}")
                    page = pdf_reader.pages[page_num]
                    
                    # Convert PDF page to image
                    page_image = pdf_page_to_image(page)
                    if not page_image:
                        self.logger.error(f"Failed to convert page {page_num + 1} to image")
                        continue
                    
                    # Create background and process image
                    background = create_background(self.config)
                    processed_image = self._process_single_image(page_image, background)
                    
                    # Convert processed image back to PDF page
                    with io.BytesIO() as image_buffer:
                        processed_image.save(image_buffer, format='PDF')
                        image_buffer.seek(0)
                        new_page = PyPDF2.PdfReader(image_buffer).pages[0]
                        pdf_writer.add_page(new_page)
                
                # Save the processed PDF
                with open(output_path, 'wb') as output_file:
                    pdf_writer.write(output_file)
            
            self.logger.info(f"PDF processing completed. Output saved to: {output_path}")
            return output_path
            
        except Exception as e:
            self.logger.error(f"Error processing PDF: {str(e)}")
            raise
    
    def process_merged_pages(self, input_path: str, output_path: Optional[str] = None) -> str:
        """Process Type 2: Merge adjacent pages and place on background."""
        try:
            if not output_path:
                output_path = self._generate_output_path(input_path, "merged")
            
            self.logger.info(f"Processing PDF with page merging: {input_path}")
            self.logger.info(f"Output path: {output_path}")
            
            # Open PDF file
            with open(input_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                total_pages = len(pdf_reader.pages)
                self.logger.info(f"Total pages to process: {total_pages}")
                
                # Create a new PDF for processed pages
                pdf_writer = PyPDF2.PdfWriter()
                
                # Process pages in pairs
                for page_num in range(0, total_pages, 2):
                    self.logger.info(f"Processing pages {page_num + 1}-{min(page_num + 2, total_pages)}/{total_pages}")
                    
                    # Get first page
                    page1 = pdf_reader.pages[page_num]
                    page1_image = pdf_page_to_image(page1)
                    if not page1_image:
                        self.logger.error(f"Failed to convert page {page_num + 1} to image")
                        continue
                    
                    # Get second page if available
                    page2_image = None
                    if page_num + 1 < total_pages:
                        page2 = pdf_reader.pages[page_num + 1]
                        page2_image = pdf_page_to_image(page2)
                        if not page2_image:
                            self.logger.error(f"Failed to convert page {page_num + 2} to image")
                    
                    # Create background and process images
                    background = create_background(self.config)
                    processed_image = self._process_merged_images(page1_image, page2_image, background)
                    
                    # Convert processed image back to PDF page
                    with io.BytesIO() as image_buffer:
                        processed_image.save(image_buffer, format='PDF')
                        image_buffer.seek(0)
                        new_page = PyPDF2.PdfReader(image_buffer).pages[0]
                        pdf_writer.add_page(new_page)
                
                # Save the processed PDF
                with open(output_path, 'wb') as output_file:
                    pdf_writer.write(output_file)
            
            self.logger.info(f"PDF processing completed. Output saved to: {output_path}")
            return output_path
            
        except Exception as e:
            self.logger.error(f"Error processing PDF: {str(e)}")
            raise
    
    def _process_single_image(self, image: Image.Image, background: Image.Image) -> Image.Image:
        """Process a single image by fitting it to background."""
        # Resize image to fit background width while maintaining aspect ratio
        resized_image = resize_image_to_fit(image, background.width, background.height)
        # Center the resized image on background
        return center_image_on_background(resized_image, background)
    
    def _process_merged_images(
        self,
        image1: Image.Image,
        image2: Optional[Image.Image],
        background: Image.Image
    ) -> Image.Image:
        """Process two images by merging them vertically and fitting to background."""
        if image2 is None:
            # If there's only one image, process it as single
            return self._process_single_image(image1, background)
        
        # Merge images vertically
        merged_image = merge_images_vertically(image1, image2)
        # Fit merged image to background
        return self._process_single_image(merged_image, background)
    
    def _generate_output_path(self, input_path: str, process_type: str) -> str:
        """Generate unique output path to avoid overwriting."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        input_file = Path(input_path)
        return str(input_file.parent / f"{input_file.stem}_{process_type}_{timestamp}{input_file.suffix}")
    
    def generate_processing_report(self, input_path: str, output_path: str, process_type: str) -> Dict:
        """Generate report of processing details."""
        return {
            "timestamp": datetime.now().isoformat(),
            "input_file": input_path,
            "output_file": output_path,
            "process_type": process_type,
            "background_resolution": self.config.resolution,
            "status": "completed"
        }

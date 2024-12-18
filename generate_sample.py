from fpdf import FPDF
import os
from PIL import Image, ImageDraw

class SamplePDFGenerator:
    def __init__(self):
        self.pdf = FPDF()
        self.pdf.set_auto_page_break(auto=True, margin=15)
        self.default_font = 'Helvetica'

    def add_text(self, x, y, text, color=(0,0,0), size=16, align='L'):
        self.pdf.set_text_color(*color)
        self.pdf.set_font(self.default_font, size=size)
        self.pdf.set_xy(x, y)
        self.pdf.cell(0, 10, text, align=align)

    def create_sample(self):
        # Page 1
        self.pdf.add_page()

        # Left side - different colors and alignments
        self.add_text(20, 30, 'Red Left Aligned', color=(255,0,0))
        self.add_text(20, 50, 'Blue Center Text', color=(0,0,255), align='C')
        self.add_text(20, 70, 'Green Right Text', color=(0,255,0), align='R')

        # Right side
        self.add_text(120, 30, 'Purple Left Text', color=(128,0,128))
        self.add_text(120, 50, 'Orange Center', color=(255,165,0), align='C')
        self.add_text(120, 70, 'Black Right', align='R')

        # Page 2
        self.pdf.add_page()

        # Create and add test image
        image_path = os.path.join(os.path.dirname(__file__), 'test_image.png')
        self.create_test_image(image_path)

        # Left side with image
        self.pdf.image(image_path, x=20, y=30, w=60)
        self.add_text(20, 100, 'Image Caption Left', color=(0,0,255))

        # Right side with image
        self.pdf.image(image_path, x=120, y=30, w=60)
        self.add_text(120, 100, 'Image Caption Right', color=(255,0,0))

        # Page 3
        self.pdf.add_page()

        # Left side - paragraphs
        self.add_text(20, 30, 'Paragraph 1 Left', size=12)
        self.add_text(20, 50, 'Paragraph 2 Left', size=12)
        self.add_text(20, 70, 'Paragraph 3 Left', size=12)

        # Right side - paragraphs
        self.add_text(120, 30, 'Paragraph 1 Right', size=12)
        self.add_text(120, 50, 'Paragraph 2 Right', size=12)
        self.add_text(120, 70, 'Paragraph 3 Right', size=12)

        # Save the PDF
        output_path = '1000021_sample.pdf'
        self.pdf.output(output_path)
        print(f"Sample PDF created: {output_path}")

    def create_test_image(self, path):
        width, height = 200, 100
        image = Image.new('RGB', (width, height), 'white')
        draw = ImageDraw.Draw(image)

        draw.rectangle([10, 10, width-10, height-10], outline='black')
        draw.line([10, 10, width-10, height-10], fill='blue', width=2)
        draw.line([10, height-10, width-10, 10], fill='red', width=2)

        image.save(path, 'PNG')

if __name__ == '__main__':
    generator = SamplePDFGenerator()
    generator.create_sample()

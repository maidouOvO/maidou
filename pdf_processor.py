import fitz  # PyMuPDF
import argparse
from typing import Tuple, List, Dict
import pandas as pd
from operator import itemgetter
import os
import re
import json
from PIL import Image
import io

class PDFProcessor:
    def __init__(self, input_pdf: str, target_width: float, target_height: float):
        base_name = os.path.splitext(os.path.basename(input_pdf))[0]
        parts = base_name.split('_')
        self.book_id = parts[0]

        # Handle sample file naming
        if len(parts) > 1 and parts[1] == "sample":
            self.book_name = "the_little_puddle"
        else:
            self.book_name = '_'.join(parts[1:]) if len(parts) > 1 else 'untitled'

        self.main_folder = f"{self.book_id}_{self.book_name}"

        self.create_folder_structure(input_pdf)

        self.pdf_doc = fitz.open(input_pdf)
        self.target_width = target_width
        self.target_height = target_height
        self.center_x = target_width / 2
        self.center_y = target_height / 2
        self.input_pdf = input_pdf

    def create_folder_structure(self, input_pdf):
        os.makedirs(self.main_folder, exist_ok=True)

        book_name_folder = os.path.join(self.main_folder, self.book_name)
        os.makedirs(book_name_folder, exist_ok=True)
        os.makedirs(os.path.join(book_name_folder, 'JPG'), exist_ok=True)

        pdf_dest = os.path.join(book_name_folder, os.path.basename(input_pdf))
        if os.path.abspath(input_pdf) != os.path.abspath(pdf_dest):
            import shutil
            shutil.copy2(input_pdf, pdf_dest)

        book_id_folder = os.path.join(self.main_folder, self.book_id)
        os.makedirs(book_id_folder, exist_ok=True)

        for folder in ['TEXT', 'AUDIO', 'MUSIC', 'VIDEO', 'IMAGE']:
            os.makedirs(os.path.join(book_id_folder, folder), exist_ok=True)

        audio_folder = os.path.join(book_id_folder, 'AUDIO')
        for lang in ['EnglishAudio', 'FranceAudio', 'GermanAudio', 'ItalyAudio', 'SpainAudio']:
            os.makedirs(os.path.join(audio_folder, lang), exist_ok=True)

    def extract_images(self, page_num: int) -> List[Dict]:
        page = self.pdf_doc[page_num]
        image_list = []

        for img_index, img in enumerate(page.get_images(full=True)):
            xref = img[0]
            base_image = self.pdf_doc.extract_image(xref)
            image_bytes = base_image["image"]

            image = Image.open(io.BytesIO(image_bytes))

            image_filename = f"page_{page_num + 1}_image_{img_index + 1}.jpg"
            image_path = os.path.join(
                self.main_folder,
                self.book_name,
                'JPG',
                image_filename
            )
            image.save(image_path)

            image_list.append({
                'page_num': page_num + 1,
                'image_num': img_index + 1,
                'width': image.width,
                'height': image.height,
                'file_path': image_path
            })

        return image_list

    def get_text_color(self, span) -> str:
        if hasattr(span, 'color'):
            color = span.color
            if isinstance(color, tuple):
                return f"rgb({color[0]}, {color[1]}, {color[2]})"
            return str(color)
        return "black"

    def get_text_alignment(self, block) -> str:
        page_width = self.pdf_doc[0].rect.width
        block_width = block[2] - block[0]
        center_x = (block[0] + block[2]) / 2

        if abs(block[0] - 0) < 20:
            return "left"
        elif abs(block[2] - page_width) < 20:
            return "right"
        elif abs(center_x - (page_width / 2)) < 20:
            return "center"
        else:
            return "left"

    def sort_boxes(self, boxes: List[dict]) -> List[dict]:
        """Sort text boxes from top to bottom and left to right."""
        # First, group boxes by their vertical position (with some tolerance)
        tolerance = 10  # pixels
        vertical_groups = {}

        for box in boxes:
            y_center = (box['y0'] + box['y1']) / 2
            grouped = False
            for group_y in vertical_groups.keys():
                if abs(y_center - group_y) < tolerance:
                    vertical_groups[group_y].append(box)
                    grouped = True
                    break
            if not grouped:
                vertical_groups[y_center] = [box]

        # Sort each group horizontally and assign box numbers
        sorted_boxes = []
        box_number = 1

        # Sort groups by vertical position
        for y_center in sorted(vertical_groups.keys()):
            # Sort boxes within group by horizontal position
            group = vertical_groups[y_center]
            group.sort(key=lambda x: x['x0'])

            # Assign box numbers
            for box in group:
                box['box_number'] = box_number
                box_number += 1
                sorted_boxes.append(box)

        return sorted_boxes

    def get_text_boxes(self, page_num: int) -> Dict[str, List[dict]]:
        page = self.pdf_doc[page_num]
        text_blocks = page.get_text("dict")["blocks"]
        left_boxes = []
        right_boxes = []

        # Group text blocks into paragraphs based on proximity
        current_paragraph = []
        last_y = None
        paragraph_spacing = 20  # Adjust based on your needs

        for block in text_blocks:
            if block.get("type") == 0:  # Text block
                for line in block.get("lines", []):
                    for span in line.get("spans", []):
                        box = {
                            'x0': span['bbox'][0],
                            'y0': span['bbox'][1],
                            'x1': span['bbox'][2],
                            'y1': span['bbox'][3],
                            'text': span['text'],
                            'color': self.get_text_color(span),
                            'alignment': self.get_text_alignment(span['bbox']),
                            'font': span.get('font', 'unknown'),
                            'size': span.get('size', 0),
                            'is_paragraph_start': False
                        }

                        # Check if this is the start of a new paragraph
                        if last_y is None or (span['bbox'][1] - last_y) > paragraph_spacing:
                            box['is_paragraph_start'] = True

                        last_y = span['bbox'][3]  # Update last y position

                        if self.is_text_box_in_half(box, 'left'):
                            left_boxes.append(box)
                        else:
                            right_boxes.append(box)

        return {
            'left': self.sort_boxes(left_boxes),
            'right': self.sort_boxes(right_boxes)
        }

    def annotate_text_boxes(self, output_pdf: str):
        os.makedirs(os.path.dirname(output_pdf), exist_ok=True)
        annotated_pdf = fitz.open(self.input_pdf)

        for page_num in range(len(self.pdf_doc)):
            page = annotated_pdf[page_num]
            page_result = self.process_page(page_num)

            # Lemon yellow color with transparency
            yellow_color = (1, 0.98, 0.8)  # RGB normalized to [0,1]

            # Draw vertical line to show page split
            mid_x = page.rect.width / 2
            page.draw_line((mid_x, 0), (mid_x, page.rect.height), color=(0.5, 0.5, 0.5))

            for box in page_result['text_boxes']['left']:
                rect = fitz.Rect(box['x0'], box['y0'], box['x1'], box['y1'])
                # Draw yellow highlight with transparency
                annot = page.add_rect_annot(rect)
                annot.set_colors(stroke=yellow_color)
                annot.set_opacity(0.3)
                annot.update()
                # Draw box number in upper left corner
                text_point = fitz.Point(box['x0'], box['y0'] - 2)
                page.insert_text(text_point, f"L{box['box_number']}", color=(0, 0, 1), fontsize=8)
                # Add paragraph marker if needed
                if box.get('is_paragraph_start', False):
                    page.insert_text(fitz.Point(box['x0'] - 10, box['y0']), "¶", color=(0, 0, 1), fontsize=8)

            for box in page_result['text_boxes']['right']:
                rect = fitz.Rect(box['x0'], box['y0'], box['x1'], box['y1'])
                # Draw yellow highlight with transparency
                annot = page.add_rect_annot(rect)
                annot.set_colors(stroke=yellow_color)
                annot.set_opacity(0.3)
                annot.update()
                # Draw box number in upper left corner
                text_point = fitz.Point(box['x0'], box['y0'] - 2)
                page.insert_text(text_point, f"R{box['box_number']}", color=(0, 0, 1), fontsize=8)
                # Add paragraph marker if needed
                if box.get('is_paragraph_start', False):
                    page.insert_text(fitz.Point(box['x0'] - 10, box['y0']), "¶", color=(0, 0, 1), fontsize=8)

        annotated_pdf.save(output_pdf)
        annotated_pdf.close()

    def calculate_page_numbers(self, physical_page: int) -> Tuple[int, int]:
        base = (physical_page + 1) * 2 + 1
        return (base, base + 1)

    def is_text_box_in_half(self, box: Dict, is_left: bool) -> bool:
        page_width = self.pdf_doc[0].rect.width
        mid_point = page_width / 2
        box_center = (box['x0'] + box['x1']) / 2
        return (box_center <= mid_point) if is_left else (box_center > mid_point)

    def calculate_box_metrics(self, box: Dict) -> Dict:
        width = box['x1'] - box['x0']
        height = box['y1'] - box['y0']
        # Calculate corners relative to screen center
        upper_left = (box['x0'] - self.center_x, box['y0'] - self.center_y)
        upper_right = (box['x1'] - self.center_x, box['y0'] - self.center_y)
        lower_left = (box['x0'] - self.center_x, box['y1'] - self.center_y)
        lower_right = (box['x1'] - self.center_x, box['y1'] - self.center_y)
        return {
            'width': width,
            'height': height,
            'rel_center_x': (box['x0'] + box['x1']) / 2 - self.center_x,  # Maintain backward compatibility
            'rel_center_y': (box['y0'] + box['y1']) / 2 - self.center_y,  # Maintain backward compatibility
            'corners': {
                'upper_left': upper_left,
                'upper_right': upper_right,
                'lower_left': lower_left,
                'lower_right': lower_right
            }
        }

    def scale_page(self, page_num: int) -> Tuple[float, float]:
        page = self.pdf_doc[page_num]
        page_rect = page.rect
        scale_x = self.target_width / page_rect.width
        scale_y = self.target_height / page_rect.height
        scale = min(scale_x, scale_y)
        return scale, scale

    def process_page(self, page_num: int) -> Dict:
        text_boxes = self.get_text_boxes(page_num)
        images = self.extract_images(page_num)
        left_page_num, right_page_num = self.calculate_page_numbers(page_num)

        for i, box in enumerate(text_boxes['left'], 1):
            box['box_number'] = i
            box['book_id'] = self.book_id
            box['page_id'] = left_page_num
            box['text_box_id'] = f"L{i}"
            metrics = self.calculate_box_metrics(box)
            box.update(metrics)

        for i, box in enumerate(text_boxes['right'], 1):
            box['box_number'] = i
            box['book_id'] = self.book_id
            box['page_id'] = right_page_num
            box['text_box_id'] = f"R{i}"
            metrics = self.calculate_box_metrics(box)
            box.update(metrics)

        return {
            'text_boxes': text_boxes,
            'images': images,
            'page_numbers': (left_page_num, right_page_num)
        }

    def process_pdf(self) -> pd.DataFrame:
        all_boxes = []

        for page_num in range(len(self.pdf_doc)):
            page_result = self.process_page(page_num)

            for box in page_result['text_boxes']['left']:
                all_boxes.append({
                    'Physical Page': page_num + 1,
                    'Page Number': page_result['page_numbers'][0],
                    'Side': 'Left',
                    'Box Number': box['box_number'],
                    'Text Content': box['text'],
                    'Book ID': self.book_id,
                    'Page ID': page_result['page_numbers'][0],
                    'Text Box ID': f"L{box['box_number']}",
                    'Width': box['width'],
                    'Height': box['height'],
                    'Relative Center X': box['rel_center_x'],
                    'Relative Center Y': box['rel_center_y'],
                    'Coordinates': f"UL({box['corners']['upper_left'][0]:.1f}, {box['corners']['upper_left'][1]:.1f}) | " \
                                   f"UR({box['corners']['upper_right'][0]:.1f}, {box['corners']['upper_right'][1]:.1f}) | " \
                                   f"LL({box['corners']['lower_left'][0]:.1f}, {box['corners']['lower_left'][1]:.1f}) | " \
                                   f"LR({box['corners']['lower_right'][0]:.1f}, {box['corners']['lower_right'][1]:.1f})",
                    'Text Color': box['color'],
                    'Text Alignment': box['alignment']
                })

            for box in page_result['text_boxes']['right']:
                all_boxes.append({
                    'Physical Page': page_num + 1,
                    'Page Number': page_result['page_numbers'][1],
                    'Side': 'Right',
                    'Box Number': box['box_number'],
                    'Text Content': box['text'],
                    'Book ID': self.book_id,
                    'Page ID': page_result['page_numbers'][1],
                    'Text Box ID': f"R{box['box_number']}",
                    'Width': box['width'],
                    'Height': box['height'],
                    'Relative Center X': box['rel_center_x'],
                    'Relative Center Y': box['rel_center_y'],
                    'Coordinates': f"UL({box['corners']['upper_left'][0]:.1f}, {box['corners']['upper_left'][1]:.1f}) | " \
                                   f"UR({box['corners']['upper_right'][0]:.1f}, {box['corners']['upper_right'][1]:.1f}) | " \
                                   f"LL({box['corners']['lower_left'][0]:.1f}, {box['corners']['lower_left'][1]:.1f}) | " \
                                   f"LR({box['corners']['lower_right'][0]:.1f}, {box['corners']['lower_right'][1]:.1f})",
                    'Text Color': box['color'],
                    'Text Alignment': box['alignment']
                })

        return pd.DataFrame(all_boxes)

    def close(self):
        self.pdf_doc.close()

def main():
    parser = argparse.ArgumentParser(description='Process PDF text boxes')
    parser.add_argument('input_pdf', help='Input PDF file (format: bookID_book_name.pdf)')
    parser.add_argument('--width', type=float, default=600, help='Target width')
    parser.add_argument('--height', type=float, default=800, help='Target height')
    parser.add_argument('--output', help='Output CSV file')
    args = parser.parse_args()

    processor = PDFProcessor(args.input_pdf, args.width, args.height)
    results_df = processor.process_pdf()

    book_name_folder = os.path.join(processor.main_folder, processor.book_name)
    annotated_pdf = os.path.join(book_name_folder, f"{processor.book_id}_{processor.book_name}_annotated.pdf")
    csv_output = args.output or os.path.join(book_name_folder, f"{processor.book_id}_{processor.book_name}_results.csv")

    processor.annotate_text_boxes(annotated_pdf)

    pd.set_option('display.max_colwidth', None)
    print("\nText Box Analysis Results:")
    print(results_df.to_string(index=False))

    results_df.to_csv(csv_output, index=False)
    print(f"\nResults saved to: {csv_output}")
    print(f"Annotated PDF saved to: {annotated_pdf}")
    print(f"Main folder created: {processor.main_folder}")

    processor.close()

if __name__ == '__main__':
    main()

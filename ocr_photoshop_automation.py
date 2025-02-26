#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
OCR and Photoshop Automation Tool

This script recognizes text in images using Tesseract OCR and automates Photoshop
to select and repair text areas based on specific rules:
- Frame/select title text
- Do not frame/select non-continuous text appearing on objects/characters
- Frame/select continuous text passages with similar size and color that form complete sentences
"""

import os
import sys
import time
import argparse
import cv2
import numpy as np
import pytesseract
from PIL import Image

# Import pyautogui conditionally to allow headless testing
try:
    import pyautogui
except (ImportError, KeyError):
    # Create a mock pyautogui for headless environments
    class PyAutoGUIMock:
        def __getattr__(self, name):
            return lambda *args, **kwargs: None
    pyautogui = PyAutoGUIMock()

class OCRPhotoshopAutomation:
    """Main class for OCR and Photoshop automation."""
    
    def __init__(self, image_path, config=None):
        """
        Initialize the OCR and Photoshop automation tool.
        
        Args:
            image_path (str): Path to the input image
            config (dict, optional): Configuration parameters
        """
        self.image_path = image_path
        
        # Default configuration
        self.config = {
            'width': 800,
            'height': 1280,
            'min_title_font_size': 20,
            'min_continuous_text_length': 10,
            'min_text_confidence': 70,
            'ps_lasso_tool_key': 'l',
            'ps_repair_tool_key': 'j',
            'debug': False
        }
        
        # Update with user configuration if provided
        if config:
            self.config.update(config)
            
        # Load the image
        self.image = cv2.imread(image_path)
        if self.image is None:
            raise ValueError(f"Could not load image: {image_path}")
            
        # Resize image if needed
        self.original_height, self.original_width = self.image.shape[:2]
        if self.original_width != self.config['width'] or self.original_height != self.config['height']:
            self.image = cv2.resize(self.image, (self.config['width'], self.config['height']))
            
        # Initialize results
        self.text_regions = []
        self.title_regions = []
        self.continuous_text_regions = []
        
    def detect_text(self):
        """
        Detect text in the image using Tesseract OCR.
        
        Returns:
            list: List of detected text regions with coordinates and text content
        """
        # Convert image to RGB (Tesseract works better with RGB)
        rgb_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
        
        # Use Tesseract to get detailed text information
        custom_config = r'--oem 3 --psm 11'
        data = pytesseract.image_to_data(rgb_image, config=custom_config, output_type=pytesseract.Output.DICT)
        
        # Process the OCR results
        n_boxes = len(data['text'])
        text_regions = []
        
        for i in range(n_boxes):
            # Skip empty text
            if int(data['conf'][i]) < self.config['min_text_confidence'] or not data['text'][i].strip():
                continue
                
            # Get bounding box coordinates
            x, y, w, h = data['left'][i], data['top'][i], data['width'][i], data['height'][i]
            
            # Calculate font size (approximate)
            font_size = h
            
            # Store text region information
            text_region = {
                'text': data['text'][i],
                'x': x,
                'y': y,
                'width': w,
                'height': h,
                'font_size': font_size,
                'confidence': data['conf'][i],
                'block_num': data['block_num'][i],
                'par_num': data['par_num'][i],
                'line_num': data['line_num'][i],
                'word_num': data['word_num'][i]
            }
            
            text_regions.append(text_region)
            
        self.text_regions = text_regions
        return text_regions
    
    def classify_text_regions(self):
        """
        Classify text regions based on the specified rules:
        - Title text
        - Continuous text passages
        - Non-continuous text on objects/characters (to be ignored)
        
        Returns:
            tuple: Lists of title regions and continuous text regions
        """
        # Group text by blocks and lines
        blocks = {}
        for region in self.text_regions:
            block_key = (region['block_num'], region['par_num'])
            if block_key not in blocks:
                blocks[block_key] = []
            blocks[block_key].append(region)
        
        # Also group by lines for better continuous text detection
        lines = {}
        for region in self.text_regions:
            line_key = (region['block_num'], region['par_num'], region['line_num'])
            if line_key not in lines:
                lines[line_key] = []
            lines[line_key].append(region)
        
        title_regions = []
        continuous_text_regions = []
        
        # First, identify title regions (large font, typically at the top)
        for block_key, regions in blocks.items():
            # Sort regions by line and word number
            regions.sort(key=lambda r: (r['line_num'], r['word_num']))
            
            # Calculate average font size for this block
            font_sizes = [r['font_size'] for r in regions]
            avg_font_size = sum(font_sizes) / len(font_sizes) if font_sizes else 0
            
            # Check if this is a title (large font size, typically at the top of the image)
            is_title = (avg_font_size >= self.config['min_title_font_size'] or 
                       (len(regions) <= 3 and regions[0]['y'] < self.image.shape[0] * 0.2 and avg_font_size > 15))
            
            if is_title:
                title_regions.extend(regions)
        
        # Now identify continuous text regions (multiple lines forming paragraphs)
        # Group lines by their vertical position to identify paragraphs
        paragraphs = self._group_lines_into_paragraphs(lines)
        
        for paragraph in paragraphs:
            # Flatten the paragraph into a list of regions
            regions = [region for line in paragraph for region in line]
            
            if len(regions) < self.config['min_continuous_text_length']:
                continue
                
            # Calculate average font size for this paragraph
            font_sizes = [r['font_size'] for r in regions]
            avg_font_size = sum(font_sizes) / len(font_sizes) if font_sizes else 0
            
            # Check if font sizes are similar (continuous text usually has consistent font size)
            font_size_variance = sum((fs - avg_font_size) ** 2 for fs in font_sizes) / len(font_sizes)
            has_consistent_font = font_size_variance < 10  # Threshold for font size consistency
            
            # Check if this paragraph has multiple lines with multiple words
            has_multiple_lines = len(paragraph) > 1
            has_multiple_words_per_line = all(len(line) > 1 for line in paragraph)
            
            # Check if this is continuous text
            is_continuous = (has_multiple_lines and has_consistent_font and 
                            len(regions) >= self.config['min_continuous_text_length'])
            
            # Exclude regions already classified as title
            if is_continuous:
                # Filter out regions that are already in title_regions
                new_regions = [r for r in regions if r not in title_regions]
                continuous_text_regions.extend(new_regions)
        
        self.title_regions = title_regions
        self.continuous_text_regions = continuous_text_regions
        
        return title_regions, continuous_text_regions
        
    def _group_lines_into_paragraphs(self, lines):
        """
        Group lines into paragraphs based on vertical spacing.
        
        Args:
            lines (dict): Dictionary of lines, where each line is a list of regions
            
        Returns:
            list: List of paragraphs, where each paragraph is a list of lines
        """
        # Convert dictionary to list of (line_key, regions) tuples
        line_items = list(lines.items())
        
        # Sort lines by y-coordinate
        line_items.sort(key=lambda item: item[1][0]['y'] if item[1] else 0)
        
        # Group lines into paragraphs
        paragraphs = []
        if not line_items:
            return paragraphs
            
        current_paragraph = [line_items[0][1]]
        prev_y = line_items[0][1][0]['y'] + line_items[0][1][0]['height'] if line_items[0][1] else 0
        
        for _, regions in line_items[1:]:
            if not regions:
                continue
                
            current_y = regions[0]['y']
            
            # If the vertical gap between lines is small, they belong to the same paragraph
            if current_y - prev_y < 2 * regions[0]['height']:
                current_paragraph.append(regions)
            else:
                paragraphs.append(current_paragraph)
                current_paragraph = [regions]
            
            prev_y = current_y + regions[0]['height']
        
        # Add the last paragraph
        if current_paragraph:
            paragraphs.append(current_paragraph)
        
        return paragraphs
        
    def _check_spatial_continuity(self, positions):
        """
        Check if words are spatially continuous (form lines of text).
        
        Args:
            positions (list): List of (x, y) coordinates of words
            
        Returns:
            bool: True if words form continuous lines, False otherwise
        """
        if len(positions) < 2:
            return False
            
        # Sort positions by y-coordinate (line by line)
        sorted_by_y = sorted(positions, key=lambda p: p[1])
        
        # Group positions by lines (words with similar y-coordinates)
        lines = []
        current_line = [sorted_by_y[0]]
        
        for i in range(1, len(sorted_by_y)):
            current_pos = sorted_by_y[i]
            prev_pos = sorted_by_y[i-1]
            
            # If y-coordinates are close, words are on the same line
            if abs(current_pos[1] - prev_pos[1]) < 20:  # Threshold for line height
                current_line.append(current_pos)
            else:
                lines.append(current_line)
                current_line = [current_pos]
        
        # Add the last line
        if current_line:
            lines.append(current_line)
        
        # Check if there are multiple lines with multiple words
        if len(lines) < 2:
            return False
            
        # Check if lines have multiple words
        lines_with_multiple_words = sum(1 for line in lines if len(line) > 1)
        
        return lines_with_multiple_words >= 2  # At least 2 lines with multiple words
    
    def visualize_regions(self, output_path=None):
        """
        Visualize the detected text regions for debugging.
        
        Args:
            output_path (str, optional): Path to save the visualization image
        
        Returns:
            numpy.ndarray: Visualization image
        """
        # Create a copy of the original image for visualization
        vis_image = self.image.copy()
        
        # Draw title regions in red
        for region in self.title_regions:
            x, y, w, h = region['x'], region['y'], region['width'], region['height']
            cv2.rectangle(vis_image, (x, y), (x + w, y + h), (0, 0, 255), 2)
        
        # Draw continuous text regions in green
        for region in self.continuous_text_regions:
            x, y, w, h = region['x'], region['y'], region['width'], region['height']
            cv2.rectangle(vis_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        # Save the visualization if output path is provided
        if output_path:
            cv2.imwrite(output_path, vis_image)
        
        return vis_image
    
    def group_regions_for_selection(self):
        """
        Group regions that should be selected together with the lasso tool.
        
        Returns:
            list: List of region groups for selection
        """
        # Group title regions by proximity
        title_groups = self._group_by_proximity(self.title_regions)
        
        # Group continuous text regions by proximity and line
        continuous_text_groups = self._group_by_proximity(self.continuous_text_regions)
        
        # Combine all groups
        selection_groups = title_groups + continuous_text_groups
        
        return selection_groups
    
    def _group_by_proximity(self, regions, max_distance=20):
        """
        Group regions by proximity.
        
        Args:
            regions (list): List of text regions
            max_distance (int): Maximum distance between regions to be grouped
            
        Returns:
            list: List of region groups
        """
        if not regions:
            return []
            
        # Sort regions by y-coordinate (top to bottom)
        sorted_regions = sorted(regions, key=lambda r: (r['y'], r['x']))
        
        groups = []
        current_group = [sorted_regions[0]]
        
        for i in range(1, len(sorted_regions)):
            current_region = sorted_regions[i]
            prev_region = sorted_regions[i-1]
            
            # Check if regions are on the same line or close enough
            same_line = abs(current_region['y'] - prev_region['y']) < max_distance
            close_horizontally = abs(current_region['x'] - (prev_region['x'] + prev_region['width'])) < max_distance
            
            if same_line and close_horizontally:
                current_group.append(current_region)
            else:
                groups.append(current_group)
                current_group = [current_region]
        
        # Add the last group
        if current_group:
            groups.append(current_group)
        
        return groups
    
    def automate_photoshop(self):
        """
        Automate Photoshop to select and repair text regions.
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Check if Photoshop is open
            ps_window = None
            for window in pyautogui.getAllWindows():
                if "photoshop" in window.title.lower():
                    ps_window = window
                    break
            
            if not ps_window:
                print("Photoshop is not open. Please open Photoshop and the image.")
                return False
            
            # Activate Photoshop window
            ps_window.activate()
            time.sleep(1)
            
            # Open the image in Photoshop if not already open
            # This is a simplified approach; in a real implementation, you might want to check
            # if the image is already open and handle that case
            pyautogui.hotkey('ctrl', 'o')
            time.sleep(1)
            pyautogui.write(os.path.abspath(self.image_path))
            pyautogui.press('enter')
            time.sleep(2)
            
            # Get selection groups
            selection_groups = self.group_regions_for_selection()
            
            # Process each group
            for group in selection_groups:
                # Select lasso tool
                pyautogui.press(self.config['ps_lasso_tool_key'])
                time.sleep(0.5)
                
                # Create a polygon from the group
                points = self._create_polygon_from_group(group)
                
                # Use lasso tool to select the region
                self._use_lasso_tool(points)
                
                # Use repair tool
                pyautogui.press(self.config['ps_repair_tool_key'])
                time.sleep(0.5)
                
                # Click in the center of the selection to apply repair
                center_x = sum(p[0] for p in points) / len(points)
                center_y = sum(p[1] for p in points) / len(points)
                pyautogui.click(center_x, center_y)
                
                # Deselect
                pyautogui.hotkey('ctrl', 'd')
                time.sleep(0.5)
            
            return True
            
        except Exception as e:
            print(f"Error automating Photoshop: {e}")
            return False
    
    def _create_polygon_from_group(self, group):
        """
        Create a polygon from a group of regions.
        
        Args:
            group (list): List of regions in a group
            
        Returns:
            list: List of points forming a polygon
        """
        # Find the bounding box of the group
        min_x = min(r['x'] for r in group)
        min_y = min(r['y'] for r in group)
        max_x = max(r['x'] + r['width'] for r in group)
        max_y = max(r['y'] + r['height'] for r in group)
        
        # Add some padding
        padding = 5
        min_x = max(0, min_x - padding)
        min_y = max(0, min_y - padding)
        max_x = min(self.config['width'], max_x + padding)
        max_y = min(self.config['height'], max_y + padding)
        
        # Create a simple rectangle polygon
        points = [
            (min_x, min_y),
            (max_x, min_y),
            (max_x, max_y),
            (min_x, max_y)
        ]
        
        return points
    
    def _use_lasso_tool(self, points):
        """
        Use the lasso tool to select a region defined by points.
        
        Args:
            points (list): List of (x, y) coordinates
        """
        # Move to the first point and click to start the lasso selection
        pyautogui.moveTo(points[0][0], points[0][1])
        pyautogui.mouseDown()
        
        # Move through all other points
        for x, y in points[1:]:
            pyautogui.moveTo(x, y, duration=0.1)
        
        # Close the selection by returning to the first point
        pyautogui.moveTo(points[0][0], points[0][1], duration=0.1)
        
        # Release the mouse button to complete the selection
        pyautogui.mouseUp()
        time.sleep(0.5)

def main():
    """Main function to run the OCR and Photoshop automation tool."""
    parser = argparse.ArgumentParser(description='OCR and Photoshop Automation Tool')
    parser.add_argument('image_path', help='Path to the input image')
    parser.add_argument('--width', type=int, default=800, help='Width for image processing')
    parser.add_argument('--height', type=int, default=1280, help='Height for image processing')
    parser.add_argument('--min-title-font-size', type=int, default=20, help='Minimum font size for title text')
    parser.add_argument('--min-continuous-text-length', type=int, default=10, help='Minimum length for continuous text')
    parser.add_argument('--min-text-confidence', type=int, default=70, help='Minimum confidence for text detection')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    parser.add_argument('--visualize', help='Path to save visualization image')
    parser.add_argument('--no-ps', action='store_true', help='Skip Photoshop automation')
    
    args = parser.parse_args()
    
    # Create configuration
    config = {
        'width': args.width,
        'height': args.height,
        'min_title_font_size': args.min_title_font_size,
        'min_continuous_text_length': args.min_continuous_text_length,
        'min_text_confidence': args.min_text_confidence,
        'debug': args.debug
    }
    
    try:
        # Initialize the tool
        tool = OCRPhotoshopAutomation(args.image_path, config)
        
        # Detect text
        print("Detecting text...")
        tool.detect_text()
        
        # Classify text regions
        print("Classifying text regions...")
        title_regions, continuous_text_regions = tool.classify_text_regions()
        
        print(f"Found {len(title_regions)} title regions and {len(continuous_text_regions)} continuous text regions")
        
        # Visualize if requested
        if args.visualize:
            print(f"Saving visualization to {args.visualize}...")
            tool.visualize_regions(args.visualize)
        
        # Automate Photoshop if not disabled
        if not args.no_ps:
            print("Automating Photoshop...")
            success = tool.automate_photoshop()
            if success:
                print("Photoshop automation completed successfully")
            else:
                print("Photoshop automation failed")
        
        print("Done!")
        return 0
        
    except Exception as e:
        print(f"Error: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())

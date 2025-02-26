#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Test script for OCR functionality without Photoshop automation.
This is useful for testing the text detection and classification
without needing to have Photoshop open.
"""

import sys
import argparse
import cv2
import os
import importlib.util

# Temporarily modify sys.modules to avoid importing pyautogui
# This allows us to test OCR functionality in headless environments
sys.modules['pyautogui'] = None

# Import OCRPhotoshopAutomation class directly from the file
spec = importlib.util.spec_from_file_location("ocr_module", 
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "ocr_photoshop_automation.py"))
ocr_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ocr_module)
OCRPhotoshopAutomation = ocr_module.OCRPhotoshopAutomation

def main():
    """Main function to test OCR functionality."""
    parser = argparse.ArgumentParser(description='Test OCR Functionality')
    parser.add_argument('image_path', help='Path to the input image')
    parser.add_argument('--width', type=int, default=800, help='Width for image processing')
    parser.add_argument('--height', type=int, default=1280, help='Height for image processing')
    parser.add_argument('--min-title-font-size', type=int, default=20, help='Minimum font size for title text')
    parser.add_argument('--min-continuous-text-length', type=int, default=10, help='Minimum length for continuous text')
    parser.add_argument('--min-text-confidence', type=int, default=70, help='Minimum confidence for text detection')
    parser.add_argument('--output', required=True, help='Path to save visualization image')
    
    args = parser.parse_args()
    
    # Create configuration
    config = {
        'width': args.width,
        'height': args.height,
        'min_title_font_size': args.min_title_font_size,
        'min_continuous_text_length': args.min_continuous_text_length,
        'min_text_confidence': args.min_text_confidence,
        'debug': True
    }
    
    try:
        # Initialize the tool
        tool = OCRPhotoshopAutomation(args.image_path, config)
        
        # Detect text
        print("Detecting text...")
        text_regions = tool.detect_text()
        print(f"Found {len(text_regions)} text regions")
        
        # Classify text regions
        print("Classifying text regions...")
        title_regions, continuous_text_regions = tool.classify_text_regions()
        
        print(f"Found {len(title_regions)} title regions and {len(continuous_text_regions)} continuous text regions")
        
        # Print details of detected regions
        print("\nTitle Regions:")
        for i, region in enumerate(title_regions):
            print(f"  {i+1}. '{region['text']}' at ({region['x']}, {region['y']}) with size {region['font_size']}")
        
        print("\nContinuous Text Regions:")
        for i, region in enumerate(continuous_text_regions):
            print(f"  {i+1}. '{region['text']}' at ({region['x']}, {region['y']}) with size {region['font_size']}")
        
        # Visualize
        print(f"Saving visualization to {args.output}...")
        vis_image = tool.visualize_regions(args.output)
        
        # Skip displaying the image in headless environments
        print("Visualization saved successfully. Image display skipped in headless environment.")
        
        print("Done!")
        return 0
        
    except Exception as e:
        print(f"Error: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())

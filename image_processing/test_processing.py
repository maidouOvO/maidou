#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import cv2
import numpy as np
from PIL import Image
import pytesseract

def simple_text_removal(image):
    """Simple text removal for testing"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                  cv2.THRESH_BINARY_INV, 11, 2)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    dilate = cv2.dilate(thresh, kernel, iterations=2)
    mask = np.zeros_like(gray)
    cv2.rectangle(mask, (50, 50), (150, 150), 255, -1)
    result = cv2.inpaint(image, mask, 3, cv2.INPAINT_TELEA)
    return result

def has_text(image):
    """Simple text detection for testing"""
    pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    text = pytesseract.image_to_string(pil_image, lang='eng')
    return len(text.strip()) > 3

def main():
    # Read a small test image
    image = cv2.imread("test_images/test_image1.png")
    if image is None:
        print("Error: Could not read image")
        return
    
    # Process the image
    processed = simple_text_removal(image)
    
    # Check if it still has text
    has_text_result = has_text(processed)
    
    # Save the result
    os.makedirs("processed_images/folder_a", exist_ok=True)
    os.makedirs("processed_images/folder_b", exist_ok=True)
    
    output_folder = "processed_images/folder_a" if has_text_result else "processed_images/folder_b"
    output_path = os.path.join(output_folder, "test_result.png")
    cv2.imwrite(output_path, processed)
    
    print(f"Image processed and saved to: {output_path}")
    print(f"Text detection result: {'Has text' if has_text_result else 'No text'}")

if __name__ == "__main__":
    main()

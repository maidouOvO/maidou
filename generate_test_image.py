#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Generate a test image with different types of text for OCR testing.
"""

import cv2
import numpy as np

def main():
    # Create a blank image with white background
    img = np.ones((1280, 800, 3), dtype=np.uint8) * 255

    # Add title text
    cv2.putText(img, 'Title Text Example', (200, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 3)

    # Add text on objects (non-continuous)
    cv2.rectangle(img, (100, 250), (200, 350), (200, 200, 200), -1)  # Draw a gray box
    cv2.putText(img, 'Text', (110, 300), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    
    cv2.rectangle(img, (300, 300), (400, 400), (200, 200, 200), -1)  # Draw a gray box
    cv2.putText(img, 'on', (320, 350), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    
    cv2.rectangle(img, (500, 350), (650, 450), (200, 200, 200), -1)  # Draw a gray box
    cv2.putText(img, 'objects', (510, 400), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

    # Add continuous text passage (more densely packed to help OCR recognize it as continuous)
    y_pos = 600
    for i, line in enumerate([
        'This is a continuous text passage that should be',
        'selected by the OCR tool. The text has similar font',
        'size and color, forming complete sentences that',
        'would appear in a book or document.'
    ]):
        # Draw each word closer together to help OCR recognize it as continuous text
        words = line.split()
        x_pos = 50
        for word in words:
            cv2.putText(img, word, (x_pos, y_pos + i*40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
            # Calculate next word position based on word length
            x_pos += len(word) * 10 + 15  # Adjust spacing between words

    # Save the image
    cv2.imwrite('test_image.jpg', img)
    print('Created test image: test_image.jpg')

if __name__ == "__main__":
    main()

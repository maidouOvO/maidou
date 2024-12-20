import os
import json
from typing import Dict

def analyze_text_data(output_dir: str) -> None:
    """Analyze the text data extracted from PDF pages."""
    # Check directory structure
    print("Directory structure:")
    for root, dirs, files in os.walk(output_dir):
        level = root.replace(output_dir, '').count(os.sep)
        indent = ' ' * 4 * level
        print(f"{indent}{os.path.basename(root)}/")
        subindent = ' ' * 4 * (level + 1)
        for f in files:
            print(f"{subindent}{f}")

    # Analyze text coordinates
    text_data_path = os.path.join(output_dir, "text_data", "text_coordinates.json")
    with open(text_data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print("\nText analysis:")
    print("-" * 40)
    for page, blocks in data.items():
        print(f"\n{page}:")
        print(f"Number of text blocks: {len(blocks)}")
        if blocks:
            # Get coordinate ranges
            x_coords = [coord for block in blocks for coord in block['bbox'][::2]]
            y_coords = [coord for block in blocks for coord in block['bbox'][1::2]]
            print(f"X coordinate range: {min(x_coords):.2f} to {max(x_coords):.2f}")
            print(f"Y coordinate range: {min(y_coords):.2f} to {max(y_coords):.2f}")
            # Show sample text
            print("Sample text blocks:")
            for block in blocks[:3]:  # Show first 3 blocks
                print(f"- '{block['text']}' at {block['bbox']}")

if __name__ == "__main__":
    analyze_text_data("output_images")

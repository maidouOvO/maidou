import fitz

def analyze_annotations(pdf_path):
    doc = fitz.open(pdf_path)
    print("\nAnalyzing PDF Annotations:")
    print("-" * 50)

    for page in doc:
        print(f'\nPage {page.number + 1} Annotations:')
        annotations = list(page.annots())

        # Sort annotations by vertical position (top to bottom)
        annotations.sort(key=lambda x: x.rect[1])

        for idx, annot in enumerate(annotations, 1):
            color_str = f"RGB: {annot.colors['stroke']}" if annot.colors.get('stroke') else "No color"
            position = f"Top-left: ({annot.rect[0]:.2f}, {annot.rect[1]:.2f})"
            print(f"Annotation {idx}:")
            print(f"  Type: {annot.type}")
            print(f"  Color: {color_str}")
            print(f"  Position: {position}")
            print(f"  Content: {annot.info.get('content', 'No content')}")

    doc.close()

if __name__ == "__main__":
    analyze_annotations("sample_untitled/untitled/sample_untitled_annotated.pdf")

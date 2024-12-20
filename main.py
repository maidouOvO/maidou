import argparse
from pdf_to_images import convert_pdf_to_images
from cli_editor import main as start_cli_editor

def main():
    parser = argparse.ArgumentParser(description='PDF Processing and Local CLI Editor')
    parser.add_argument('--pdf', help='PDF file to process')
    parser.add_argument('--output', help='Output directory', default='output_images')
    parser.add_argument('--width', type=int, default=800, help='Background width')
    parser.add_argument('--height', type=int, default=1280, help='Background height')
    parser.add_argument('--edit', action='store_true', help='Start local CLI editor')

    args = parser.parse_args()

    if args.pdf:
        print(f"Processing PDF: {args.pdf}")
        convert_pdf_to_images(args.pdf, args.output, args.width, args.height)
        print(f"PDF processed. Images saved to: {args.output}")

    if args.edit:
        print("Starting local CLI editor...")
        start_cli_editor(args.output)

    if not args.pdf and not args.edit:
        parser.print_help()

if __name__ == '__main__':
    main()

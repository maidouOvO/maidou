import argparse
from pdf_to_images import convert_pdf_to_images
from app import app

def main():
    parser = argparse.ArgumentParser(description='PDF Processing and Web Interface')
    parser.add_argument('--pdf', help='PDF file to process')
    parser.add_argument('--output', help='Output directory', default='output_images')
    parser.add_argument('--width', type=int, default=800, help='Background width')
    parser.add_argument('--height', type=int, default=1280, help='Background height')
    parser.add_argument('--serve', action='store_true', help='Start web server')

    args = parser.parse_args()

    if args.pdf:
        print(f"Processing PDF: {args.pdf}")
        convert_pdf_to_images(args.pdf, args.output, args.width, args.height)
        print(f"PDF processed. Images saved to: {args.output}")

    if args.serve:
        print("Starting web server on http://localhost:5000")
        app.run(debug=True, port=5000)

    if not args.pdf and not args.serve:
        parser.print_help()

if __name__ == '__main__':
    main()

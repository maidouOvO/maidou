# Maidou PDF Tool

A powerful tool for extracting and editing text from PDF files. This tool allows you to:
- Convert PDF pages to images with configurable background dimensions
- Extract and identify text content with precise coordinates
- Edit text boxes through an intuitive command-line interface
- Process images with automatic centering and scaling

## Installation

```bash
# Install from GitHub
pip install git+https://github.com/maidouOvO/maidou.git

# Or clone and install locally
git clone https://github.com/maidouOvO/maidou.git
cd maidou
pip install -e .
```

## Usage

1. Process a PDF file:
```bash
maidou-pdf --pdf your_file.pdf --output output_directory
```

2. Start the CLI editor:
```bash
maidou-pdf --edit
```

### CLI Editor Commands
- 'n': Navigate to next page
- 'p': Navigate to previous page
- 'd <number>': Delete text box (e.g., 'd 1' to delete text box #1)
- 'q': Quit editor

## Configuration

You can customize the background dimensions (default: 800x1280):
```bash
maidou-pdf --pdf input.pdf --output output --width 1000 --height 1500
```

## Requirements
- Python 3.6 or higher
- Dependencies (automatically installed):
  - PyMuPDF: PDF processing
  - Pillow: Image processing
  - pandas: Data handling
  - fpdf: PDF generation

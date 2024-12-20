import os
import flask
import fitz
from PIL import Image, ImageDraw

def check_environment():
    print('Package Versions:')
    print('----------------')
    print(f'Flask version: {flask.__version__}')
    print(f'PyMuPDF version: {fitz.__version__}')
    print(f'Pillow version: {Image.__version__}')
    
    print('\nDirectory Structure:')
    print('------------------')
    for root, dirs, files in os.walk('.'):
        level = root.replace('.', '').count(os.sep)
        indent = ' ' * 4 * level
        print(f'{indent}{os.path.basename(root) or "."}/') 
        subindent = ' ' * 4 * (level + 1)
        for f in files:
            print(f'{subindent}{f}')

if __name__ == '__main__':
    check_environment()

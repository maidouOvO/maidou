"""PDF processing routes."""
import os
from flask import Blueprint, request, jsonify, current_app
from werkzeug.utils import secure_filename
from src.pdf_processor.processor import PDFProcessor
from src.pdf_processor.config import BackgroundConfig

bp = Blueprint('pdf', __name__, url_prefix='/api/pdf')

ALLOWED_EXTENSIONS = {'pdf'}

def allowed_file(filename):
    """Check if file extension is allowed."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@bp.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({'status': 'healthy'})

@bp.route('/process', methods=['POST'])
def process_pdf():
    """Process PDF file endpoint."""
    # Check if file was uploaded
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if not file or not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type'}), 400
    
    try:
        # Get processing parameters
        width = int(request.form.get('width', 800))
        height = int(request.form.get('height', 1280))
        process_type = request.form.get('process_type', 'single')
        blank_page = request.form.get('blank_page')  # 'start' or 'end' or None
        
        # Create temporary directory for processing
        upload_dir = os.path.join(current_app.instance_path, 'uploads')
        os.makedirs(upload_dir, exist_ok=True)
        
        # Save uploaded file
        filename = secure_filename(file.filename)
        filepath = os.path.join(upload_dir, filename)
        file.save(filepath)
        
        # Initialize processor with background config
        config = BackgroundConfig(width=width, height=height)
        processor = PDFProcessor(config)
        
        # Add blank page if requested
        if blank_page in ['start', 'end']:
            filepath = processor.add_blank_page(filepath, blank_page)
        
        # Process PDF based on type
        if process_type == 'merged':
            output_path = processor.process_merged_pages(filepath)
        else:
            output_path = processor.process_single_pages(filepath)
        
        # Generate processing report
        report = processor.generate_processing_report(
            process_type=process_type,
            background_resolution=(width, height),
            blank_page=blank_page
        )
        
        return jsonify({
            'message': 'PDF processed successfully',
            'output_path': output_path,
            'report': report
        })
    
    except ValueError as e:
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        current_app.logger.error(f"Error processing PDF: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500
    finally:
        # Clean up temporary files
        if os.path.exists(filepath):
            os.remove(filepath)

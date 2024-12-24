"""Routes for PDF preprocessing service."""
import os
from flask import Blueprint, request, jsonify, current_app
from werkzeug.utils import secure_filename
from ..pdf_processor import PDFProcessor, BackgroundConfig

bp = Blueprint('pdf', __name__, url_prefix='/api/pdf')

ALLOWED_EXTENSIONS = {'pdf'}

def allowed_file(filename):
    """Check if the file extension is allowed."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@bp.route('/process', methods=['POST'])
def process_pdf():
    """Process PDF file according to specified parameters."""
    # Check if file was uploaded
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type. Only PDF files are allowed'}), 400
    
    try:
        # Get processing parameters
        process_type = request.form.get('process_type', 'single')
        if process_type not in ['single', 'merged']:
            return jsonify({'error': 'Invalid process type'}), 400
        
        # Get background configuration
        width = int(request.form.get('width', 800))
        height = int(request.form.get('height', 1280))
        
        # Get blank page option
        blank_page = request.form.get('blank_page')
        if blank_page and blank_page not in ['start', 'end']:
            return jsonify({'error': 'Invalid blank page position'}), 400
        
        # Save uploaded file
        filename = secure_filename(file.filename)
        upload_path = os.path.join('/tmp', filename)
        file.save(upload_path)
        
        # Initialize processor with configuration
        config = BackgroundConfig(width=width, height=height)
        processor = PDFProcessor(config)
        
        # Add blank page if requested
        if blank_page:
            upload_path = processor.add_blank_page(upload_path, blank_page)
        
        # Process the PDF
        if process_type == 'single':
            output_path = processor.process_single_pages(upload_path)
        else:
            output_path = processor.process_merged_pages(upload_path)
        
        # Generate processing report
        report = processor.generate_processing_report(
            upload_path,
            output_path,
            process_type
        )
        
        # Clean up temporary file
        os.remove(upload_path)
        
        return jsonify({
            'message': 'PDF processed successfully',
            'report': report,
            'output_path': output_path
        }), 200
        
    except Exception as e:
        current_app.logger.error(f"Error processing PDF: {str(e)}")
        return jsonify({'error': str(e)}), 500

@bp.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({'status': 'healthy'}), 200

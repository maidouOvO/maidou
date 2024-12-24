"""PDF processing routes."""
import os
from flask import Blueprint, request, jsonify, send_file
from werkzeug.utils import secure_filename
from src.pdf_processor.config import BackgroundConfig
from src.pdf_processor.processor import PDFProcessor

bp = Blueprint('pdf', __name__, url_prefix='/api/pdf')

ALLOWED_EXTENSIONS = {'pdf'}

def allowed_file(filename):
    """Check if file extension is allowed."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@bp.route('/process', methods=['POST'])
def process_pdf():
    """Process PDF file with specified configuration."""
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if not file or not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type'}), 400

    try:
        filename = secure_filename(file.filename)
        process_type = request.form.get('process_type', 'single')
        width = int(request.form.get('width', 800))
        height = int(request.form.get('height', 1280))
        blank_page = request.form.get('blank_page', None)
        
        upload_path = os.path.join('/tmp', filename)
        file.save(upload_path)
        
        # Initialize processor with configuration
        config = BackgroundConfig(width=width, height=height)
        processor = PDFProcessor(config)
        
        # Add blank page if requested
        if blank_page:
            processor.add_blank_page(upload_path, blank_page)
        
        # Process PDF based on type
        if process_type == 'single':
            output_path = processor.process_single_pages(upload_path)
        elif process_type == 'merged':
            output_path = processor.process_merged_pages(upload_path)
        else:
            return jsonify({'error': 'Invalid process type'}), 400
        
        # Generate processing report
        report = processor.generate_processing_report(
            process_type=process_type,
            background_resolution=(width, height),
            blank_page=blank_page
        )
        
        return send_file(
            output_path,
            as_attachment=True,
            download_name=f'processed_{filename}',
            mimetype='application/pdf'
        )
    
    except Exception as e:
        return jsonify({'error': f'Error processing PDF: {str(e)}'}), 500

@bp.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({'status': 'healthy'}), 200

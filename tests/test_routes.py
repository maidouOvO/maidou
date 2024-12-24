"""Test cases for PDF processing routes."""
import os
import json
from io import BytesIO

def test_health_check(client):
    """Test the health check endpoint."""
    response = client.get('/api/pdf/health')
    assert response.status_code == 200
    assert response.json['status'] == 'healthy'

def test_process_pdf_no_file(client):
    """Test PDF processing without file."""
    response = client.post('/api/pdf/process')
    assert response.status_code == 400
    assert 'error' in response.json
    assert response.json['error'] == 'No file provided'

def test_process_pdf_empty_filename(client):
    """Test PDF processing with empty filename."""
    response = client.post('/api/pdf/process', data={
        'file': (BytesIO(), '')
    })
    assert response.status_code == 400
    assert 'error' in response.json
    assert response.json['error'] == 'No file selected'

def test_process_pdf_invalid_type(client):
    """Test PDF processing with invalid file type."""
    response = client.post('/api/pdf/process', data={
        'file': (BytesIO(b'not a pdf'), 'test.txt')
    })
    assert response.status_code == 400
    assert 'error' in response.json
    assert response.json['error'] == 'Invalid file type. Only PDF files are allowed'

def test_process_pdf_single_page(client, sample_pdf):
    """Test single page PDF processing."""
    with open(sample_pdf, 'rb') as f:
        data = {
            'process_type': 'single',
            'width': '800',
            'height': '1280'
        }
        response = client.post('/api/pdf/process', 
                             data=data,
                             content_type='multipart/form-data',
                             buffered=True,
                             files={'file': (sample_pdf, f, 'application/pdf')})
        
        assert response.status_code == 200
        assert 'message' in response.json
        assert response.json['message'] == 'PDF processed successfully'
        assert 'output_path' in response.json
        assert os.path.exists(response.json['output_path'])

def test_process_pdf_merged_pages(client, sample_pdf):
    """Test merged pages PDF processing."""
    with open(sample_pdf, 'rb') as f:
        data = {
            'process_type': 'merged',
            'width': '800',
            'height': '1280'
        }
        response = client.post('/api/pdf/process',
                             data=data,
                             content_type='multipart/form-data',
                             buffered=True,
                             files={'file': (sample_pdf, f, 'application/pdf')})
        
        assert response.status_code == 200
        assert 'message' in response.json
        assert response.json['message'] == 'PDF processed successfully'
        assert 'output_path' in response.json
        assert os.path.exists(response.json['output_path'])

def test_process_pdf_with_blank_page(client, sample_pdf):
    """Test PDF processing with blank page addition."""
    with open(sample_pdf, 'rb') as f:
        data = {
            'process_type': 'single',
            'width': '800',
            'height': '1280',
            'blank_page': 'start'
        }
        response = client.post('/api/pdf/process',
                             data=data,
                             content_type='multipart/form-data',
                             buffered=True,
                             files={'file': (sample_pdf, f, 'application/pdf')})
        
        assert response.status_code == 200
        assert 'message' in response.json
        assert response.json['message'] == 'PDF processed successfully'
        assert 'output_path' in response.json
        assert os.path.exists(response.json['output_path'])

def test_process_pdf_invalid_blank_page_position(client, sample_pdf):
    """Test PDF processing with invalid blank page position."""
    with open(sample_pdf, 'rb') as f:
        data = {
            'process_type': 'single',
            'blank_page': 'invalid'
        }
        response = client.post('/api/pdf/process',
                             data=data,
                             content_type='multipart/form-data',
                             buffered=True,
                             files={'file': (sample_pdf, f, 'application/pdf')})
        
        assert response.status_code == 400
        assert 'error' in response.json
        assert response.json['error'] == 'Invalid blank page position'

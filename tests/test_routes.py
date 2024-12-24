"""Test PDF processing routes."""
import io
import pytest
from werkzeug.datastructures import FileStorage

def test_health_check(client):
    """Test health check endpoint."""
    response = client.get('/api/pdf/health')
    assert response.status_code == 200
    assert response.json['status'] == 'healthy'

def test_process_pdf_no_file(client):
    """Test PDF processing without file."""
    response = client.post('/api/pdf/process')
    assert response.status_code == 400
    assert 'error' in response.json

def test_process_pdf_empty_filename(client):
    """Test PDF processing with empty filename."""
    data = {'file': (io.BytesIO(b''), '')}
    response = client.post('/api/pdf/process', data=data)
    assert response.status_code == 400
    assert 'error' in response.json

def test_process_pdf_invalid_type(client):
    """Test PDF processing with invalid file type."""
    data = {'file': (io.BytesIO(b'not a pdf'), 'test.txt')}
    response = client.post('/api/pdf/process', data=data)
    assert response.status_code == 400
    assert 'error' in response.json

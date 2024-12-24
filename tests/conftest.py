"""Test configuration and fixtures."""
import os
import pytest
from src.web import create_app

@pytest.fixture
def app():
    """Create and configure a test Flask application."""
    app = create_app({
        'TESTING': True,
    })
    return app

@pytest.fixture
def client(app):
    """Create a test client for the Flask application."""
    return app.test_client()

@pytest.fixture
def sample_pdf():
    """Create a sample PDF file for testing."""
    from reportlab.pdfgen import canvas
    from reportlab.lib.pagesizes import letter
    
    # Create a sample PDF with two pages
    filename = '/tmp/test.pdf'
    c = canvas.Canvas(filename, pagesize=letter)
    
    # First page
    c.drawString(100, 750, "Test Page 1")
    c.showPage()
    
    # Second page
    c.drawString(100, 750, "Test Page 2")
    c.save()
    
    yield filename
    
    # Cleanup
    if os.path.exists(filename):
        os.remove(filename)

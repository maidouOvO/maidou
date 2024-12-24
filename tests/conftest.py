"""Test configuration and fixtures."""
import pytest
from src.web import create_app

@pytest.fixture
def app():
    """Create and configure a Flask application for testing."""
    app = create_app()
    app.config['TESTING'] = True
    return app

@pytest.fixture
def client(app):
    """Create a test client for the application."""
    return app.test_client()

@pytest.fixture
def runner(app):
    """Create a CLI runner for the application."""
    return app.test_cli_runner()

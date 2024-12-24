"""Web application package for PDF preprocessing service."""
from flask import Flask
from flask.logging import default_handler
import logging

def create_app(config=None):
    """Create and configure the Flask application."""
    app = Flask(__name__)
    
    # Configure logging
    app.logger.removeHandler(default_handler)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    app.logger.addHandler(logging.StreamHandler())
    
    # Load configuration
    if config:
        app.config.from_mapping(config)
    
    # Register blueprints
    from .routes import bp as pdf_bp
    app.register_blueprint(pdf_bp)
    
    return app

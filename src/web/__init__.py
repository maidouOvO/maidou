"""Web application package."""
from flask import Flask

def create_app():
    """Create and configure the Flask application."""
    app = Flask(__name__)
    
    from src.web.routes import bp
    app.register_blueprint(bp)
    
    return app

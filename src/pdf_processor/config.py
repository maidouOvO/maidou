"""Configuration module for PDF processing."""
from dataclasses import dataclass

@dataclass
class BackgroundConfig:
    """Configuration for PDF background processing."""
    width: int
    height: int

    def __post_init__(self):
        """Validate configuration values."""
        if self.width <= 0 or self.height <= 0:
            raise ValueError("Width and height must be positive integers")

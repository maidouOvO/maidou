from dataclasses import dataclass
from typing import Tuple

@dataclass
class BackgroundConfig:
    """Configuration for PDF background processing."""
    width: int = 800  # Default width
    height: int = 1280  # Default height
    
    @property
    def resolution(self) -> Tuple[int, int]:
        """Get background resolution as tuple."""
        return (self.width, self.height)
    
    def validate(self) -> bool:
        """Validate resolution settings."""
        return self.width > 0 and self.height > 0

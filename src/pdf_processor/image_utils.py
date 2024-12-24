"""Image processing utilities."""
from PIL import Image

def resize_image_to_fit(image: Image.Image, target_width: int, target_height: int) -> Image.Image:
    """Resize image to fit within target dimensions while maintaining aspect ratio."""
    aspect_ratio = image.width / image.height
    target_ratio = target_width / target_height

    if aspect_ratio > target_ratio:
        # Image is wider than target ratio, fit to width
        new_width = target_width
        new_height = int(target_width / aspect_ratio)
    else:
        # Image is taller than target ratio, fit to height
        new_height = target_height
        new_width = int(target_height * aspect_ratio)

    return image.resize((new_width, new_height), Image.Resampling.LANCZOS)

"""Image loading and preprocessing utilities for Anton's pipeline."""

class ImageLoader:
    """Handles image loading and preprocessing."""
    def __init__(self):
        self.current_image = None

    def load(self, image_path):
        """Load image from path (mock implementation)."""
        self.current_image = image_path  # In real code, load the image data
        return self.current_image 
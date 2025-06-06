"""Image loading and preprocessing utilities for Anton's pipeline."""

from pathlib import Path
from typing import Union, Tuple, Optional, List
import numpy as np
from PIL import Image
import logging

logger = logging.getLogger(__name__)

class ImageLoader:
    """Handles image loading and preprocessing for microscopy analysis."""
    
    def __init__(self):
        """Initialize ImageLoader."""
        self.current_image = None
        self.current_image_path = None
        self.metadata = {}
    
    def load(self, image_path: Union[str, Path]) -> np.ndarray:
        """Load image from path.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            numpy array of the loaded image
        """
        try:
            image_path = Path(image_path)
            if not image_path.exists():
                raise FileNotFoundError(f"Image not found: {image_path}")
            
            # Load image using PIL (supports many formats including TIFF)
            pil_image = Image.open(image_path)
            
            # Convert to numpy array
            image_array = np.array(pil_image)
            
            # Store for later use
            self.current_image = image_array
            self.current_image_path = image_path
            
            # Extract basic metadata
            self.metadata = {
                'shape': image_array.shape,
                'dtype': str(image_array.dtype),
                'path': str(image_path),
                'format': pil_image.format,
                'mode': pil_image.mode
            }
            
            logger.info(f"Loaded image: {image_path}, shape: {image_array.shape}")
            return image_array
            
        except Exception as e:
            logger.error(f"Failed to load image {image_path}: {e}")
            raise
    
    def preprocess(self, image: np.ndarray, normalize: bool = True, 
                   channels: Optional[List[int]] = None) -> np.ndarray:
        """Preprocess image for analysis.
        
        Args:
            image: Input image array
            normalize: Whether to normalize intensity values
            channels: Specific channels to extract (for multi-channel images)
            
        Returns:
            Preprocessed image array
        """
        try:
            processed = image.copy()
            
            # Extract specific channels if requested
            if channels is not None and len(image.shape) > 2:
                if len(image.shape) == 3:
                    # RGB/multi-channel image
                    processed = processed[:, :, channels]
                elif len(image.shape) == 4:
                    # Multi-channel with additional dimension
                    processed = processed[:, :, :, channels]
            
            # Normalize if requested
            if normalize:
                processed = self._normalize_image(processed)
            
            return processed
            
        except Exception as e:
            logger.error(f"Failed to preprocess image: {e}")
            raise
    
    def _normalize_image(self, image: np.ndarray) -> np.ndarray:
        """Normalize image intensity values to 0-1 range."""
        if image.dtype == np.uint8:
            return image.astype(np.float32) / 255.0
        elif image.dtype == np.uint16:
            return image.astype(np.float32) / 65535.0
        else:
            # For float images, normalize to 0-1 range
            min_val = image.min()
            max_val = image.max()
            if max_val > min_val:
                return (image - min_val) / (max_val - min_val)
            else:
                return image
    
    def extract_channel(self, image: np.ndarray, channel: int) -> np.ndarray:
        """Extract a specific channel from multi-channel image.
        
        Args:
            image: Multi-channel image array
            channel: Channel index to extract
            
        Returns:
            Single-channel image array
        """
        try:
            if len(image.shape) == 2:
                # Grayscale image
                return image
            elif len(image.shape) == 3:
                # Multi-channel image
                if channel < image.shape[2]:
                    return image[:, :, channel]
                else:
                    raise ValueError(f"Channel {channel} not available in image with {image.shape[2]} channels")
            else:
                raise ValueError(f"Unsupported image shape: {image.shape}")
                
        except Exception as e:
            logger.error(f"Failed to extract channel {channel}: {e}")
            raise
    
    def convert_to_8bit(self, image: np.ndarray) -> np.ndarray:
        """Convert image to 8-bit for display/export.
        
        Args:
            image: Input image array
            
        Returns:
            8-bit image array
        """
        try:
            if image.dtype == np.uint8:
                return image
            
            # Normalize to 0-1 range first
            normalized = self._normalize_image(image)
            
            # Convert to 8-bit
            return (normalized * 255).astype(np.uint8)
            
        except Exception as e:
            logger.error(f"Failed to convert to 8-bit: {e}")
            raise
    
    def save_image(self, image: np.ndarray, output_path: Union[str, Path], 
                   format: str = 'PNG') -> None:
        """Save image to file.
        
        Args:
            image: Image array to save
            output_path: Output file path
            format: Image format (PNG, TIFF, etc.)
        """
        try:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Convert to 8-bit if needed
            if image.dtype != np.uint8:
                image = self.convert_to_8bit(image)
            
            # Create PIL image and save
            pil_image = Image.fromarray(image)
            pil_image.save(output_path, format=format)
            
            logger.info(f"Saved image to: {output_path}")
            
        except Exception as e:
            logger.error(f"Failed to save image to {output_path}: {e}")
            raise
    
    def get_image_stats(self, image: Optional[np.ndarray] = None) -> dict:
        """Get basic statistics about the image.
        
        Args:
            image: Image array (uses current_image if None)
            
        Returns:
            Dictionary with image statistics
        """
        if image is None:
            image = self.current_image
        
        if image is None:
            return {}
        
        try:
            stats = {
                'shape': image.shape,
                'dtype': str(image.dtype),
                'min': float(image.min()),
                'max': float(image.max()),
                'mean': float(image.mean()),
                'std': float(image.std())
            }
            
            if len(image.shape) > 2:
                stats['channels'] = image.shape[2] if len(image.shape) == 3 else image.shape[-1]
            
            return stats
            
        except Exception as e:
            logger.error(f"Failed to compute image statistics: {e}")
            return {}
    
    def create_rgb_composite(self, channels: List[np.ndarray], 
                           colors: List[Tuple[float, float, float]] = None) -> np.ndarray:
        """Create RGB composite from multiple channels.
        
        Args:
            channels: List of single-channel images
            colors: List of RGB colors for each channel (default: R, G, B)
            
        Returns:
            RGB composite image
        """
        try:
            if not channels:
                raise ValueError("No channels provided")
            
            # Default colors (R, G, B)
            if colors is None:
                colors = [(1, 0, 0), (0, 1, 0), (0, 0, 1)]
            
            # Ensure all channels have the same shape
            shape = channels[0].shape
            for i, ch in enumerate(channels):
                if ch.shape != shape:
                    raise ValueError(f"Channel {i} shape {ch.shape} doesn't match expected {shape}")
            
            # Create RGB composite
            composite = np.zeros((*shape, 3), dtype=np.float32)
            
            for i, (channel, color) in enumerate(zip(channels, colors)):
                # Normalize channel
                norm_channel = self._normalize_image(channel)
                
                # Apply color
                for c in range(3):
                    composite[:, :, c] += norm_channel * color[c]
            
            # Clip to valid range
            composite = np.clip(composite, 0, 1)
            
            return composite
            
        except Exception as e:
            logger.error(f"Failed to create RGB composite: {e}")
            raise
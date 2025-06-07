# Generic Multi-Channel Image Handling Design

## Current Problem
- BBBC013 has separate files: Channel1-01-A-01.BMP, Channel2-01-A-01.BMP  
- Current VLM interface only handles single images
- Need generic solution for any multi-channel microscopy data

## Proposed Generic Solution

### 1. ImageLoader Enhancement
```python
def load_multi_channel(self, image_patterns: List[str], 
                      channel_names: Optional[List[str]] = None) -> np.ndarray:
    """Load multiple channel images and combine into single array.
    
    Args:
        image_patterns: List of file paths or glob patterns for each channel
        channel_names: Optional names for channels (e.g., ['DAPI', 'GFP', 'RFP'])
    
    Returns:
        Combined numpy array with shape (H, W, C) where C is number of channels
    """
```

### 2. VLM Interface Enhancement  
```python
def _prepare_multi_channel_image(self, channels: Union[np.ndarray, List[str]], 
                                composition_method: str = 'rgb_composite') -> str:
    """Prepare multi-channel image for VLM.
    
    Args:
        channels: Either stacked array (H,W,C) or list of channel file paths
        composition_method: 'rgb_composite', 'side_by_side', 'overlay'
    
    Returns:
        Base64 encoded image for VLM
    """
```

### 3. Generic Channel Configuration
```python
config = {
    'channels': {
        'paths': ['Channel1-{well}.BMP', 'Channel2-{well}.BMP'],
        'names': ['protein', 'nuclei'],  # Generic names, not BBBC013-specific
        'composition': 'rgb_composite',  # How to combine for VLM
        'weights': [1.0, 0.8]  # Optional channel weights
    }
}
```

### 4. Flexible Composition Methods
- **RGB Composite**: Map channels to R, G, B (most common)
- **Side-by-side**: Show channels separately in one image
- **Overlay**: Weighted combination with transparency
- **Individual**: Send each channel separately to VLM

## Benefits
- Works with any microscopy dataset structure
- User configurable channel combination
- No hardcoded experiment specifics
- Supports various file naming patterns
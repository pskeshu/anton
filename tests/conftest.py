"""Test configuration and fixtures for Anton tests."""

import pytest
import sys
from pathlib import Path

# Add the anton package to the path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

@pytest.fixture
def sample_image_path():
    """Path to sample test image."""
    return PROJECT_ROOT / "data" / "sample_images" / "demo_images" / "img00.png"

@pytest.fixture
def test_config():
    """Basic test configuration."""
    return {
        "channels": [0, 1, 2],
        "vlm_provider": "claude",
        "quantitative": {
            "min_object_area": 30,
            "max_object_area": 5000
        },
        "batch_size": 5
    }
"""Configuration management for Anton's analysis pipeline."""

from pathlib import Path
from typing import Dict, List, Optional, Union
import json
import logging

logger = logging.getLogger(__name__)

class Config:
    """Configuration management for Anton's analysis pipeline."""
    
    DEFAULT_CONFIG = {
        "channels": [0],
        "neighborhood_size": [100, 100],
        "vlm": {
            "model": "gpt-4-vision-preview",
            "temperature": 0.7,
            "max_tokens": 1000
        },
        "analysis": {
            "min_confidence": 0.7,
            "batch_size": 10
        },
        "output": {
            "save_intermediate": True,
            "format": "json"
        }
    }
    
    def __init__(self, config_path: Optional[Union[str, Path]] = None):
        """Initialize configuration from file or defaults."""
        self.config = self.DEFAULT_CONFIG.copy()
        
        if config_path:
            self.load_config(config_path)
    
    def load_config(self, config_path: Union[str, Path]) -> None:
        """Load configuration from JSON file."""
        try:
            with open(config_path, 'r') as f:
                user_config = json.load(f)
            
            # Update default config with user settings
            self._update_config(self.config, user_config)
            logger.info(f"Loaded configuration from {config_path}")
            
        except Exception as e:
            logger.error(f"Failed to load configuration: {str(e)}")
            raise
    
    def _update_config(self, base: Dict, update: Dict) -> None:
        """Recursively update configuration dictionary."""
        for key, value in update.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                self._update_config(base[key], value)
            else:
                base[key] = value
    
    def get(self, key: str, default: Optional[any] = None) -> any:
        """Get configuration value by key."""
        keys = key.split('.')
        value = self.config
        
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
            else:
                return default
            
            if value is None:
                return default
        
        return value
    
    def set(self, key: str, value: any) -> None:
        """Set configuration value by key."""
        keys = key.split('.')
        config = self.config
        
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        config[keys[-1]] = value
    
    def save(self, config_path: Union[str, Path]) -> None:
        """Save current configuration to file."""
        try:
            with open(config_path, 'w') as f:
                json.dump(self.config, f, indent=4)
            logger.info(f"Saved configuration to {config_path}")
            
        except Exception as e:
            logger.error(f"Failed to save configuration: {str(e)}")
            raise 
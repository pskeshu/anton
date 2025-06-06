#!/usr/bin/env python3
"""Test script for VLM interface with sample images."""

import asyncio
import os
import sys
from pathlib import Path

# Add the anton package to the path
sys.path.insert(0, str(Path(__file__).parent))

from anton.vlm.interface import VLMInterface

async def test_vlm():
    """Test VLM interface with sample images."""
    
    # Path to sample image
    image_path = "data/sample_images/demo_images/img00.png"
    
    if not os.path.exists(image_path):
        print(f"Image not found: {image_path}")
        return
    
    print("Testing VLM interface...")
    print(f"Image: {image_path}")
    
    # Test with different providers
    providers = ["claude", "gemini"]
    
    for provider in providers:
        print(f"\n=== Testing {provider.upper()} ===")
        
        try:
            # Initialize VLM interface
            if provider == "gemini":
                # You can set your Gemini API key here
                gemini_key = os.getenv("GOOGLE_API_KEY")
                if not gemini_key:
                    print(f"Skipping {provider} - no API key found")
                    continue
                vlm = VLMInterface(provider=provider, api_key=gemini_key)
            else:
                vlm = VLMInterface(provider=provider)
            
            print(f"Initialized {provider} VLM interface")
            print(f"Model: {vlm.model}")
            print(f"Available prompts: {list(vlm.prompts.keys())}")
            
            # Test Stage 1: Global scene analysis
            print("\nRunning Stage 1: Global scene analysis...")
            result = await vlm.analyze_global_scene(image_path, channels=[0, 1])
            print(f"Result: {result}")
            
        except Exception as e:
            print(f"Error testing {provider}: {str(e)}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_vlm())
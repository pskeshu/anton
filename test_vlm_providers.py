#!/usr/bin/env python3
"""
Test Anton with multiple VLM providers
Usage: 
  # For Claude
  export ANTHROPIC_API_KEY='your-claude-key'
  python test_vlm_providers.py claude
  
  # For Gemini  
  export GOOGLE_API_KEY='your-gemini-key'
  python test_vlm_providers.py gemini
  
  # For OpenAI
  export OPENAI_API_KEY='your-openai-key'  
  python test_vlm_providers.py openai
"""

import asyncio
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from anton.core.pipeline import AnalysisPipeline

async def test_vlm_provider(provider):
    """Test Anton with specific VLM provider."""
    
    # Check for appropriate API key
    api_key_env = {
        "claude": "ANTHROPIC_API_KEY",
        "gemini": "GOOGLE_API_KEY", 
        "openai": "OPENAI_API_KEY"
    }
    
    if provider not in api_key_env:
        print(f"❌ Unknown provider: {provider}")
        print("Supported: claude, gemini, openai")
        return
    
    api_key = os.getenv(api_key_env[provider])
    if not api_key:
        print(f"❌ No {api_key_env[provider]} found!")
        print(f"Please set your {provider} API key:")
        print(f"export {api_key_env[provider]}='your-key-here'")
        return
    
    print(f"🔬 Testing Anton with {provider.upper()} VLM")
    print("=" * 50)
    
    # Check for sample image
    sample_image = Path("data/sample_images/demo_images/img00.png")
    if not sample_image.exists():
        print("❌ Sample image not found!")
        return
    
    # Configure with the provider
    config = {
        "vlm_provider": provider,
        "vlm_api_key": api_key,
        "channels": [0, 1, 2]
    }
    
    try:
        # Initialize pipeline
        pipeline = AnalysisPipeline(config)
        print(f"✅ Initialized pipeline with {provider}")
        print(f"📱 Model: {pipeline.vlm.model}")
        
        # Test Stage 1 - this should call real VLM API
        print(f"\n🧠 Calling real {provider} API for image analysis...")
        stage1_result = await pipeline.run_stage_1(sample_image)
        
        print("🎉 SUCCESS! Real VLM Response:")
        print("=" * 50)
        print(f"Quality Score: {stage1_result.get('quality_score', 'N/A')}")
        print(f"Analysis: {stage1_result.get('analysis', 'N/A')[:200]}...")
        print(f"Recommendation: {stage1_result.get('recommended_analysis', 'N/A')}")
        
        # Test Stage 2 
        print(f"\n🎯 Testing object detection with {provider}...")
        stage2_result = await pipeline.run_stage_2()
        detected_objects = stage2_result.get('detected_objects', [])
        print(f"✅ Detected {len(detected_objects)} object types")
        
        for obj in detected_objects[:3]:  # Show first 3
            print(f"   • {obj.get('type', 'Unknown')}: {obj.get('confidence', 0):.1%}")
        
        print(f"\n✨ {provider.upper()} integration working perfectly!")
        
    except Exception as e:
        print(f"❌ {provider} VLM test failed: {e}")
        import traceback
        traceback.print_exc()

def main():
    if len(sys.argv) != 2:
        print("Usage: python test_vlm_providers.py <provider>")
        print("Providers: claude, gemini, openai")
        print("")
        print("Examples:")
        print("  export ANTHROPIC_API_KEY='sk-...' && python test_vlm_providers.py claude")
        print("  export GOOGLE_API_KEY='AI...' && python test_vlm_providers.py gemini")
        print("  export OPENAI_API_KEY='sk-...' && python test_vlm_providers.py openai")
        return
    
    provider = sys.argv[1].lower()
    asyncio.run(test_vlm_provider(provider))

if __name__ == "__main__":
    main()
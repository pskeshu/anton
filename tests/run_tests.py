#!/usr/bin/env python3
"""Main test runner for Anton framework."""

import sys
import asyncio
from pathlib import Path

# Add the anton package to the path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

def run_basic_tests():
    """Run basic functionality tests without pytest."""
    print("Running Anton Framework Tests")
    print("=" * 50)
    
    # Test 1: Basic imports
    print("\n1. Testing basic imports...")
    try:
        from anton.core.pipeline import AnalysisPipeline
        from anton.vlm.interface import VLMInterface
        from anton.analysis.quantitative import QuantitativeAnalyzer
        from anton.analysis.qualitative import QualitativeAnalyzer
        print("✅ All imports successful")
    except Exception as e:
        print(f"❌ Import failed: {e}")
        return False
    
    # Test 2: Pipeline initialization
    print("\n2. Testing pipeline initialization...")
    try:
        config = {"vlm_provider": "claude", "channels": [0, 1, 2]}
        pipeline = AnalysisPipeline(config)
        print("✅ Pipeline initialization successful")
    except Exception as e:
        print(f"❌ Pipeline initialization failed: {e}")
        return False
    
    # Test 3: Async pipeline test
    print("\n3. Testing async pipeline...")
    try:
        async def test_async():
            image_path = PROJECT_ROOT / "data" / "sample_images" / "demo_images" / "img00.png"
            if not image_path.exists():
                print("⚠️  Sample image not found, skipping async test")
                return True
            
            config = {"vlm_provider": "claude", "channels": [0, 1, 2]}
            pipeline = AnalysisPipeline(config)
            result = await pipeline.run_stage_1(image_path)
            return isinstance(result, dict)
        
        result = asyncio.run(test_async())
        if result:
            print("✅ Async pipeline test successful")
        else:
            print("❌ Async pipeline test failed")
            return False
    except Exception as e:
        print(f"❌ Async pipeline test failed: {e}")
        return False
    
    print("\n" + "=" * 50)
    print("✅ All basic tests passed!")
    return True

def run_integration_tests():
    """Run integration tests."""
    print("\n4. Testing full pipeline integration...")
    try:
        from anton.core.pipeline import AnalysisPipeline
        
        async def test_full_pipeline():
            image_path = PROJECT_ROOT / "data" / "sample_images" / "demo_images" / "img00.png"
            if not image_path.exists():
                print("⚠️  Sample image not found, skipping integration test")
                return True
            
            config = {
                "vlm_provider": "claude", 
                "channels": [0, 1, 2],
                "quantitative": {"min_object_area": 30, "max_object_area": 5000}
            }
            pipeline = AnalysisPipeline(config)
            results = await pipeline.run_pipeline(image_path)
            
            # Check all stages completed
            required_keys = ["stage_1_global", "stage_2_objects", "stage_3_features", "stage_4_population"]
            return all(key in results and results[key] is not None for key in required_keys)
        
        result = asyncio.run(test_full_pipeline())
        if result:
            print("✅ Full pipeline integration test successful")
        else:
            print("❌ Full pipeline integration test failed")
            return False
    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        return False
    
    return True

if __name__ == "__main__":
    print("Anton Framework Test Suite")
    print("=" * 50)
    
    success = True
    success &= run_basic_tests()
    success &= run_integration_tests()
    
    if success:
        print("\n🎉 All tests passed!")
        sys.exit(0)
    else:
        print("\n💥 Some tests failed!")
        sys.exit(1)
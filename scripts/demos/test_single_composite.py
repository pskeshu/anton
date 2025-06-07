#!/usr/bin/env python3
"""
Test Single Composite Analysis

Quick test of one high-quality composite image.
"""

import os
import sys
import asyncio
from pathlib import Path

# Add Anton to path
sys.path.append(str(Path(__file__).parent))

try:
    from anton.core.pipeline import AnalysisPipeline
    anton_available = True
except ImportError as e:
    anton_available = False
    print(f"❌ Anton not available: {e}")

async def test_single_composite():
    """Test analysis of one composite."""
    
    print("🧬 Testing Single Composite Analysis")
    print("=" * 40)
    
    # Use first composite
    composite_path = "data/bbbc021/composites/Week1_150607_B02_s2_composite.png"
    
    if not Path(composite_path).exists():
        print(f"❌ Composite not found: {composite_path}")
        return
    
    if not anton_available:
        print("❌ Anton not available")
        return
    
    print(f"📸 Analyzing: {composite_path}")
    print(f"🔬 This is DMSO (control) treatment")
    
    try:
        # Minimal config for speed
        config = {
            'vlm_provider': 'gemini' if os.getenv('GOOGLE_API_KEY') else 'claude',
            'biological_context': {
                'experiment': 'MCF7_3channel_fluorescence',
                'channels': 'blue_DAPI_green_tubulin_red_actin',
                'compound': 'DMSO_control'
            }
        }
        
        print("🔄 Running VLM analysis...")
        
        pipeline = AnalysisPipeline(config)
        results = await pipeline.run_pipeline(composite_path)
        
        print("✅ Analysis complete!")
        
        # Quick summary
        total_text = 0
        for stage, data in results.items():
            if isinstance(data, str):
                length = len(data)
                preview = data[:100] + "..." if len(data) > 100 else data
                print(f"\n📋 {stage}: {length} chars")
                print(f"   Preview: {preview}")
                total_text += length
            elif isinstance(data, dict):
                print(f"\n📋 {stage}: Structured data")
                if 'description' in data:
                    desc_len = len(str(data['description']))
                    print(f"   Description: {desc_len} chars")
                    total_text += desc_len
        
        print(f"\n📊 Total analysis: {total_text} characters")
        
        # Look for biological terms
        all_text = str(results).lower()
        bio_terms = ['cell', 'nuclear', 'nuclei', 'tubulin', 'actin', 'cytoskeleton', 'morphology']
        found_terms = [term for term in bio_terms if term in all_text]
        
        print(f"🧬 Biological terms detected: {', '.join(found_terms)}")
        
        if len(found_terms) >= 4:
            print("🎉 Excellent - VLM recognizing multi-channel biology!")
        elif len(found_terms) >= 2:
            print("✅ Good - VLM detecting cellular features")
        else:
            print("⚠️ Limited - VLM may not be processing optimally")
            
    except Exception as e:
        print(f"❌ Analysis failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_single_composite())
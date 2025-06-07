#!/usr/bin/env python3
"""
BBBC013 Protein Translocation Analysis Demo

Tests Anton framework on real FKHR-GFP protein translocation dataset
from Broad Institute. Analyzes nuclear vs cytoplasmic localization
under drug treatment conditions.

Dataset: Human U2OS cells, FKHR-GFP protein, Wortmannin/LY294002 drugs
"""

import os
import sys
import random
from pathlib import Path

# Add anton to path
sys.path.append(str(Path(__file__).parent))

from anton.core.pipeline import AnalysisPipeline
from anton.utils.image_io import ImageLoader

def main():
    print("🧬 BBBC013 Protein Translocation Analysis with Anton")
    print("=" * 60)
    
    # Check if dataset exists
    dataset_path = Path("data/bbbc013/BBBC013_v1_images_bmp")
    if not dataset_path.exists():
        print("❌ BBBC013 dataset not found. Please run download first.")
        return
    
    # Find available images
    image_files = list(dataset_path.glob("*.BMP"))
    if not image_files:
        print("❌ No BMP images found in dataset.")
        return
    
    print(f"📊 Found {len(image_files)} images in BBBC013 dataset")
    
    # Separate channels
    channel1_files = [f for f in image_files if "Channel1" in f.name]  # FKHR-GFP
    channel2_files = [f for f in image_files if "Channel2" in f.name]  # DNA (DRAQ)
    
    print(f"🟢 Channel 1 (FKHR-GFP): {len(channel1_files)} images")
    print(f"🔵 Channel 2 (DNA): {len(channel2_files)} images")
    
    # Select a representative image pair for analysis
    if channel1_files and channel2_files:
        # Pick a random well for demonstration
        sample_gfp = random.choice(channel1_files)
        well_id = sample_gfp.name.split('-')[1:3]  # Extract well position
        sample_dna = None
        
        # Find corresponding DNA channel
        for dna_file in channel2_files:
            if all(part in dna_file.name for part in well_id):
                sample_dna = dna_file
                break
        
        if sample_dna:
            print(f"\n🎯 Analyzing sample well: {'-'.join(well_id)}")
            print(f"   FKHR-GFP: {sample_gfp.name}")
            print(f"   DNA:      {sample_dna.name}")
            
            # Run Anton analysis
            analyze_protein_translocation(sample_gfp, sample_dna)
        else:
            print("❌ Could not find matching DNA channel image")
    else:
        print("❌ Missing channel images")

def analyze_protein_translocation(gfp_path, dna_path):
    """Analyze protein translocation using Anton pipeline"""
    
    print("\n🔬 Starting Anton Analysis...")
    print("-" * 40)
    
    try:
        # Configure pipeline with BBBC013 biological context
        import os
        
        # Create enhanced config with biological context
        bbbc013_context = {
            "experiment_type": "protein_translocation",
            "cell_line": "U2OS_osteosarcoma", 
            "protein": "FKHR-GFP",
            "drugs": ["Wortmannin", "LY294002"],
            "readout": "nuclear_vs_cytoplasmic_localization",
            "channels": ["FKHR-GFP", "DNA_DRAQ"]
        }
        
        if os.getenv("GOOGLE_API_KEY"):
            config = {
                "vlm_provider": "gemini", 
                "channels": [0],
                "biological_context": bbbc013_context
            }
            print("🟢 Using Gemini VLM with BBBC013 protein translocation context")
        elif os.getenv("ANTHROPIC_API_KEY"):
            config = {
                "vlm_provider": "claude", 
                "channels": [0],
                "biological_context": bbbc013_context
            }
            print("🔵 Using Claude VLM with BBBC013 protein translocation context")
        else:
            config = {
                "vlm_provider": "mock", 
                "channels": [0],
                "biological_context": bbbc013_context
            }
            print("⚠️  Using mock VLM (no API key)")
        
        # Initialize pipeline
        pipeline = AnalysisPipeline(config)
        
        # Load the GFP channel (primary analysis target)
        print(f"📖 Loading FKHR-GFP image: {gfp_path.name}")
        loader = ImageLoader()
        image = loader.load(str(gfp_path))
        
        if image is None:
            print("❌ Failed to load image")
            return
        
        print(f"✅ Image loaded: {image.shape} pixels")
        
        print("\n🧬 BBBC013 Context: Analyzing FKHR-GFP protein translocation under drug treatment")
        print("🎯 Expected: Nuclear vs cytoplasmic localization patterns")
        print("\n🚀 Running Anton pipeline with protein context...")
        
        # Enable more verbose logging
        import logging
        logging.basicConfig(level=logging.INFO)
        
        # Show what prompts are loaded
        print(f"📝 LOADED PROMPTS:")
        for prompt_name, prompt_text in pipeline.vlm.prompts.items():
            print(f"   {prompt_name}: {prompt_text[:100]}...")
        
        # Show biological context being used
        print(f"\n🧬 BIOLOGICAL CONTEXT:")
        if pipeline.vlm.biological_context:
            for key, value in pipeline.vlm.biological_context.items():
                print(f"   {key}: {value}")
        else:
            print("   ❌ No biological context found!")
        print()
        
        try:
            results = pipeline.run_pipeline_sync(gfp_path)
        except Exception as e:
            print(f"❌ Pipeline failed: {e}")
            import traceback
            traceback.print_exc()
            return
        
        # Debug: Print detailed results content
        print(f"🔍 DEBUG: Detailed results analysis")
        if results:
            for stage_key, stage_data in results.items():
                print(f"   {stage_key}:")
                if stage_data:
                    if isinstance(stage_data, dict):
                        print(f"      Keys: {list(stage_data.keys())}")
                        if 'description' in stage_data:
                            desc_len = len(stage_data['description'])
                            print(f"      Description: {desc_len} chars")
                        if 'detected_objects' in stage_data:
                            obj_count = len(stage_data['detected_objects'])
                            print(f"      Detected objects: {obj_count}")
                    else:
                        print(f"      Type: {type(stage_data)}")
                else:
                    print(f"      Status: EMPTY/None")
        print()
        
        # Display results with focus on protein localization
        display_protein_results(results, gfp_path.name)
        
    except Exception as e:
        print(f"❌ Analysis failed: {str(e)}")
        import traceback
        traceback.print_exc()

def display_protein_results(results, filename):
    """Display results focused on protein translocation insights"""
    
    print(f"\n🎯 PROTEIN TRANSLOCATION ANALYSIS RESULTS")
    print(f"📁 File: {filename}")
    print("=" * 60)
    
    # Handle the actual stage-based results structure
    stages = ['stage_1_global', 'stage_2_objects', 'stage_3_features', 'stage_4_population']
    
    print(f"🧠 VLM BIOLOGICAL INSIGHTS:")
    
    for stage_key in stages:
        if stage_key in results and results[stage_key]:
            stage_data = results[stage_key]
            stage_num = stage_key.split('_')[1]
            stage_name = stage_key.split('_')[2]
            
            print(f"\n   📋 STAGE {stage_num} ({stage_name.upper()}):")
            
            # Extract description (Stage 1)
            if 'description' in stage_data:
                desc = stage_data['description']
                # Focus on protein localization keywords
                if any(keyword in desc.lower() for keyword in 
                      ['nuclear', 'cytoplasmic', 'translocation', 'localization', 'fkhr', 'gfp', 'protein']):
                    print(f"      🎯 {desc}")
                else:
                    # Show abbreviated version for less relevant content
                    short_desc = desc[:150] + "..." if len(desc) > 150 else desc
                    print(f"      📝 {short_desc}")
            
            # Show Stage 2 detected objects
            if 'detected_objects' in stage_data:
                objects = stage_data['detected_objects']
                print(f"      📊 Detected objects: {len(objects)}")
                if 'segmentation_guidance' in stage_data:
                    guidance = stage_data['segmentation_guidance']
                    if len(guidance) > 200:
                        guidance = guidance[:200] + "..."
                    print(f"      🔬 Segmentation guidance: {guidance}")
            
            # Show Stage 3 feature analyses
            if 'object_analyses' in stage_data:
                analyses = stage_data['object_analyses']
                print(f"      🔍 Feature analyses: {len(analyses)}")
                if 'feature_descriptions' in stage_data:
                    descriptions = stage_data['feature_descriptions']
                    if descriptions and len(descriptions[0]) > 200:
                        desc_short = descriptions[0][:200] + "..."
                    else:
                        desc_short = descriptions[0] if descriptions else "No description"
                    print(f"      📝 Features: {desc_short}")
            
            # Show Stage 4 population insights
            if 'population_summary' in stage_data:
                summary = stage_data['population_summary']
                if len(summary) > 200:
                    summary = summary[:200] + "..."
                print(f"      👥 Population: {summary}")
            
            # Show CMPO terms if available
            if 'cmpo_terms' in stage_data and stage_data['cmpo_terms']:
                cmpo_terms = stage_data['cmpo_terms']
                print(f"      🧬 CMPO Terms ({len(cmpo_terms)}):")
                
                # Separate localization vs other terms
                localization_terms = []
                other_terms = []
                
                for term_data in cmpo_terms:
                    if isinstance(term_data, dict):
                        term_name = term_data.get('term', 'Unknown')
                        confidence = term_data.get('confidence', 0)
                    else:
                        term_name = str(term_data)
                        confidence = 1.0
                    
                    if any(keyword in term_name.lower() for keyword in 
                          ['nuclear', 'cytoplasm', 'localization', 'translocation']):
                        localization_terms.append((term_name, confidence))
                    else:
                        other_terms.append((term_name, confidence))
                
                # Show top localization terms
                if localization_terms:
                    print(f"         🎯 Localization: {localization_terms[0][0]} ({localization_terms[0][1]:.2f})")
                
                # Show top other terms
                if other_terms:
                    top_other = sorted(other_terms, key=lambda x: x[1], reverse=True)[:2]
                    for term, conf in top_other:
                        print(f"         • {term} ({conf:.2f})")
    
    # Test CMPO mapping on all stages with actual content
    print(f"\n🧬 TESTING CMPO MAPPING ON ALL STAGES:")
    
    try:
        from anton.cmpo.mapping import map_to_cmpo
        from anton.cmpo.ontology import CMPOOntology
        
        cmpo_mapper = CMPOOntology()
        
        for stage_key in ['stage_1_global', 'stage_2_objects', 'stage_3_features', 'stage_4_population']:
            if stage_key in results and results[stage_key]:
                stage_data = results[stage_key]
                stage_num = stage_key.split('_')[1]
                
                # Extract text content for CMPO mapping
                stage_text = ""
                if 'description' in stage_data:
                    stage_text = stage_data['description']
                elif 'segmentation_guidance' in stage_data:
                    stage_text = stage_data['segmentation_guidance']
                elif 'feature_descriptions' in stage_data and stage_data['feature_descriptions']:
                    stage_text = stage_data['feature_descriptions'][0] if stage_data['feature_descriptions'] else ""
                elif 'population_summary' in stage_data:
                    stage_text = stage_data['population_summary']
                
                if stage_text and len(stage_text) > 50:  # Only test substantial content
                    # Clean JSON formatting if present
                    if stage_text.startswith('```json') or stage_text.startswith('```python'):
                        lines = stage_text.split('\n')
                        stage_text = '\n'.join([line for line in lines if not line.strip().startswith('```')])
                    
                    cmpo_results = map_to_cmpo(stage_text, cmpo_mapper, context='protein_localization')
                    
                    print(f"   📋 STAGE {stage_num}:")
                    if cmpo_results:
                        print(f"      ✅ Found {len(cmpo_results)} CMPO mappings:")
                        for i, mapping in enumerate(cmpo_results[:3]):  # Show top 3 per stage
                            term = mapping.get('term_name', 'Unknown')
                            confidence = mapping.get('confidence', 0)
                            cmpo_id = mapping.get('CMPO_ID', 'Unknown')
                            print(f"         {i+1}. {term} ({confidence:.2f})")
                    else:
                        print(f"      ❌ No CMPO mappings found")
                else:
                    print(f"   📋 STAGE {stage_num}: ⚠️ No substantial text content")
                    
    except Exception as e:
        print(f"   ❌ CMPO mapping failed: {e}")
    
    print(f"\n✅ Analysis complete for {filename}")
    print("=" * 60)

if __name__ == "__main__":
    # Check for API key
    if not os.getenv('GOOGLE_API_KEY') and not os.getenv('ANTHROPIC_API_KEY'):
        print("⚠️  No API key found. Set GOOGLE_API_KEY or ANTHROPIC_API_KEY")
        print("   Example: export GOOGLE_API_KEY='your-key-here'")
        sys.exit(1)
    
    main()
#!/usr/bin/env python3
"""
Quick 30-second demo of Anton Framework
Shows the core functionality without detailed explanations.
"""

import asyncio
import sys
from pathlib import Path

# ANSI color codes for terminal output
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    
    @staticmethod
    def colorize(text, color):
        return f"{color}{text}{Colors.ENDC}"

# Add anton to path
sys.path.insert(0, str(Path(__file__).parent))

from anton.core.pipeline import AnalysisPipeline

async def quick_demo():
    """Quick demonstration of Anton's core functionality."""
    
    print(Colors.colorize("🔬 Anton Framework - Quick Demo", Colors.HEADER + Colors.BOLD))
    print(Colors.colorize("=" * 50, Colors.BLUE))
    
    # Simple config - try Gemini first if key available, then Claude
    import os
    if os.getenv("GOOGLE_API_KEY"):
        config = {"vlm_provider": "gemini", "channels": [0, 1, 2]}
        print(Colors.colorize("🟢 Using Gemini VLM (API key detected)", Colors.GREEN))
    else:
        config = {"vlm_provider": "claude", "channels": [0, 1, 2]}
        print(Colors.colorize("🔵 Using Claude VLM", Colors.BLUE))
    
    # Initialize pipeline
    print(Colors.colorize("📸 Loading fluorescence microscopy image...", Colors.CYAN))
    pipeline = AnalysisPipeline(config)
    
    # Check for sample image
    sample_image = Path("data/sample_images/demo_images/img00.png")
    if not sample_image.exists():
        print(Colors.colorize("❌ Sample image not found!", Colors.RED))
        return
    
    print(Colors.colorize("🧠 VLM analyzing global scene...", Colors.CYAN))
    
    try:
        # Run analysis stage by stage to show outputs
        print(Colors.colorize("\n--- Stage 1: Global Scene Understanding ---", Colors.HEADER + Colors.BOLD))
        stage1 = await pipeline.run_stage_1(sample_image)
        quality_score = stage1.get('quality_score', 'N/A')
        print(f"🎯 {Colors.colorize('Quality Score:', Colors.YELLOW)} {Colors.colorize(str(quality_score), Colors.GREEN)}")
        analysis = stage1.get('analysis') or stage1.get('description', 'N/A')
        print(f"📝 {Colors.colorize('VLM Analysis:', Colors.YELLOW)} {Colors.colorize(str(analysis), Colors.CYAN)}")
        recommendation = stage1.get('recommended_analysis', 'N/A')
        print(f"💡 {Colors.colorize('Recommendation:', Colors.YELLOW)} {Colors.colorize(str(recommendation), Colors.GREEN)}")
        
        print(Colors.colorize("\n--- Stage 2: Object Detection ---", Colors.HEADER + Colors.BOLD))
        stage2 = await pipeline.run_stage_2()
        detected_objects = stage2.get('detected_objects', [])
        print(f"🔍 {Colors.colorize('Found', Colors.YELLOW)} {Colors.colorize(str(len(detected_objects)), Colors.GREEN)} {Colors.colorize('object types:', Colors.YELLOW)}")
        for obj in detected_objects:
            obj_type = obj.get('type', 'Unknown')
            confidence = obj.get('confidence', 0)
            print(f"   • {Colors.colorize(obj_type, Colors.CYAN)}: {Colors.colorize(f'{confidence:.1%}', Colors.GREEN)} confidence")
        guidance = stage2.get('segmentation_guidance', 'N/A')
        print(f"🛠️  {Colors.colorize('Segmentation guidance:', Colors.YELLOW)} {Colors.colorize(str(guidance), Colors.CYAN)}")
        
        print(Colors.colorize("\n--- Stage 3: Feature Analysis ---", Colors.HEADER + Colors.BOLD))
        stage3 = await pipeline.run_stage_3()
        analyses = stage3.get('object_analyses', [])
        print(f"🧬 {Colors.colorize('Analyzed', Colors.YELLOW)} {Colors.colorize(str(len(analyses)), Colors.GREEN)} {Colors.colorize('objects:', Colors.YELLOW)}")
        for i, analysis in enumerate(analyses[:2]):  # Show first 2
            features = analysis.get('features', [])
            feature_text = ', '.join(features[:3]) + ('...' if len(features) > 3 else '')
            print(f"   {Colors.colorize(f'Object {i+1}:', Colors.CYAN)} {Colors.colorize(feature_text, Colors.GREEN)}")
        
        print(Colors.colorize("\n--- Stage 4: Population Insights ---", Colors.HEADER + Colors.BOLD))
        stage4 = await pipeline.run_stage_4()
        pop_summary = stage4.get('population_summary', 'Generated population analysis')
        
        # Clean up population summary if it contains code
        if 'def ' in str(pop_summary) or 'python' in str(pop_summary).lower():
            pop_summary = "Population analysis completed (VLM returned code format - parsing needed)"
        
        print(f"📊 {Colors.colorize(str(pop_summary)[:200], Colors.CYAN)}{'...' if len(str(pop_summary)) > 200 else ''}")
        
        # Show CMPO mappings if available
        qual_results = stage4.get('qualitative_features', {})
        cmpo_summary = qual_results.get('cmpo_summary', {})
        if cmpo_summary and cmpo_summary.get('top_terms'):
            print(Colors.colorize("\n--- CMPO Phenotype Classification ---", Colors.HEADER + Colors.BOLD))
            top_terms = cmpo_summary.get('top_terms', [])[:3]  # Show top 3
            total_terms = cmpo_summary.get('total_unique_terms', 0)
            print(f"🧬 {Colors.colorize('Identified', Colors.YELLOW)} {Colors.colorize(str(total_terms), Colors.GREEN)} {Colors.colorize('unique phenotype terms:', Colors.YELLOW)}")
            for i, term in enumerate(top_terms, 1):
                term_name = term.get('term', 'Unknown')
                confidence = term.get('confidence', 0)
                stages = ', '.join(term.get('stages', []))
                print(f"   {Colors.colorize(f'{i}.', Colors.CYAN)} {Colors.colorize(term_name, Colors.GREEN)} (confidence: {Colors.colorize(f'{confidence:.2f}', Colors.YELLOW)})")
                print(f"      {Colors.colorize('Detected in:', Colors.CYAN)} {Colors.colorize(stages, Colors.BLUE)}")
        else:
            print(Colors.colorize("\n--- CMPO Phenotype Classification ---", Colors.HEADER + Colors.BOLD))
            print(f"🧬 {Colors.colorize('No CMPO terms mapped', Colors.RED)} - {Colors.colorize('VLM descriptions may need richer biological content', Colors.YELLOW)}")
        
        # Now show CMPO results organized by stage
        def show_cmpo_by_stage(stage_name, cmpo_mappings):
            """Helper to show CMPO results for a specific stage."""
            stage_mappings = [m for m in cmpo_mappings if m.get('stage') == stage_name]
            if stage_mappings:
                print(f"   🧬 {Colors.colorize('CMPO Terms:', Colors.YELLOW + Colors.BOLD)}")
                for mapping in stage_mappings[:2]:  # Show top 2 per stage
                    term_name = mapping.get('term_name', 'Unknown')
                    confidence = mapping.get('confidence', 0)
                    print(f"      • {Colors.colorize(term_name, Colors.GREEN)} ({Colors.colorize(f'{confidence:.2f}', Colors.YELLOW)})")
            else:
                print(f"   🧬 {Colors.colorize('CMPO Terms:', Colors.YELLOW + Colors.BOLD)} {Colors.colorize('None mapped', Colors.RED)}")

        # Extract CMPO mappings for stage-by-stage display
        all_mappings = cmpo_summary.get('mappings', []) if cmpo_summary else []
        
        print(Colors.colorize("\n" + "=" * 50, Colors.BLUE))
        print(Colors.colorize("STAGE-BY-STAGE CMPO ANALYSIS", Colors.HEADER + Colors.BOLD))
        print(Colors.colorize("=" * 50, Colors.BLUE))
        
        print(Colors.colorize("\n📊 Stage 1 (Global Context):", Colors.HEADER))
        show_cmpo_by_stage('global_context', all_mappings)
        
        print(Colors.colorize("\n📊 Stage 4 (Population Insights):", Colors.HEADER))
        show_cmpo_by_stage('population_insights', all_mappings)
        
        vlm_success = True
        
    except Exception as e:
        print(Colors.colorize(f"❌ VLM analysis failed: {e}", Colors.RED))
        vlm_success = False
        
        if "API" in str(e) or "key" in str(e).lower():
            print(Colors.colorize("\n🔑 Need API Key for VLM Analysis!", Colors.YELLOW + Colors.BOLD))
            print(Colors.colorize("Choose your VLM provider:", Colors.CYAN))
            print("")
            print(Colors.colorize("For Claude:", Colors.BLUE))
            print(f"  {Colors.colorize('export ANTHROPIC_API_KEY=', Colors.YELLOW)}{Colors.colorize("'your-key'", Colors.GREEN)}")
            print(f"  {Colors.colorize('Get key:', Colors.CYAN)} https://console.anthropic.com/")
            print("")
            print(Colors.colorize("For Gemini:", Colors.GREEN))
            print(f"  {Colors.colorize('export GOOGLE_API_KEY=', Colors.YELLOW)}{Colors.colorize("'your-key'", Colors.GREEN)}")
            print(f"  {Colors.colorize('Get key:', Colors.CYAN)} https://makersuite.google.com/app/apikey")
            print("")
            print(f"Then set {Colors.colorize('vlm_provider', Colors.YELLOW)} in config: {Colors.colorize("'claude'", Colors.BLUE)} or {Colors.colorize("'gemini'", Colors.GREEN)}")
            print(Colors.colorize("\nShowing quantitative analysis instead...", Colors.CYAN))
        
    # Also show quantitative results
    print(Colors.colorize("\n--- Quantitative Analysis ---", Colors.HEADER + Colors.BOLD))
    from anton.analysis.quantitative import QuantitativeAnalyzer, SegmentationStrategy
    quant = QuantitativeAnalyzer()
    quant_results = quant.extract_quantitative_features(
        sample_image, channels=[2], method=SegmentationStrategy.THRESHOLD
    )
    num_objects = quant_results['num_objects']
    print(f"🔬 {Colors.colorize('Traditional CV detected', Colors.YELLOW)} {Colors.colorize(str(num_objects), Colors.GREEN)} {Colors.colorize('objects', Colors.YELLOW)}")
    
    print(Colors.colorize("\n" + "=" * 50, Colors.BLUE))
    if vlm_success:
        print(Colors.colorize("✨ Anton = Computer Vision + Vision Language Models + CMPO", Colors.GREEN + Colors.BOLD))
        print(Colors.colorize("🚀 Hybrid analysis with standardized phenotype classification!", Colors.CYAN + Colors.BOLD))
        print(Colors.colorize("🧬 VLM-validated biological reasoning ensures scientific accuracy", Colors.YELLOW + Colors.BOLD))
    else:
        print(Colors.colorize("🔬 Anton's quantitative analysis is working!", Colors.GREEN + Colors.BOLD))
        print(Colors.colorize("🔑 Add VLM API key to see the full hybrid power + CMPO!", Colors.YELLOW + Colors.BOLD))

if __name__ == "__main__":
    asyncio.run(quick_demo())
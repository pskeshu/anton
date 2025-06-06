"""
CMPO Mapping Examples and Demonstrations

This file demonstrates the key concepts and usage patterns of the CMPO integration module.
Run with: 
  python -m anton.cmpo.examples  (from project root)
  OR
  python examples.py  (from anton/cmpo/ directory)
"""

import sys
import logging
from pathlib import Path

# Handle both direct execution and module execution
if __name__ == "__main__" and __package__ is None:
    # Add parent directories to path for direct execution
    current_dir = Path(__file__).parent
    project_root = current_dir.parent.parent
    sys.path.insert(0, str(project_root))
    from anton.cmpo.ontology import CMPOOntology
    from anton.cmpo.mapping import map_to_cmpo
else:
    # Normal relative imports for module execution
    from .ontology import CMPOOntology
    from .mapping import map_to_cmpo

logging.basicConfig(level=logging.INFO)

def demonstrate_basic_mapping():
    """Demonstrate basic CMPO mapping functionality."""
    print("=" * 60)
    print("BASIC CMPO MAPPING DEMONSTRATION")
    print("=" * 60)
    
    # Initialize CMPO ontology
    cmpo = CMPOOntology()
    print(f"Loaded CMPO ontology with {len(cmpo.ontology)} terms\n")
    
    # Example descriptions from VLM analysis
    test_descriptions = [
        "cells arrested in metaphase with condensed chromosomes",
        "fragmented nuclei with membrane blebbing indicating apoptosis",
        "abnormal spindle formation during cell division",
        "enlarged cell bodies with irregular morphology",
        "normal healthy fibroblast cells with typical morphology"
    ]
    
    for desc in test_descriptions:
        print(f"Description: '{desc}'")
        results = map_to_cmpo(desc, cmpo)
        
        if results:
            print(f"Found {len(results)} CMPO mappings:")
            for i, result in enumerate(results[:3], 1):
                print(f"  {i}. {result['CMPO_ID']}: {result['term_name']}")
                print(f"     Confidence: {result['confidence']:.3f}")
                print(f"     Evidence: {result['supporting_evidence']}")
                if result.get('hierarchy_path'):
                    print(f"     Hierarchy: {' → '.join(result['hierarchy_path'])}")
                print()
        else:
            print("  No CMPO mappings found")
        print("-" * 50)

def demonstrate_context_aware_mapping():
    """Demonstrate context-aware mapping with research focus."""
    print("\n" + "=" * 60)
    print("CONTEXT-AWARE MAPPING DEMONSTRATION") 
    print("=" * 60)
    
    cmpo = CMPOOntology()
    
    # Same description, different research contexts
    description = "abnormal cell division with chromosome segregation defects"
    
    contexts = [
        ("cell_cycle", "Cell cycle research focus"),
        ("morphology", "Morphology research focus"),
        (None, "No specific context")
    ]
    
    for context, context_desc in contexts:
        print(f"\n{context_desc}:")
        print(f"Description: '{description}'")
        results = map_to_cmpo(description, cmpo, context=context)
        
        if results:
            for i, result in enumerate(results[:2], 1):
                print(f"  {i}. {result['term_name']} (confidence: {result['confidence']:.3f})")
                print(f"     Context boost from: {result['supporting_evidence']}")
        else:
            print("  No mappings found")

def demonstrate_hierarchical_navigation():
    """Show how CMPO terms relate hierarchically."""
    print("\n" + "=" * 60)
    print("HIERARCHICAL NAVIGATION DEMONSTRATION")
    print("=" * 60)
    
    cmpo = CMPOOntology()
    
    # Find a term with rich hierarchy
    for term_id, term_data in cmpo.ontology.items():
        if term_data.get('parent_terms') and len(term_data['parent_terms']) > 0:
            print(f"Term: {term_data['name']} ({term_id})")
            print(f"Description: {term_data.get('description', 'No description')}")
            
            if term_data.get('synonyms'):
                print(f"Synonyms: {', '.join(term_data['synonyms'])}")
            
            print(f"Parent terms:")
            for parent_id in term_data['parent_terms']:
                parent_term = cmpo.get_term(parent_id)
                if parent_term:
                    print(f"  → {parent_term['name']} ({parent_id})")
            
            if term_data.get('equivalent_to'):
                print(f"Equivalent to: {term_data['equivalent_to']}")
            
            break
        
def demonstrate_semantic_analysis():
    """Show semantic component analysis."""
    print("\n" + "=" * 60)
    print("SEMANTIC ANALYSIS DEMONSTRATION")
    print("=" * 60)
    
    # Import internal functions for demonstration
    if __name__ == "__main__" and __package__ is None:
        from anton.cmpo.mapping import _extract_biological_tokens, _find_direct_matches
    else:
        from .mapping import _extract_biological_tokens, _find_direct_matches
    
    cmpo = CMPOOntology()
    
    description = "apoptotic cells with fragmented nuclei and chromatin condensation"
    print(f"Analyzing: '{description}'")
    
    # Show token extraction
    tokens = _extract_biological_tokens(description)
    print(f"Biological tokens: {sorted(tokens)}")
    
    # Show direct matches
    direct_matches = _find_direct_matches(description.lower(), cmpo)
    if direct_matches:
        print("\nDirect matches found:")
        for term_id, confidence, evidence in direct_matches[:3]:
            term = cmpo.get_term(term_id)
            if term:
                print(f"  {term['name']}: {confidence:.3f} (matched: {evidence})")

def demonstrate_integration_patterns():
    """Show how CMPO integrates with Anton pipeline."""
    print("\n" + "=" * 60)
    print("INTEGRATION PATTERNS DEMONSTRATION")
    print("=" * 60)
    
    # Simulate VLM output from different pipeline stages
    vlm_outputs = {
        "stage_1_global": "Dense population of adherent cells with fibroblast morphology",
        "stage_3_features": "Individual cells show elongated spindle shape with prominent stress fibers",
        "stage_4_population": "Population exhibits normal growth patterns with typical cell-cell contacts"
    }
    
    cmpo = CMPOOntology()
    
    print("Simulating Anton pipeline integration:")
    for stage, output in vlm_outputs.items():
        print(f"\n{stage.replace('_', ' ').title()}:")
        print(f"VLM Output: {output}")
        
        # Map to CMPO
        mappings = map_to_cmpo(output, cmpo)
        if mappings:
            best_match = mappings[0]
            print(f"Best CMPO Match: {best_match['term_name']}")
            print(f"Confidence: {best_match['confidence']:.3f}")
        else:
            print("No CMPO mappings found")

def demonstrate_multi_stage_cmpo():
    """Demonstrate multi-stage CMPO integration across pipeline stages."""
    print("\n" + "=" * 60)
    print("MULTI-STAGE CMPO INTEGRATION DEMONSTRATION")
    print("=" * 60)
    
    cmpo = CMPOOntology()
    
    # Simulate different types of biological observations at each stage
    stage_data = {
        "Stage 1 - Global Context": {
            "description": "Dense cell population with mitotic figures visible throughout",
            "context": "cell_population"
        },
        "Stage 3 - Individual Cells": {
            "description": "Cell arrested in metaphase with condensed chromosomes",
            "context": "cellular_phenotype"  
        },
        "Stage 4 - Population Insights": {
            "description": "20% of population shows apoptotic markers with fragmented nuclei",
            "context": "cell_population"
        }
    }
    
    all_mappings = {}
    
    print("🔬 Multi-Stage CMPO Analysis:")
    for stage_name, data in stage_data.items():
        print(f"\n{stage_name}:")
        print(f"Description: '{data['description']}'")
        print(f"Context: {data['context']}")
        
        # Map with stage-appropriate context
        mappings = map_to_cmpo(data['description'], cmpo, context=data['context'])
        
        if mappings:
            print(f"Found {len(mappings)} CMPO mappings:")
            for i, mapping in enumerate(mappings[:2], 1):
                print(f"  {i}. {mapping['term_name']} (confidence: {mapping['confidence']:.3f})")
                
                # Track for cross-stage analysis
                cmpo_id = mapping['CMPO_ID']
                if cmpo_id not in all_mappings:
                    all_mappings[cmpo_id] = {
                        'term': mapping['term_name'],
                        'stages': [],
                        'max_confidence': 0
                    }
                all_mappings[cmpo_id]['stages'].append(stage_name.split(' - ')[0])
                all_mappings[cmpo_id]['max_confidence'] = max(
                    all_mappings[cmpo_id]['max_confidence'], 
                    mapping['confidence']
                )
        else:
            print("  No CMPO mappings found")
    
    # Cross-stage analysis
    print("\n🔍 Cross-Stage CMPO Analysis:")
    multi_stage_terms = {k: v for k, v in all_mappings.items() if len(v['stages']) > 1}
    
    if multi_stage_terms:
        print("Terms detected across multiple stages:")
        for cmpo_id, data in multi_stage_terms.items():
            print(f"  • {data['term']} - detected in: {', '.join(data['stages'])}")
            print(f"    Max confidence: {data['max_confidence']:.3f}")
    else:
        print("No terms detected across multiple stages (expected - different biological levels)")
    
    print(f"\nTotal unique CMPO terms identified: {len(all_mappings)}")
    print("✅ Multi-stage integration provides comprehensive phenotype classification!")

def main():
    """Run all demonstrations."""
    print("CMPO Module Demonstration Suite")
    print("This script demonstrates the key capabilities of Anton's CMPO integration")
    
    try:
        demonstrate_basic_mapping()
        demonstrate_context_aware_mapping()
        demonstrate_hierarchical_navigation()
        demonstrate_semantic_analysis()
        demonstrate_integration_patterns()
        demonstrate_multi_stage_cmpo()  # New multi-stage demo
        
        print("\n" + "=" * 60)
        print("DEMONSTRATION COMPLETE")
        print("=" * 60)
        print("For more information, see anton/cmpo/README.md")
        print("✨ NEW: Multi-stage CMPO integration across all pipeline stages!")
        
    except Exception as e:
        print(f"Error during demonstration: {e}")
        print("Ensure CMPO ontology is properly loaded")

if __name__ == "__main__":
    main()
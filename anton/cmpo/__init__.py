"""
CMPO (Cellular Microscopy Phenotype Ontology) Integration Module for Anton

This module provides sophisticated ontology-based phenotype classification for microscopy analysis.
It bridges the gap between VLM-generated natural language descriptions and standardized 
scientific terminology through hierarchical semantic mapping.

Key Components:
- CMPOOntology: Loads and manages the full CMPO ontology with rich semantic relations
- map_to_cmpo: Context-aware mapping from descriptions to CMPO terms
- Hierarchical subgraph navigation for research-context-specific mapping

Usage:
    from anton.cmpo import CMPOOntology, map_to_cmpo
    
    cmpo = CMPOOntology()
    results = map_to_cmpo("cells arrested in metaphase", cmpo, context="cell_cycle")
"""

from .ontology import CMPOOntology
from .mapping import map_to_cmpo, validate_mappings_with_vlm

__all__ = ['CMPOOntology', 'map_to_cmpo', 'validate_mappings_with_vlm']
__version__ = '1.0.0'
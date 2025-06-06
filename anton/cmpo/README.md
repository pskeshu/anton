# CMPO Integration Module

## Overview

The Cellular Microscopy Phenotype Ontology (CMPO) integration module is a core component of Anton that provides **semantic mapping between natural language descriptions and standardized scientific terminology**. This module enables Anton to translate VLM-generated insights into scientifically compliant, searchable, and interoperable phenotype classifications.

## Problem Statement

Modern microscopy analysis faces a critical challenge: **bridging the semantic gap** between AI-generated natural language descriptions and standardized scientific terminology. While VLMs can provide expert-level biological insights ("cells arrested in metaphase with condensed chromosomes"), these descriptions need to be mapped to formal ontology terms for:

- **Scientific standardization**: Ensuring consistent terminology across studies
- **Data interoperability**: Enabling cross-dataset comparisons and meta-analyses  
- **Knowledge integration**: Connecting observations to broader biological knowledge graphs
- **Reproducible research**: Providing precise, unambiguous phenotype classifications

## Conceptual Framework

### 1. Multi-Level Hierarchical Mapping

CMPO is organized in a hierarchical structure with multiple branches:

```
CMPO Root
├── biological_process (GO terms)
├── cellular_phenotype (398 terms)
│   ├── cell_population_phenotype (73)
│   ├── cell_process_phenotype (157)
│   │   ├── cell_cycle_phenotype (46)
│   │   │   ├── cell_cycle_arrested_phenotype (6)
│   │   │   │   ├── G2_arrested_phenotype
│   │   │   │   ├── M_phase_arrested_phenotype
│   │   │   │   └── metaphase_arrested_phenotype
│   │   │   └── mitotic_process_phenotype (37)
│   │   └── cell_death_phenotype (1)
│   └── cellular_component_phenotype (186)
├── molecular_entity (CHEBI terms)
├── molecular_function (GO terms)
└── quality (PATO terms)
```

### 2. Research Context-Aware Subgraph Navigation

**Key Insight**: Researchers often have specific analytical intentions that determine which CMPO subgraphs are most relevant.

**Context Types**:
- **Process-focused**: Studying cell division, apoptosis, migration → `cell_process_phenotype` subgraph
- **Component-focused**: Analyzing organelles, structures → `cellular_component_phenotype` subgraph  
- **Multi-intent**: Cell cycle AND mitochondrial analysis → Multiple overlapping subgraphs
- **Population-level**: Colony behavior, density effects → `cell_population_phenotype` subgraph

### 3. Two-Strategy VLM Mapping Approach

#### Strategy 1: Description → CMPO Mapping
```
VLM Analysis: "Cells show metaphase arrest with hyperconnected chromosomes"
    ↓
Semantic Parsing: Extract ['metaphase', 'arrest', 'chromosomes', 'condensed']
    ↓
CMPO Mapping: → CMPO:0000XXX "metaphase arrested phenotype"
```

#### Strategy 2: CMPO-Guided Evidence Detection  
```
Research Context: "Studying cell cycle defects"
    ↓
Subgraph Selection: Focus on cell_cycle_phenotype branch
    ↓
VLM Query: "Do you see evidence of: metaphase arrest, anaphase defects, etc.?"
    ↓  
Targeted Classification: Direct mapping to specific terms
```

## Technical Implementation

### Semantic Mapping Pipeline

1. **Ontology Loading**: Parse full CMPO .obo file with rich semantic relations
2. **Multi-Modal Matching**:
   - **Direct matching**: Term names and synonyms
   - **Semantic matching**: Logical definitions and cross-ontology references
   - **Contextual matching**: Hierarchical subgraph relevance
3. **Confidence Scoring**: Weighted combination of multiple evidence sources
4. **Hierarchy Navigation**: Maintain relationships for downstream analysis

### Rich Ontological Information

Each CMPO term contains:

```python
{
    "CMPO:0001234": {
        "name": "metaphase arrested phenotype",
        "description": "A phenotype in which cells are arrested in metaphase",
        "synonyms": ["metaphase arrest", "M-phase block"],
        "subclass_of": ["cell_cycle_arrested_phenotype", "mitotic_phenotype"],
        "equivalent_to": "has_part(arrested and characteristic_of(mitotic_metaphase))",
        "xrefs": ["GO:0000819"],  # Cross-ontology links
        "subset": ["cmpo_core"]
    }
}
```

### Two-Stage Mapping Pipeline

```python
async def map_to_cmpo_enhanced(description, cmpo_ontology, vlm_interface, context=None):
    # Stage 1: Ontology-Aware Candidate Generation
    candidates = ontology_aware_mapping(description, cmpo_ontology, context)
    
    # Stage 2: VLM Biological Reasoning & Pruning
    if len(candidates) > 1:
        validated_mappings = await vlm_biological_validation(description, candidates, vlm_interface)
        return validated_mappings
    else:
        return candidates

def ontology_aware_mapping(description, cmpo_ontology, context=None):
    # 1. Enhanced token extraction with exact matching priority
    exact_tokens = extract_exact_biological_matches(description)
    fuzzy_tokens = extract_fuzzy_biological_tokens(description)
    
    # 2. Hierarchical scoring
    for term_id, term_data in cmpo_ontology.ontology.items():
        score = 0
        
        # Exact token matches (highest weight)
        exact_score = calculate_exact_matches(exact_tokens, term_data) * 1.0
        
        # Hierarchical specificity (deeper = more specific = higher score)
        specificity_score = calculate_hierarchy_depth(term_id, cmpo_ontology) * 0.3
        
        # Ontological distance (closer = more related = higher score)
        distance_score = calculate_ontological_distance(term_id, context_terms) * 0.2
        
        # Fuzzy similarity (lowest weight)
        fuzzy_score = calculate_fuzzy_similarity(fuzzy_tokens, term_data) * 0.1
        
        total_score = exact_score + specificity_score + distance_score + fuzzy_score
    
    return ranked_candidates

async def vlm_biological_validation(description, candidates, vlm_interface):
    validation_prompt = f"""
    Original biological description: "{description}"
    
    Candidate CMPO term mappings:
    {format_candidates_for_review(candidates)}
    
    Task: Evaluate biological plausibility and ranking of these mappings.
    
    Consider:
    - Biological consistency and logical compatibility
    - Temporal/spatial relationships in biological processes
    - Phenotypic co-occurrence patterns
    - Mechanistic plausibility
    - Specificity vs generality trade-offs
    
    Provide:
    1. Biologically valid mappings (with confidence 0-1)
    2. Brief scientific reasoning for each acceptance/rejection
    3. Final ranked list
    
    Focus on biological accuracy over textual similarity.
    """
    
    reasoning_result = await vlm_interface.reason_about_mappings(validation_prompt)
    return parse_and_apply_biological_reasoning(candidates, reasoning_result)
```

## Usage Examples

### Basic Mapping
```python
from anton.cmpo import CMPOOntology, map_to_cmpo

cmpo = CMPOOntology()
results = map_to_cmpo("cells arrested in metaphase with condensed chromosomes", cmpo)

# Output:
# [
#   {
#     "CMPO_ID": "CMPO:0001234",
#     "term_name": "metaphase arrested phenotype", 
#     "confidence": 0.92,
#     "supporting_evidence": "Direct match: metaphase; Semantic: arrested + mitotic",
#     "hierarchy_path": ["metaphase arrested phenotype", "cell cycle arrested phenotype", "cell cycle phenotype"]
#   }
# ]
```

### Context-Aware Mapping
```python
# Research studying apoptosis
results = map_to_cmpo("fragmented nuclei with membrane blebbing", cmpo, context="apoptosis")
# → Higher confidence for apoptotic_cell_phenotype terms

# Research studying cell division  
results = map_to_cmpo("abnormal spindle formation", cmpo, context="cell_cycle")
# → Higher confidence for mitotic_process_phenotype terms
```

### Integration with Anton Pipeline
```python
# Within QualitativeAnalyzer
population_insights = await vlm.analyze_population(image)
cmpo_mappings = map_to_cmpo(
    description=population_insights['description'],
    cmpo_ontology=self.cmpo_mapper,
    context=self.research_context
)
```

## Validation and Quality Assurance

### Confidence Thresholds
- **High confidence (>0.8)**: Direct term matches with strong semantic support
- **Medium confidence (0.5-0.8)**: Semantic matches with contextual support  
- **Low confidence (0.3-0.5)**: Weak matches requiring human review
- **Below threshold (<0.3)**: Excluded from results

### Evidence Tracking
Each mapping includes:
- **Supporting evidence**: Specific text that triggered the match
- **Mapping type**: Direct, semantic, or contextual
- **Hierarchy path**: Full taxonomic classification
- **Cross-references**: Links to related GO/PATO terms

## Future Enhancements

### 1. Machine Learning Integration
- **Embedding-based similarity**: Use biological language models (BioBERT, etc.)
- **Context learning**: Train models on researcher annotation patterns
- **Active learning**: Improve mappings based on user feedback

### 2. Advanced Semantic Reasoning  
- **Logical inference**: Use formal ontology reasoning for complex mappings
- **Negation handling**: Detect and properly handle negative evidence
- **Uncertainty quantification**: Bayesian confidence estimates

### 3. Multi-Ontology Integration
- **Cross-ontology alignment**: Map to GO, PATO, CHEBI simultaneously  
- **Knowledge graph construction**: Build comprehensive phenotype knowledge graphs
- **Standardized interfaces**: FAIR data principles compliance

### 4. Dynamic Ontology Updates
- **Version management**: Handle CMPO ontology updates gracefully
- **Backward compatibility**: Maintain mapping consistency across versions
- **Community integration**: Contribute mappings back to CMPO community

## Research Applications

### Enabled Use Cases
1. **Large-scale phenotype screens**: Standardized classification across thousands of images
2. **Cross-study meta-analysis**: Combine results from different research groups  
3. **Drug discovery**: Map compound effects to standardized phenotype profiles
4. **Disease research**: Connect cellular phenotypes to pathological processes
5. **Evolutionary studies**: Compare phenotypes across species using common vocabulary

### Scientific Impact
- **Reproducibility**: Eliminates ambiguity in phenotype descriptions
- **Discoverability**: Enables semantic search across phenotype databases
- **Integration**: Connects microscopy data to broader biological knowledge
- **Collaboration**: Provides common language for interdisciplinary research

---

## Development Notes

### Design Decisions

**Why hierarchical subgraph mapping?**
- CMPO contains >600 terms across diverse biological domains
- Research context dramatically improves mapping accuracy  
- Enables both broad screening and focused deep analysis

**Why two-strategy VLM approach?**
- Strategy 1 (description→CMPO) handles unexpected discoveries
- Strategy 2 (CMPO-guided) ensures comprehensive coverage of known phenotypes
- Combination provides both discovery and validation capabilities

**Why rich semantic relations?**
- Simple keyword matching fails for scientific terminology
- Logical definitions enable precise semantic matching
- Cross-ontology links expand vocabulary and validation

### Code Organization
- `ontology.py`: CMPO data loading, parsing, and management
- `mapping.py`: Core mapping algorithms and semantic analysis
- `__init__.py`: Module interface and public API
- `README.md`: Comprehensive documentation (this file)

### Testing Strategy
- Unit tests for individual mapping functions
- Integration tests with full CMPO ontology
- Validation against expert-annotated datasets
- Performance benchmarks for large-scale analysis

---

*This module represents a significant advancement in automated microscopy phenotype classification, bridging AI-generated insights with rigorous scientific standards.*
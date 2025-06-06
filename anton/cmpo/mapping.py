"""Free-form to CMPO mapping for Anton's pipeline."""

import re
from typing import Dict, List, Tuple, Set
from difflib import SequenceMatcher

def map_to_cmpo(description: str, cmpo_ontology, context: str = None) -> List[Dict]:
    """Convert a free-form description to CMPO terms using semantic mapping."""
    if not description or not cmpo_ontology:
        return []
    
    description_lower = description.lower()
    mappings = []
    
    # 1. Direct name/synonym matching
    direct_matches = _find_direct_matches(description_lower, cmpo_ontology)
    
    # 2. Semantic component matching
    semantic_matches = _find_semantic_matches(description_lower, cmpo_ontology)
    
    # 3. Hierarchical context matching (if context provided)
    context_matches = _find_context_matches(description_lower, cmpo_ontology, context) if context else []
    
    # Combine and score all matches
    all_matches = {}
    
    # Weight direct matches highest (preserve enhanced scoring differences)
    for term_id, confidence, evidence in direct_matches:
        if term_id not in all_matches:
            all_matches[term_id] = {'confidence': 0, 'evidence': []}
        all_matches[term_id]['confidence'] += confidence  # Don't flatten with 0.8 multiplier
        all_matches[term_id]['evidence'].append(f"Direct match: {evidence}")
    
    # Weight semantic matches moderately  
    for term_id, confidence, evidence in semantic_matches:
        if term_id not in all_matches:
            all_matches[term_id] = {'confidence': 0, 'evidence': []}
        all_matches[term_id]['confidence'] += confidence * 0.3  # Lower weight for semantic
        all_matches[term_id]['evidence'].append(f"Semantic: {evidence}")
    
    # Weight context matches lower but still valuable
    for term_id, confidence, evidence in context_matches:
        if term_id not in all_matches:
            all_matches[term_id] = {'confidence': 0, 'evidence': []}
        all_matches[term_id]['confidence'] += confidence * 0.2  # Lower weight for context
        all_matches[term_id]['evidence'].append(f"Context: {evidence}")
    
    # Convert to final format
    for term_id, match_data in all_matches.items():
        term_info = cmpo_ontology.get_term(term_id)
        if term_info:
            mappings.append({
                "CMPO_ID": term_id,
                "term_name": term_info['name'],
                "confidence": match_data['confidence'],  # Preserve full confidence for sorting
                "supporting_evidence": "; ".join(match_data['evidence'][:3]),
                "description": term_info.get('description', ''),
                "hierarchy_path": _get_hierarchy_path(term_id, cmpo_ontology)
            })
    
    # Sort by confidence and return top matches
    mappings.sort(key=lambda x: x['confidence'], reverse=True)
    return mappings[:5]

def _find_direct_matches(description: str, cmpo_ontology) -> List[Tuple[str, float, str]]:
    """Find direct matches with ontology-aware scoring."""
    matches = []
    description_tokens = set(_extract_biological_tokens(description))
    
    for term_id, term_data in cmpo_ontology.ontology.items():
        base_score = 0.0
        matched_evidence = []
        
        # 1. Exact token matches (highest priority)
        term_tokens = set(_extract_biological_tokens(term_data.get('name', '')))
        exact_matches = description_tokens.intersection(term_tokens)
        if exact_matches:
            # Higher score for exact matches
            exact_score = len(exact_matches) / max(len(term_tokens), 1) * 2.0
            base_score += exact_score
            matched_evidence.extend(exact_matches)
        
        # 2. Check term name substring matches
        term_name = term_data.get('name', '').lower()
        if term_name and term_name in description:
            substring_score = len(term_name) / len(description) * 1.5
            base_score += substring_score
            matched_evidence.append(f"name:{term_name}")
        
        # 3. Check synonyms with exact token priority
        for synonym in term_data.get('synonyms', []):
            synonym_tokens = set(_extract_biological_tokens(synonym))
            syn_exact_matches = description_tokens.intersection(synonym_tokens)
            if syn_exact_matches:
                syn_score = len(syn_exact_matches) / max(len(synonym_tokens), 1) * 1.8
                base_score += syn_score
                matched_evidence.extend(syn_exact_matches)
            elif synonym.lower() in description:
                substring_score = len(synonym) / len(description) * 1.2
                base_score += substring_score
                matched_evidence.append(f"synonym:{synonym}")
        
        # 4. Ontology-aware bonuses
        if base_score > 0:
            # Specificity bonus (deeper in hierarchy = more specific = higher score)
            specificity_bonus = _calculate_specificity_bonus(term_id, cmpo_ontology)
            
            # Multi-token exact match bonus (matches multiple key terms)
            multi_token_bonus = 0.0
            if len(exact_matches) > 1:
                multi_token_bonus = len(exact_matches) * 0.5  # Strong bonus for multiple exact matches
            
            # Apply ontology bonuses
            final_score = base_score + specificity_bonus + multi_token_bonus
            
            matches.append((term_id, min(final_score, 5.0), f"exact:{','.join(matched_evidence[:3])}"))
    
    return matches

def _find_semantic_matches(description: str, cmpo_ontology) -> List[Tuple[str, float, str]]:
    """Find matches based on semantic component analysis."""
    matches = []
    
    # Extract meaningful terms from description
    desc_tokens = _extract_biological_tokens(description)
    
    for term_id, term_data in cmpo_ontology.ontology.items():
        # Analyze equivalent_to relations for semantic components
        for equiv in term_data.get('equivalent_to', []):
            semantic_score = _score_semantic_overlap(desc_tokens, equiv)
            if semantic_score > 0.3:
                matches.append((term_id, semantic_score, f"Semantic components in {equiv}"))
        
        # Check description overlap
        term_desc = term_data.get('description', '').lower()
        if term_desc:
            desc_overlap = _calculate_text_similarity(description, term_desc)
            if desc_overlap > 0.4:
                matches.append((term_id, desc_overlap, "Description similarity"))
    
    return matches

def _find_context_matches(description: str, cmpo_ontology, context: str) -> List[Tuple[str, float, str]]:
    """Find matches considering hierarchical context."""
    matches = []
    
    # Define context-based subgraph priorities
    context_subgraphs = {
        'cell_cycle': ['cell_cycle_phenotype', 'mitotic_process_phenotype'],
        'apoptosis': ['cell_death_phenotype', 'apoptotic'],
        'morphology': ['cellular_component_phenotype', 'abnormal_cell_morphology'],
        'process': ['cell_process_phenotype', 'biological_process']
    }
    
    relevant_subgraphs = []
    context_lower = context.lower() if context else ""
    
    for ctx_key, subgraphs in context_subgraphs.items():
        if ctx_key in context_lower:
            relevant_subgraphs.extend(subgraphs)
    
    # Score terms within relevant subgraphs higher
    for term_id, term_data in cmpo_ontology.ontology.items():
        for subgraph in relevant_subgraphs:
            if _term_in_subgraph(term_id, subgraph, cmpo_ontology):
                base_score = 0.5
                # Boost if term also matches description
                term_name = term_data.get('name', '').lower()
                if any(token in term_name for token in description.split()):
                    base_score += 0.3
                matches.append((term_id, base_score, f"Context subgraph: {subgraph}"))
    
    return matches

def _extract_biological_tokens(text: str) -> Set[str]:
    """Extract biologically relevant tokens from text."""
    # Common biological stop words to exclude
    bio_stop_words = {'cell', 'cells', 'cellular', 'the', 'and', 'or', 'with', 'in', 'of'}
    
    # Extract tokens
    tokens = set(re.findall(r'\b\w+\b', text.lower()))
    
    # Filter for biological relevance (length > 3, not stop words)
    bio_tokens = {token for token in tokens 
                 if len(token) > 3 and token not in bio_stop_words}
    
    return bio_tokens

def _score_semantic_overlap(desc_tokens: Set[str], equivalent_to: str) -> float:
    """Score overlap between description tokens and semantic definition."""
    equiv_tokens = _extract_biological_tokens(equivalent_to)
    
    if not equiv_tokens:
        return 0.0
    
    overlap = len(desc_tokens.intersection(equiv_tokens))
    return overlap / max(len(equiv_tokens), 1)

def _calculate_text_similarity(text1: str, text2: str) -> float:
    """Calculate text similarity using sequence matching."""
    return SequenceMatcher(None, text1, text2).ratio()

def _term_in_subgraph(term_id: str, subgraph_name: str, cmpo_ontology) -> bool:
    """Check if a term belongs to a specific subgraph via hierarchy."""
    term_data = cmpo_ontology.get_term(term_id)
    if not term_data:
        return False
    
    # Check if term name contains subgraph keyword
    term_name = term_data.get('name', '').lower()
    if subgraph_name.lower() in term_name:
        return True
    
    # Check parent terms recursively (simple implementation)
    for parent in term_data.get('parent_terms', []):
        parent_data = cmpo_ontology.get_term(parent)
        if parent_data and subgraph_name.lower() in parent_data.get('name', '').lower():
            return True
    
    return False

def _get_hierarchy_path(term_id: str, cmpo_ontology) -> List[str]:
    """Get the hierarchical path for a term."""
    path = []
    current_term = cmpo_ontology.get_term(term_id)
    
    if current_term:
        path.append(current_term.get('name', term_id))
        
        # Add immediate parents (simplified - could be recursive)
        for parent_id in current_term.get('parent_terms', [])[:2]:  # Limit to 2 parents
            parent_term = cmpo_ontology.get_term(parent_id)
            if parent_term:
                path.append(parent_term.get('name', parent_id))
    
    return path

def _calculate_specificity_bonus(term_id: str, cmpo_ontology) -> float:
    """Calculate specificity bonus based on hierarchy depth."""
    try:
        depth = _calculate_hierarchy_depth(term_id, cmpo_ontology)
        # Deeper terms are more specific, get higher bonus
        # Max bonus of 0.5 for terms at depth 4+
        return min(depth * 0.1, 0.5)
    except:
        return 0.0

def _calculate_hierarchy_depth(term_id: str, cmpo_ontology, visited=None) -> int:
    """Calculate depth of term in CMPO hierarchy."""
    if visited is None:
        visited = set()
    
    if term_id in visited:  # Avoid cycles
        return 0
    
    visited.add(term_id)
    term_data = cmpo_ontology.get_term(term_id)
    
    if not term_data or not term_data.get('parent_terms'):
        return 1  # Root level
    
    # Find maximum depth among parents
    max_parent_depth = 0
    for parent_id in term_data.get('parent_terms', []):
        parent_depth = _calculate_hierarchy_depth(parent_id, cmpo_ontology, visited.copy())
        max_parent_depth = max(max_parent_depth, parent_depth)
    
    return max_parent_depth + 1

def _detect_mutual_exclusion(term1_id: str, term2_id: str, cmpo_ontology) -> bool:
    """Detect if two terms are mutually exclusive based on ontology structure."""
    term1 = cmpo_ontology.get_term(term1_id)
    term2 = cmpo_ontology.get_term(term2_id)
    
    if not term1 or not term2:
        return False
    
    # Check if they share the same immediate parent (sibling terms often mutually exclusive)
    term1_parents = set(term1.get('parent_terms', []))
    term2_parents = set(term2.get('parent_terms', []))
    
    shared_parents = term1_parents.intersection(term2_parents)
    
    # If they share parents and are both specific (depth > 2), likely mutually exclusive
    if shared_parents and len(shared_parents) > 0:
        depth1 = _calculate_hierarchy_depth(term1_id, cmpo_ontology)
        depth2 = _calculate_hierarchy_depth(term2_id, cmpo_ontology)
        
        # Heuristic: sibling terms at depth 3+ often mutually exclusive
        if depth1 > 2 and depth2 > 2:
            return True
    
    return False

# Add VLM validation function for the two-stage pipeline
async def validate_mappings_with_vlm(description: str, candidate_mappings: List[Dict], vlm_interface, max_candidates: int = 5) -> List[Dict]:
    """Stage 2: VLM biological reasoning and pruning."""
    if len(candidate_mappings) <= 1:
        return candidate_mappings
    
    # Format candidates for VLM review
    candidates_text = "\n".join([
        f"{i+1}. {mapping['term_name']} (CMPO:{mapping['CMPO_ID']}) - Confidence: {mapping['confidence']:.3f}"
        for i, mapping in enumerate(candidate_mappings[:max_candidates])
    ])
    
    validation_prompt = f"""Original biological description: "{description}"

Candidate CMPO term mappings:
{candidates_text}

Task: Evaluate biological plausibility and ranking of these mappings.

Consider:
- Biological consistency and logical compatibility
- Temporal/spatial relationships in biological processes  
- Phenotypic co-occurrence patterns
- Mechanistic plausibility
- Specificity vs generality trade-offs

Provide:
1. Biologically valid mappings with updated confidence (0-1)
2. Brief scientific reasoning for each acceptance/rejection
3. Final ranked list

Focus on biological accuracy over textual similarity.

Format your response as:
VALID: [term_name] - confidence: [0-1] - reasoning: [brief explanation]
INVALID: [term_name] - reasoning: [brief explanation]
"""
    
    try:
        # This would be implemented as part of VLM interface
        reasoning_result = await vlm_interface.analyze_biological_reasoning(validation_prompt)
        
        # Parse VLM response and update mappings
        validated_mappings = _parse_vlm_validation_response(reasoning_result, candidate_mappings)
        
        return validated_mappings
        
    except Exception as e:
        # Fallback to original mappings if VLM validation fails
        logging.warning(f"VLM validation failed: {e}, using original mappings")
        return candidate_mappings

def _parse_vlm_validation_response(vlm_response: str, original_mappings: List[Dict]) -> List[Dict]:
    """Parse VLM validation response and update mapping confidences."""
    validated = []
    
    # Simple parsing - in production would be more robust
    for line in vlm_response.split('\n'):
        if line.startswith('VALID:'):
            # Extract confidence and reasoning
            parts = line.split(' - ')
            if len(parts) >= 3:
                term_name = parts[0].replace('VALID: ', '').strip()
                confidence_str = parts[1].replace('confidence: ', '').strip()
                reasoning = parts[2].replace('reasoning: ', '').strip()
                
                # Find corresponding original mapping
                for mapping in original_mappings:
                    if mapping['term_name'].lower() == term_name.lower():
                        updated_mapping = mapping.copy()
                        try:
                            updated_mapping['confidence'] = float(confidence_str)
                            updated_mapping['vlm_reasoning'] = reasoning
                            validated.append(updated_mapping)
                        except ValueError:
                            validated.append(mapping)  # Keep original if parsing fails
                        break
    
    # Sort by updated confidence
    validated.sort(key=lambda x: x['confidence'], reverse=True)
    return validated 
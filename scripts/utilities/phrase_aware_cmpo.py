#!/usr/bin/env python3
"""
Phrase-Aware CMPO Mapping

Solves the fundamental bag-of-words problem by implementing hierarchical 
phrase matching that respects multi-word term semantics.
"""

import re
import json
from typing import List, Dict, Tuple, Set
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

@dataclass
class PhrasePattern:
    """Represents a phrase pattern for CMPO term matching."""
    pattern: str
    required_words: Set[str]
    forbidden_words: Set[str]
    pattern_type: str  # 'exact', 'ordered', 'unordered'
    specificity_score: float

class PhraseAwareCMPOMapper:
    """CMPO mapper that understands multi-word phrase semantics."""
    
    def __init__(self, cmpo_ontology):
        self.cmpo_ontology = cmpo_ontology
        self.phrase_patterns = {}
        self.specificity_hierarchy = {}
        self.conflicting_patterns = defaultdict(set)
        self._build_phrase_patterns()
    
    def _build_phrase_patterns(self):
        """Build phrase patterns for all CMPO terms."""
        print("🔧 Building phrase patterns for 399 CMPO terms...")
        
        for term_id, term_data in self.cmpo_ontology.ontology.items():
            term_name = term_data.get('name', '').lower()
            if not term_name:
                continue
            
            # Create multiple patterns for each term
            patterns = self._extract_phrase_patterns(term_name, term_id)
            self.phrase_patterns[term_id] = patterns
            
            # Build specificity hierarchy
            self._calculate_term_specificity(term_id, term_name)
        
        # Detect conflicting patterns
        self._detect_pattern_conflicts()
        
        print(f"✅ Built {len(self.phrase_patterns)} phrase pattern sets")
        print(f"⚠️  Detected {len(self.conflicting_patterns)} conflicting pattern groups")
    
    def _extract_phrase_patterns(self, term_name: str, term_id: str) -> List[PhrasePattern]:
        """Extract multiple phrase patterns from a CMPO term name."""
        patterns = []
        words = term_name.split()
        
        # 1. Exact phrase pattern (highest specificity)
        patterns.append(PhrasePattern(
            pattern=term_name,
            required_words=set(words),
            forbidden_words=set(),
            pattern_type='exact',
            specificity_score=len(words) * 2.0
        ))
        
        # 2. Key phrase patterns (for complex terms)
        if len(words) > 3:
            key_phrases = self._extract_key_phrases(term_name)
            for phrase in key_phrases:
                patterns.append(PhrasePattern(
                    pattern=phrase,
                    required_words=set(phrase.split()),
                    forbidden_words=set(),
                    pattern_type='ordered',
                    specificity_score=len(phrase.split()) * 1.5
                ))
        
        # 3. Essential word combinations (for disambiguation)
        essential_combos = self._extract_essential_combinations(words)
        for combo in essential_combos:
            patterns.append(PhrasePattern(
                pattern=' '.join(combo),
                required_words=set(combo),
                forbidden_words=set(),
                pattern_type='unordered',
                specificity_score=len(combo) * 1.0
            ))
        
        return patterns
    
    def _extract_key_phrases(self, term_name: str) -> List[str]:
        """Extract key phrases that must appear together."""
        # Define important phrase patterns that should be kept together
        key_phrase_patterns = [
            r'nuclear pore',
            r'nuclear envelope',
            r'nuclear membrane', 
            r'cell cycle',
            r'cell membrane',
            r'endoplasmic reticulum',
            r'golgi apparatus',
            r'mitochondrial membrane',
            r'cytoplasmic membrane',
            r'G1 phase', r'G2 phase', r'S phase', r'M phase',
            r'abnormal \w+',
            r'increased \w+',
            r'decreased \w+',
            r'protein localized in \w+',
            r'protein localized to \w+'
        ]
        
        key_phrases = []
        for pattern in key_phrase_patterns:
            matches = re.finditer(pattern, term_name, re.IGNORECASE)
            for match in matches:
                key_phrases.append(match.group().lower())
        
        return key_phrases
    
    def _extract_essential_combinations(self, words: List[str]) -> List[Tuple[str, ...]]:
        """Extract combinations of words that are essential for disambiguation."""
        essential_combos = []
        
        # Define words that must appear together for specificity
        must_pair_words = {
            'nuclear': ['pore', 'envelope', 'membrane', 'body', 'speckle'],
            'cell': ['cycle', 'membrane', 'wall', 'death'],
            'protein': ['localized', 'translocation', 'transport'],
            'abnormal': ['shape', 'size', 'morphology', 'organization'],
            'mitochondrial': ['membrane', 'matrix', 'transport'],
            'endoplasmic': ['reticulum'],
            'golgi': ['apparatus']
        }
        
        for word in words:
            if word in must_pair_words:
                for pair_word in must_pair_words[word]:
                    if pair_word in words:
                        essential_combos.append((word, pair_word))
        
        return essential_combos
    
    def _calculate_term_specificity(self, term_id: str, term_name: str):
        """Calculate specificity score for term hierarchy."""
        # More specific terms have more words and specialized vocabulary
        words = term_name.split()
        word_count_score = len(words)
        
        # Specialized vocabulary bonus
        specialized_words = {
            'nuclear', 'mitochondrial', 'endoplasmic', 'golgi', 'cytoplasmic',
            'membrane', 'envelope', 'pore', 'reticulum', 'apparatus',
            'localized', 'translocation', 'transport', 'organization'
        }
        specialization_score = sum(1 for word in words if word in specialized_words)
        
        # Modifier words (increase specificity)
        modifier_words = {'abnormal', 'increased', 'decreased', 'enlarged', 'reduced'}
        modifier_score = sum(1 for word in words if word in modifier_words)
        
        total_specificity = word_count_score + specialization_score + modifier_score
        self.specificity_hierarchy[term_id] = total_specificity
    
    def _detect_pattern_conflicts(self):
        """Detect patterns that might conflict and need disambiguation."""
        # Group terms by overlapping patterns
        pattern_to_terms = defaultdict(set)
        
        for term_id, patterns in self.phrase_patterns.items():
            for pattern in patterns:
                pattern_key = pattern.pattern
                pattern_to_terms[pattern_key].add(term_id)
        
        # Find conflicts
        for pattern_key, term_ids in pattern_to_terms.items():
            if len(term_ids) > 1:
                self.conflicting_patterns[pattern_key] = term_ids
    
    def map_to_cmpo(self, description: str, max_results: int = 10) -> List[Dict]:
        """Map description to CMPO terms using phrase-aware matching."""
        description_lower = description.lower()
        matches = []
        
        # Phase 1: Exact phrase matching (highest confidence)
        exact_matches = self._find_exact_phrase_matches(description_lower)
        matches.extend(exact_matches)
        
        # Phase 2: Key phrase matching (medium confidence)
        if len(matches) < max_results:
            key_phrase_matches = self._find_key_phrase_matches(description_lower)
            matches.extend(key_phrase_matches)
        
        # Phase 3: Essential combination matching (lower confidence)
        if len(matches) < max_results:
            combo_matches = self._find_combination_matches(description_lower)
            matches.extend(combo_matches)
        
        # Remove duplicates and sort by confidence
        unique_matches = self._deduplicate_and_rank(matches)
        
        return unique_matches[:max_results]
    
    def _find_exact_phrase_matches(self, description: str) -> List[Dict]:
        """Find exact phrase matches with highest confidence."""
        matches = []
        
        for term_id, patterns in self.phrase_patterns.items():
            for pattern in patterns:
                if pattern.pattern_type == 'exact' and pattern.pattern in description:
                    confidence = min(5.0, pattern.specificity_score)
                    matches.append({
                        'CMPO_ID': term_id,
                        'term_name': self.cmpo_ontology.ontology[term_id].get('name', ''),
                        'confidence': confidence,
                        'supporting_evidence': f"Exact phrase: '{pattern.pattern}'",
                        'match_type': 'exact_phrase'
                    })
        
        return matches
    
    def _find_key_phrase_matches(self, description: str) -> List[Dict]:
        """Find key phrase matches with medium confidence."""
        matches = []
        
        for term_id, patterns in self.phrase_patterns.items():
            for pattern in patterns:
                if pattern.pattern_type == 'ordered' and pattern.pattern in description:
                    # Ensure no conflicting patterns are present
                    if self._check_pattern_conflicts(description, pattern, term_id):
                        continue
                    
                    confidence = min(4.5, pattern.specificity_score)
                    matches.append({
                        'CMPO_ID': term_id,
                        'term_name': self.cmpo_ontology.ontology[term_id].get('name', ''),
                        'confidence': confidence,
                        'supporting_evidence': f"Key phrase: '{pattern.pattern}'",
                        'match_type': 'key_phrase'
                    })
        
        return matches
    
    def _find_combination_matches(self, description: str) -> List[Dict]:
        """Find essential word combination matches with lower confidence."""
        matches = []
        
        description_words = set(description.split())
        
        for term_id, patterns in self.phrase_patterns.items():
            for pattern in patterns:
                if pattern.pattern_type == 'unordered':
                    # Check if all required words are present
                    if pattern.required_words.issubset(description_words):
                        # Check forbidden words are not present
                        if not pattern.forbidden_words.intersection(description_words):
                            confidence = min(4.0, pattern.specificity_score)
                            matches.append({
                                'CMPO_ID': term_id,
                                'term_name': self.cmpo_ontology.ontology[term_id].get('name', ''),
                                'confidence': confidence,
                                'supporting_evidence': f"Word combination: {', '.join(pattern.required_words)}",
                                'match_type': 'word_combination'
                            })
        
        return matches
    
    def _check_pattern_conflicts(self, description: str, pattern: PhrasePattern, term_id: str) -> bool:
        """Check if pattern conflicts with other patterns in the description."""
        pattern_key = pattern.pattern
        
        if pattern_key in self.conflicting_patterns:
            conflicting_terms = self.conflicting_patterns[pattern_key]
            
            # Check if any conflicting terms have better matches
            for conflict_term_id in conflicting_terms:
                if conflict_term_id != term_id:
                    conflict_specificity = self.specificity_hierarchy.get(conflict_term_id, 0)
                    current_specificity = self.specificity_hierarchy.get(term_id, 0)
                    
                    # If conflicting term is more specific and also matches, reject current
                    if conflict_specificity > current_specificity:
                        conflict_patterns = self.phrase_patterns.get(conflict_term_id, [])
                        for conflict_pattern in conflict_patterns:
                            if conflict_pattern.pattern in description:
                                return True  # Conflict detected
        
        return False  # No conflict
    
    def _deduplicate_and_rank(self, matches: List[Dict]) -> List[Dict]:
        """Remove duplicates and rank by confidence and specificity."""
        # Group by CMPO_ID
        term_matches = defaultdict(list)
        for match in matches:
            term_id = match['CMPO_ID']
            term_matches[term_id].append(match)
        
        # Keep best match for each term
        unique_matches = []
        for term_id, term_match_list in term_matches.items():
            # Sort by confidence then by specificity
            best_match = max(term_match_list, key=lambda x: (
                x['confidence'], 
                self.specificity_hierarchy.get(term_id, 0)
            ))
            unique_matches.append(best_match)
        
        # Sort final results
        unique_matches.sort(key=lambda x: x['confidence'], reverse=True)
        
        return unique_matches

def demonstrate_phrase_aware_mapping():
    """Demonstrate the phrase-aware mapping vs bag-of-words."""
    from anton.cmpo.ontology import CMPOOntology
    
    # Load CMPO ontology
    cmpo = CMPOOntology()
    
    # Initialize phrase-aware mapper
    phrase_mapper = PhraseAwareCMPOMapper(cmpo)
    
    # Test cases that were problematic before
    test_descriptions = [
        "The protein shows nuclear localization with increased cytoplasmic distribution",
        "Nuclear envelope protein transport is disrupted",
        "Abnormal mitochondrial membrane potential detected",
        "Cell cycle arrest in G1 phase observed",
        "Protein localized to nuclear pore complex",
        "Increased nuclear membrane permeability"
    ]
    
    print("🧪 Testing Phrase-Aware vs Bag-of-Words Mapping")
    print("=" * 60)
    
    for i, description in enumerate(test_descriptions, 1):
        print(f"\n{i}. Test: '{description}'")
        print("-" * 50)
        
        # Get phrase-aware results
        phrase_results = phrase_mapper.map_to_cmpo(description, max_results=3)
        
        print("✅ Phrase-Aware Results:")
        for result in phrase_results:
            print(f"   • {result['term_name']}: {result['confidence']:.1f}")
            print(f"     Evidence: {result['supporting_evidence']}")
        
        if not phrase_results:
            print("   • No valid matches found")
        
        print()

if __name__ == "__main__":
    demonstrate_phrase_aware_mapping()
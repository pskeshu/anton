"""Manage CMPO ontology data and provide lookup functionality."""

import json
import requests
import pickle
from pathlib import Path
from typing import Dict, List, Optional
import logging

class CMPOOntology:
    """Manage CMPO ontology data and provide lookup functionality"""
    
    def __init__(self, data_path="data/cmpo.json", cache_path="data/cmpo_cache.pkl"):
        self.data_path = Path(data_path)
        self.cache_path = Path(cache_path)
        self.ontology = {}
        self.term_index = {}  # For fast lookup
        self.feature_index = {}  # Map features to terms
        self.keyword_index = {}  # Map keywords to terms
        
        self._load_ontology()
    
    def _load_ontology(self):
        """Load CMPO ontology from JSON file or download if needed"""
        if self.data_path.exists():
            logging.info(f"Loading CMPO ontology from {self.data_path}")
            with open(self.data_path, 'r') as f:
                self.ontology = json.load(f)
        else:
            logging.info("CMPO ontology not found, downloading...")
            self._download_and_process_cmpo()
        
        self._build_indices()
    
    def _download_and_process_cmpo(self):
        """Download CMPO from official repository and convert to JSON"""
        try:
            # Option 1: Parse OBO file directly (preferred for rich semantic info)
            self._download_and_parse_obo()
        except Exception as e:
            logging.warning(f"Failed to download OBO: {e}")
            try:
                # Option 2: Use OLS API (Ontology Lookup Service)
                self._download_from_ols()
            except Exception as e2:
                logging.warning(f"Failed to download from OLS: {e2}")
                try:
                    # Option 3: Parse OWL file directly
                    self._download_owl_file()
                except Exception as e3:
                    logging.error(f"Failed to download OWL: {e3}")
                    # Option 4: Use minimal hardcoded ontology
                    self._create_minimal_ontology()
    
    def _download_and_parse_obo(self):
        """Download and parse CMPO OBO file for rich semantic information"""
        obo_url = "https://raw.githubusercontent.com/EBISPOT/CMPO/master/cmpo.obo"
        
        logging.info(f"Downloading CMPO OBO file from {obo_url}")
        response = requests.get(obo_url)
        response.raise_for_status()
        
        # Parse OBO content
        ontology_data = self._parse_obo_content(response.text)
        
        # Save processed data
        self.data_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.data_path, 'w') as f:
            json.dump(ontology_data, f, indent=2)
        
        self.ontology = ontology_data
        logging.info(f"Successfully loaded {len(ontology_data)} CMPO terms")
    
    def _parse_obo_content(self, obo_text: str) -> Dict:
        """Parse OBO format text into structured data"""
        ontology_data = {}
        current_term = None
        current_term_id = None
        
        for line in obo_text.split('\n'):
            line = line.strip()
            
            if line == '[Term]':
                # Save previous term if exists
                if current_term and current_term_id:
                    ontology_data[current_term_id] = current_term
                # Start new term
                current_term = {
                    'name': '',
                    'description': '',
                    'synonyms': [],
                    'features': [],
                    'parent_terms': [],
                    'subclass_of': [],
                    'equivalent_to': [],
                    'subset': [],
                    'xrefs': [],
                    'iri': ''
                }
                current_term_id = None
                
            elif line.startswith('id:') and current_term is not None:
                current_term_id = line.split(':', 1)[1].strip()
                current_term['iri'] = f"http://purl.obolibrary.org/obo/{current_term_id.replace(':', '_')}"
                
            elif line.startswith('name:') and current_term is not None:
                current_term['name'] = line.split(':', 1)[1].strip()
                
            elif line.startswith('def:') and current_term is not None:
                # Extract definition (remove quotes and references)
                def_text = line.split(':', 1)[1].strip()
                if def_text.startswith('"') and '" [' in def_text:
                    current_term['description'] = def_text.split('" [')[0][1:]
                else:
                    current_term['description'] = def_text
                    
            elif line.startswith('synonym:') and current_term is not None:
                # Extract synonym text (format: synonym: "text" EXACT [])
                syn_text = line.split(':', 1)[1].strip()
                if syn_text.startswith('"'):
                    synonym = syn_text.split('"')[1]
                    current_term['synonyms'].append(synonym)
                    
            elif line.startswith('is_a:') and current_term is not None:
                # Extract parent term ID
                parent = line.split(':', 1)[1].strip().split('!')[0].strip()
                current_term['parent_terms'].append(parent)
                current_term['subclass_of'].append(parent)
                
            elif line.startswith('equivalent_to:') and current_term is not None:
                equiv = line.split(':', 1)[1].strip()
                current_term['equivalent_to'].append(equiv)
                
            elif line.startswith('subset:') and current_term is not None:
                subset = line.split(':', 1)[1].strip()
                current_term['subset'].append(subset)
                
            elif line.startswith('xref:') and current_term is not None:
                xref = line.split(':', 1)[1].strip()
                current_term['xrefs'].append(xref)
        
        # Don't forget the last term
        if current_term and current_term_id:
            ontology_data[current_term_id] = current_term
            
        return ontology_data
    
    def _download_from_ols(self):
        """Download CMPO terms using OLS REST API"""
        base_url = "https://www.ebi.ac.uk/ols/api/ontologies/cmpo/terms"
        ontology_data = {}
        
        # Get all terms
        page = 0
        while True:
            response = requests.get(f"{base_url}?page={page}&size=500")
            response.raise_for_status()
            data = response.json()
            
            if '_embedded' not in data or 'terms' not in data['_embedded']:
                break
                
            for term in data['_embedded']['terms']:
                term_id = term['obo_id'] if 'obo_id' in term else term['iri'].split('/')[-1]
                
                ontology_data[term_id] = {
                    'name': term.get('label', ''),
                    'description': term.get('description', [''])[0] if term.get('description') else '',
                    'synonyms': term.get('synonyms', []),
                    'features': self._extract_features_from_term(term),
                    'parent_terms': self._extract_parents(term),
                    'iri': term.get('iri', '')
                }
            
            # Check if there are more pages
            if data['page']['number'] >= data['page']['totalPages'] - 1:
                break
            page += 1
        
        # Save to file
        self.data_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.data_path, 'w') as f:
            json.dump(ontology_data, f, indent=2)
        
        self.ontology = ontology_data
    
    def _download_owl_file(self):
        """Download and parse OWL file directly"""
        try:
            import owlready2
            
            # Download CMPO OWL file
            owl_url = "https://raw.githubusercontent.com/EBISPOT/CMPO/master/cmpo.owl"
            response = requests.get(owl_url)
            response.raise_for_status()
            
            # Save temporarily
            temp_owl = "temp_cmpo.owl"
            with open(temp_owl, 'wb') as f:
                f.write(response.content)
            
            # Parse with owlready2
            onto = owlready2.get_ontology(f"file://{Path(temp_owl).absolute()}").load()
            
            ontology_data = {}
            for cls in onto.classes():
                if hasattr(cls, 'label') and cls.label:
                    term_id = cls.name
                    ontology_data[term_id] = {
                        'name': cls.label[0] if cls.label else cls.name,
                        'description': cls.comment[0] if hasattr(cls, 'comment') and cls.comment else '',
                        'synonyms': list(cls.hasExactSynonym) if hasattr(cls, 'hasExactSynonym') else [],
                        'features': self._extract_owl_features(cls),
                        'parent_terms': [p.name for p in cls.is_a if hasattr(p, 'name')],
                        'iri': str(cls.iri)
                    }
            
            # Clean up
            Path(temp_owl).unlink()
            
            # Save processed data
            self.data_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.data_path, 'w') as f:
                json.dump(ontology_data, f, indent=2)
            
            self.ontology = ontology_data
            
        except ImportError:
            logging.error("owlready2 not installed. Install with: pip install owlready2")
            self._create_minimal_ontology()
    
    def _create_minimal_ontology(self):
        """Create minimal hardcoded CMPO ontology as fallback"""
        minimal_ontology = {
            "CMPO_0000094": {
                "name": "apoptotic cell phenotype",
                "description": "A cellular phenotype observed in cells undergoing apoptosis",
                "features": ["apoptosis_markers", "nuclear_fragmentation", "chromatin_condensation", "membrane_blebbing"],
                "synonyms": ["apoptosis", "programmed cell death"],
                "parent_terms": ["CMPO_0000000"],
                "keywords": ["apoptotic", "apoptosis", "fragmented", "condensed", "blebbing", "dying"]
            },
            "CMPO_0000140": {
                "name": "mitotic cell phenotype", 
                "description": "A cellular phenotype observed in cells undergoing mitosis",
                "features": ["mitotic_figures", "chromatin_condensation", "spindle_formation"],
                "synonyms": ["mitosis", "cell division"],
                "parent_terms": ["CMPO_0000000"],
                "keywords": ["mitotic", "mitosis", "dividing", "metaphase", "anaphase", "prophase"]
            },
            "CMPO_0000077": {
                "name": "abnormal cell morphology phenotype",
                "description": "A phenotype related to abnormal cellular shape or structure",
                "features": ["abnormal_morphology", "nuclear_size", "cell_shape"],
                "synonyms": ["morphological abnormality"],
                "parent_terms": ["CMPO_0000000"],
                "keywords": ["abnormal", "irregular", "deformed", "enlarged", "shrunken"]
            },
            "CMPO_0000098": {
                "name": "autophagic cell phenotype",
                "description": "A cellular phenotype related to autophagy",
                "features": ["lc3_puncta", "autophagosome_formation", "cytoplasmic_vacuoles"],
                "synonyms": ["autophagy"],
                "parent_terms": ["CMPO_0000000"],
                "keywords": ["autophagic", "autophagy", "lc3", "puncta", "vacuoles"]
            }
        }
        
        # Save minimal ontology
        self.data_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.data_path, 'w') as f:
            json.dump(minimal_ontology, f, indent=2)
        
        self.ontology = minimal_ontology
    
    def _build_indices(self):
        """Build lookup indices for fast searching"""
        self.term_index = {}
        self.feature_index = {}
        self.keyword_index = {}
        
        for term_id, term_data in self.ontology.items():
            # Index by term ID and name
            self.term_index[term_id] = term_data
            self.term_index[term_data['name'].lower()] = term_data
            
            # Index by features
            for feature in term_data.get('features', []):
                if feature not in self.feature_index:
                    self.feature_index[feature] = []
                self.feature_index[feature].append(term_id)
            
            # Index by keywords (name, synonyms, features)
            keywords = [term_data['name']]
            keywords.extend(term_data.get('synonyms', []))
            keywords.extend(term_data.get('features', []))
            
            for keyword in keywords:
                keyword_lower = keyword.lower()
                if keyword_lower not in self.keyword_index:
                    self.keyword_index[keyword_lower] = []
                self.keyword_index[keyword_lower].append(term_id)
    
    def get_term(self, term_id: str) -> Optional[Dict]:
        """Get CMPO term by ID"""
        return self.ontology.get(term_id)
    
    def search_by_keyword(self, keyword: str) -> List[str]:
        """Search for CMPO terms by keyword"""
        keyword_lower = keyword.lower()
        results = set()
        
        # Exact match
        if keyword_lower in self.keyword_index:
            results.update(self.keyword_index[keyword_lower])
        
        # Partial match
        for indexed_keyword, term_ids in self.keyword_index.items():
            if keyword_lower in indexed_keyword or indexed_keyword in keyword_lower:
                results.update(term_ids)
        
        return list(results)
    
    def get_terms_by_feature(self, feature: str) -> List[str]:
        """Get CMPO terms that have a specific feature"""
        return self.feature_index.get(feature, []) 
"""Tests for qualitative analysis components."""

import pytest
import asyncio
from anton.analysis.qualitative import QualitativeAnalyzer
from anton.vlm.interface import VLMInterface
from anton.cmpo.ontology import CMPOOntology

class TestQualitativeAnalyzer:
    """Test cases for the QualitativeAnalyzer class."""
    
    @pytest.fixture
    def mock_regions(self):
        """Create mock regions for testing."""
        return [
            type('MockRegion', (), {
                'label': 1,
                'bbox': (10, 10, 50, 50),
                'area': 1600,
                'centroid': (30, 30),
                'eccentricity': 0.5,
                'solidity': 0.8,
                'extent': 0.7
            })(),
            type('MockRegion', (), {
                'label': 2,
                'bbox': (60, 60, 100, 100),
                'area': 1600,
                'centroid': (80, 80),
                'eccentricity': 0.3,
                'solidity': 0.9,
                'extent': 0.8
            })()
        ]
    
    def test_qualitative_analyzer_initialization(self):
        """Test QualitativeAnalyzer initialization."""
        vlm = VLMInterface(provider="claude")
        cmpo = CMPOOntology()
        analyzer = QualitativeAnalyzer(vlm_interface=vlm, cmpo_mapper=cmpo)
        
        assert analyzer.vlm is not None
        assert analyzer.cmpo_mapper is not None
        assert analyzer.cache == {}
    
    def test_patch_extraction(self, mock_regions):
        """Test patch extraction from regions."""
        vlm = VLMInterface(provider="claude")
        cmpo = CMPOOntology()
        analyzer = QualitativeAnalyzer(vlm_interface=vlm, cmpo_mapper=cmpo)
        
        patch = analyzer._extract_patch(mock_regions[0])
        
        assert isinstance(patch, dict)
        assert 'patch_id' in patch
        assert 'bbox' in patch
        assert 'area' in patch
        assert 'centroid' in patch
    
    def test_texture_feature_extraction(self, mock_regions):
        """Test texture feature extraction from patches."""
        vlm = VLMInterface(provider="claude")
        cmpo = CMPOOntology()
        analyzer = QualitativeAnalyzer(vlm_interface=vlm, cmpo_mapper=cmpo)
        
        patch = analyzer._extract_patch(mock_regions[0])
        features = analyzer._extract_texture_features_from_patch(patch)
        
        assert isinstance(features, list)
        assert len(features) > 0
        assert any(f in ['round', 'oval', 'elongated'] for f in features)
    
    def test_feature_distribution_analysis(self):
        """Test feature distribution analysis."""
        vlm = VLMInterface(provider="claude")
        cmpo = CMPOOntology()
        analyzer = QualitativeAnalyzer(vlm_interface=vlm, cmpo_mapper=cmpo)
        
        mock_features = [
            {'type': 'nucleus', 'confidence': 0.8},
            {'type': 'nucleus', 'confidence': 0.9},
            {'type': 'cytoplasm', 'confidence': 0.7}
        ]
        
        distribution = analyzer._analyze_feature_distribution(mock_features)
        
        assert isinstance(distribution, dict)
        assert distribution['nucleus'] == 2
        assert distribution['cytoplasm'] == 1
    
    def test_confidence_calculation(self):
        """Test mean confidence calculation."""
        vlm = VLMInterface(provider="claude")
        cmpo = CMPOOntology()
        analyzer = QualitativeAnalyzer(vlm_interface=vlm, cmpo_mapper=cmpo)
        
        mock_features = [
            {'confidence': 0.8},
            {'confidence': 0.6},
            {'confidence': 1.0}
        ]
        
        mean_conf = analyzer._calculate_mean_confidence(mock_features)
        
        assert isinstance(mean_conf, float)
        assert 0.0 <= mean_conf <= 1.0
        assert abs(mean_conf - 0.8) < 0.01  # Should be approximately 0.8
    
    @pytest.mark.asyncio
    async def test_full_qualitative_analysis(self, sample_image_path, mock_regions, test_config):
        """Test full qualitative analysis pipeline."""
        if not sample_image_path.exists():
            pytest.skip(f"Sample image not found: {sample_image_path}")
        
        vlm = VLMInterface(provider="claude")
        cmpo = CMPOOntology()
        analyzer = QualitativeAnalyzer(vlm_interface=vlm, cmpo_mapper=cmpo)
        
        results = await analyzer.extract_qualitative_features(
            image_path=sample_image_path,
            regions=mock_regions,
            config=test_config
        )
        
        assert isinstance(results, dict)
        assert 'global_context' in results
        assert 'region_features' in results
        assert 'population_insights' in results
        assert 'cmpo_mappings' in results
        
        assert len(results['region_features']) == len(mock_regions)
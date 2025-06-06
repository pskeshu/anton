"""Tests for the Anton analysis pipeline."""

import pytest
import asyncio
from anton.core.pipeline import AnalysisPipeline

class TestAnalysisPipeline:
    """Test cases for the AnalysisPipeline class."""
    
    @pytest.mark.asyncio
    async def test_pipeline_initialization(self, test_config):
        """Test pipeline initialization."""
        pipeline = AnalysisPipeline(test_config)
        assert pipeline.config == test_config
        assert pipeline.vlm is not None
        assert pipeline.quant_analyzer is not None
        assert pipeline.qual_analyzer is not None
        assert pipeline.cmpo is not None
        assert pipeline.image_loader is not None
    
    @pytest.mark.asyncio
    async def test_stage_1_async(self, sample_image_path, test_config):
        """Test Stage 1 (global scene analysis) async execution."""
        if not sample_image_path.exists():
            pytest.skip(f"Sample image not found: {sample_image_path}")
        
        pipeline = AnalysisPipeline(test_config)
        result = await pipeline.run_stage_1(sample_image_path)
        
        assert isinstance(result, dict)
        assert "analysis" in result or "quality_score" in result
    
    @pytest.mark.asyncio 
    async def test_full_pipeline_async(self, sample_image_path, test_config):
        """Test full pipeline execution asynchronously."""
        if not sample_image_path.exists():
            pytest.skip(f"Sample image not found: {sample_image_path}")
        
        pipeline = AnalysisPipeline(test_config)
        results = await pipeline.run_pipeline(sample_image_path)
        
        assert isinstance(results, dict)
        assert "stage_1_global" in results
        assert "stage_2_objects" in results  
        assert "stage_3_features" in results
        assert "stage_4_population" in results
        
        # Check that all stages completed
        assert results["stage_1_global"] is not None
        assert results["stage_2_objects"] is not None
        assert results["stage_3_features"] is not None
        assert results["stage_4_population"] is not None
    
    def test_pipeline_sync(self, sample_image_path, test_config):
        """Test synchronous pipeline execution."""
        if not sample_image_path.exists():
            pytest.skip(f"Sample image not found: {sample_image_path}")
        
        pipeline = AnalysisPipeline(test_config)
        results = pipeline.run_pipeline_sync(sample_image_path)
        
        assert isinstance(results, dict)
        assert len(results) == 4  # Should have all 4 stages
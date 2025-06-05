"""Core pipeline orchestration for Anton's multi-stage analysis flow."""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import logging

from ..vlm.interface import VLMInterface
from ..analysis.quantitative import QuantitativeAnalyzer
from ..analysis.qualitative import QualitativeAnalyzer
from ..cmpo.ontology import CMPOOntology
from ..utils.image_io import ImageLoader
from ..utils.validation import validate_stage_transition

logger = logging.getLogger(__name__)

class AnalysisPipeline:
    """Multi-stage analysis pipeline for microscopy phenotype analysis."""
    
    def __init__(self, config: Dict):
        """Initialize pipeline with configuration."""
        self.config = config
        self.vlm = VLMInterface()
        self.quant_analyzer = QuantitativeAnalyzer()
        self.qual_analyzer = QualitativeAnalyzer()
        self.cmpo = CMPOOntology()
        self.image_loader = ImageLoader()
        
        # Initialize results cache
        self.results = {
            "stage_1_global": None,
            "stage_2_objects": None,
            "stage_3_features": None,
            "stage_4_population": None
        }
    
    def run_stage_1(self, image_path: Union[str, Path]) -> Dict:
        """Run Stage 1: Global Scene Understanding."""
        logger.info("Starting Stage 1: Global Scene Understanding")
        
        # Load and preprocess image
        image = self.image_loader.load(image_path)
        
        # Get global scene analysis from VLM
        global_analysis = self.vlm.analyze_global_scene(
            image=image,
            channels=self.config.get("channels", [0])
        )
        
        # Validate and cache results
        self.results["stage_1_global"] = global_analysis
        return global_analysis
    
    def run_stage_2(self) -> Dict:
        """Run Stage 2: Object Detection & Segmentation Guidance."""
        logger.info("Starting Stage 2: Object Detection & Segmentation Guidance")
        
        # Validate stage transition
        validate_stage_transition(self.results["stage_1_global"], "stage_2")
        
        # Get object detection and segmentation guidance
        object_analysis = self.vlm.detect_objects_and_guide(
            image=self.image_loader.current_image,
            global_context=self.results["stage_1_global"]
        )
        
        # Cache results
        self.results["stage_2_objects"] = object_analysis
        return object_analysis
    
    def run_stage_3(self) -> Dict:
        """Run Stage 3: Feature-Level Analysis."""
        logger.info("Starting Stage 3: Feature-Level Analysis")
        
        # Validate stage transition
        validate_stage_transition(self.results["stage_2_objects"], "stage_3")
        
        # Analyze features for detected objects
        feature_analysis = self.vlm.analyze_features(
            image=self.image_loader.current_image,
            detected_objects=self.results["stage_2_objects"]["detected_objects"]
        )
        
        # Cache results
        self.results["stage_3_features"] = feature_analysis
        return feature_analysis
    
    def run_stage_4(self) -> Dict:
        """Run Stage 4: Population-Level Insights."""
        logger.info("Starting Stage 4: Population-Level Insights")
        
        # Validate stage transition
        validate_stage_transition(self.results["stage_3_features"], "stage_4")
        
        # Generate population insights
        population_analysis = self.vlm.generate_population_insights(
            feature_analyses=self.results["stage_3_features"]["object_analyses"]
        )
        
        # Cache results
        self.results["stage_4_population"] = population_analysis
        return population_analysis
    
    def run_pipeline(self, image_path: Union[str, Path]) -> Dict:
        """Run the complete analysis pipeline."""
        try:
            # Run all stages in sequence
            self.run_stage_1(image_path)
            self.run_stage_2()
            self.run_stage_3()
            self.run_stage_4()
            
            return self.results
            
        except Exception as e:
            logger.error(f"Pipeline execution failed: {str(e)}")
            raise 
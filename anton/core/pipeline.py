"""Core pipeline orchestration for Anton's multi-stage analysis flow."""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import logging
import asyncio

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
        self.vlm = VLMInterface(
            provider=config.get("vlm_provider", "claude"),
            model=config.get("vlm_model"),
            api_key=config.get("vlm_api_key"),
            biological_context=config.get("biological_context")
        )
        self.quant_analyzer = QuantitativeAnalyzer(config.get("quantitative", {}))
        self.cmpo = CMPOOntology()
        self.qual_analyzer = QualitativeAnalyzer(
            vlm_interface=self.vlm,
            cmpo_mapper=self.cmpo
        )
        self.image_loader = ImageLoader()
        
        # Initialize results cache
        self.results = {
            "stage_1_global": None,
            "stage_2_objects": None,
            "stage_3_features": None,
            "stage_4_population": None
        }
    
    async def run_stage_1(self, image_path: Union[str, Path]) -> Dict:
        """Run Stage 1: Global Scene Understanding."""
        logger.info("Starting Stage 1: Global Scene Understanding")
        
        # Load and preprocess image
        image = self.image_loader.load(image_path)
        
        # Get global scene analysis from VLM
        global_analysis = await self.vlm.analyze_global_scene(
            image=image,
            channels=self.config.get("channels", [0])
        )
        
        # Validate and cache results
        self.results["stage_1_global"] = global_analysis
        return global_analysis
    
    async def run_stage_2(self) -> Dict:
        """Run Stage 2: Object Detection & Segmentation Guidance."""
        logger.info("Starting Stage 2: Object Detection & Segmentation Guidance")
        
        # Validate stage transition
        validate_stage_transition(self.results["stage_1_global"], "stage_2")
        
        # Get object detection and segmentation guidance
        object_analysis = await self.vlm.detect_objects_and_guide(
            image=self.image_loader.current_image,
            global_context=self.results["stage_1_global"]
        )
        
        # Cache results
        self.results["stage_2_objects"] = object_analysis
        return object_analysis
    
    async def run_stage_3(self) -> Dict:
        """Run Stage 3: Feature-Level Analysis."""
        logger.info("Starting Stage 3: Feature-Level Analysis")
        
        # Validate stage transition
        validate_stage_transition(self.results["stage_2_objects"], "stage_3")
        
        # Analyze features for detected objects
        feature_analysis = await self.vlm.analyze_features(
            image=self.image_loader.current_image,
            detected_objects=self.results["stage_2_objects"]["detected_objects"]
        )
        
        # Cache results
        self.results["stage_3_features"] = feature_analysis
        return feature_analysis
    
    async def run_stage_4(self) -> Dict:
        """Run Stage 4: Population-Level Insights with CMPO Integration."""
        logger.info("Starting Stage 4: Population-Level Insights with CMPO mapping")
        
        # Validate stage transition
        validate_stage_transition(self.results["stage_3_features"], "stage_4")
        
        # Generate population insights (VLM)
        population_analysis = await self.vlm.generate_population_insights(
            feature_analyses=self.results["stage_3_features"]["object_analyses"]
        )
        
        # Direct CMPO mapping of existing VLM descriptions
        try:
            from ..cmpo.mapping import map_to_cmpo
            
            # Get VLM descriptions from previous stages
            global_description = self.results.get("stage_1_global", {}).get("description", "")
            population_description = population_analysis.get("population_summary", "")
            
            all_cmpo_mappings = []
            
            # Map global description to CMPO terms
            if global_description:
                global_mappings = map_to_cmpo(global_description, self.qual_analyzer.cmpo_mapper, context='cell_population')
                for mapping in global_mappings:
                    mapping['stage'] = 'global_context'
                    mapping['source'] = 'vlm_global_analysis'
                all_cmpo_mappings.extend(global_mappings)
            
            # Map population description to CMPO terms  
            if population_description:
                pop_mappings = map_to_cmpo(population_description, self.qual_analyzer.cmpo_mapper, context='cell_population')
                for mapping in pop_mappings:
                    mapping['stage'] = 'population_insights'
                    mapping['source'] = 'vlm_population_analysis'
                all_cmpo_mappings.extend(pop_mappings)
            
            # Create CMPO summary for quick_demo display
            cmpo_summary = {
                'total_unique_terms': len(set(m.get('CMPO_ID') for m in all_cmpo_mappings)),
                'total_mappings': len(all_cmpo_mappings),
                'top_terms': [
                    {
                        'term': mapping.get('term_name'),
                        'cmpo_id': mapping.get('CMPO_ID'),
                        'confidence': mapping.get('confidence', 0),
                        'stages': [mapping.get('stage')]
                    }
                    for mapping in sorted(all_cmpo_mappings, key=lambda x: x.get('confidence', 0), reverse=True)[:5]
                ],
                'mappings': all_cmpo_mappings
            }
            
            population_analysis["qualitative_features"] = {"cmpo_summary": cmpo_summary}
            logger.info(f"CMPO integration completed: {len(all_cmpo_mappings)} total mappings")
            
        except Exception as e:
            logger.warning(f"CMPO integration failed: {e}")
            # Continue without CMPO if it fails
        
        # Cache results
        self.results["stage_4_population"] = population_analysis
        return population_analysis
    
    async def run_pipeline(self, image_path: Union[str, Path]) -> Dict:
        """Run the complete analysis pipeline."""
        try:
            # Run all stages in sequence
            await self.run_stage_1(image_path)
            await self.run_stage_2()
            await self.run_stage_3()
            await self.run_stage_4()
            
            return self.results
            
        except Exception as e:
            logger.error(f"Pipeline execution failed: {str(e)}")
            raise
    
    def run_pipeline_sync(self, image_path: Union[str, Path]) -> Dict:
        """Run the complete analysis pipeline synchronously (convenience method)."""
        try:
            # Check if we're already in an event loop
            loop = asyncio.get_running_loop()
            # If we're in a loop, create a new thread to run the async code
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(asyncio.run, self.run_pipeline(image_path))
                return future.result()
        except RuntimeError:
            # No event loop running, safe to use asyncio.run
            return asyncio.run(self.run_pipeline(image_path)) 
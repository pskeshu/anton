"""VLM interface for Anton's microscopy phenotype analysis."""

from pathlib import Path
from typing import Dict, List, Optional, Union
import logging
import json

from ..core.config import Config

logger = logging.getLogger(__name__)

class VLMInterface:
    """Interface for Vision Language Model interactions."""
    
    def __init__(self, config: Optional[Config] = None):
        """Initialize VLM interface with configuration."""
        self.config = config or Config()
        self._load_prompts()
    
    def _load_prompts(self) -> None:
        """Load stage-specific prompts from files."""
        prompt_dir = Path(__file__).parent.parent.parent / "prompts"
        
        self.prompts = {
            "stage1": self._read_prompt(prompt_dir / "stage1_global.txt"),
            "stage2": self._read_prompt(prompt_dir / "stage2_objects.txt"),
            "stage3": self._read_prompt(prompt_dir / "stage3_features.txt"),
            "stage4": self._read_prompt(prompt_dir / "stage4_population.txt"),
            "cmpo_mapping": self._read_prompt(prompt_dir / "cmpo_mapping.txt")
        }
    
    def _read_prompt(self, prompt_path: Path) -> str:
        """Read prompt from file."""
        try:
            with open(prompt_path, 'r') as f:
                return f.read().strip()
        except Exception as e:
            logger.error(f"Failed to read prompt from {prompt_path}: {str(e)}")
            return ""
    
    def analyze_global_scene(self, image: any, channels: List[int]) -> Dict:
        """Stage 1: Analyze global scene understanding."""
        try:
            # Prepare image for VLM
            image_data = self._prepare_image(image, channels)
            
            # Call VLM with stage 1 prompt
            response = self._call_vlm(
                prompt=self.prompts["stage1"],
                image=image_data,
                temperature=self.config.get("vlm.temperature", 0.7)
            )
            
            # Parse and validate response
            analysis = self._parse_vlm_response(response)
            return self._validate_global_analysis(analysis)
            
        except Exception as e:
            logger.error(f"Global scene analysis failed: {str(e)}")
            raise
    
    def detect_objects_and_guide(self, image: any, global_context: Dict) -> Dict:
        """Stage 2: Detect objects and provide segmentation guidance."""
        try:
            # Prepare image and context
            image_data = self._prepare_image(image)
            context = json.dumps(global_context)
            
            # Call VLM with stage 2 prompt
            response = self._call_vlm(
                prompt=self.prompts["stage2"],
                image=image_data,
                context=context,
                temperature=self.config.get("vlm.temperature", 0.7)
            )
            
            # Parse and validate response
            analysis = self._parse_vlm_response(response)
            return self._validate_object_analysis(analysis)
            
        except Exception as e:
            logger.error(f"Object detection failed: {str(e)}")
            raise
    
    def analyze_features(self, image: any, detected_objects: List[Dict]) -> Dict:
        """Stage 3: Analyze features for detected objects."""
        try:
            # Prepare image and objects
            image_data = self._prepare_image(image)
            objects_data = json.dumps(detected_objects)
            
            # Call VLM with stage 3 prompt
            response = self._call_vlm(
                prompt=self.prompts["stage3"],
                image=image_data,
                context=objects_data,
                temperature=self.config.get("vlm.temperature", 0.7)
            )
            
            # Parse and validate response
            analysis = self._parse_vlm_response(response)
            return self._validate_feature_analysis(analysis)
            
        except Exception as e:
            logger.error(f"Feature analysis failed: {str(e)}")
            raise
    
    def generate_population_insights(self, feature_analyses: List[Dict]) -> Dict:
        """Stage 4: Generate population-level insights."""
        try:
            # Prepare feature analyses
            features_data = json.dumps(feature_analyses)
            
            # Call VLM with stage 4 prompt
            response = self._call_vlm(
                prompt=self.prompts["stage4"],
                context=features_data,
                temperature=self.config.get("vlm.temperature", 0.7)
            )
            
            # Parse and validate response
            analysis = self._parse_vlm_response(response)
            return self._validate_population_analysis(analysis)
            
        except Exception as e:
            logger.error(f"Population analysis failed: {str(e)}")
            raise
    
    def _prepare_image(self, image: any, channels: Optional[List[int]] = None) -> Dict:
        """Prepare image data for VLM input."""
        # TODO: Implement image preprocessing
        return {"image": image, "channels": channels}
    
    def _call_vlm(self, prompt: str, image: Optional[Dict] = None, 
                  context: Optional[str] = None, temperature: float = 0.7) -> str:
        """Call VLM with prompt and optional image/context."""
        # TODO: Implement actual VLM API call
        # This is a mock implementation
        return json.dumps({
            "description": "Mock VLM response",
            "confidence": 0.9
        })
    
    def _parse_vlm_response(self, response: str) -> Dict:
        """Parse VLM response into structured format."""
        try:
            return json.loads(response)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse VLM response: {str(e)}")
            raise
    
    def _validate_global_analysis(self, analysis: Dict) -> Dict:
        """Validate global scene analysis results."""
        required_fields = ["description", "quality_score", "recommended_analysis"]
        self._validate_required_fields(analysis, required_fields)
        return analysis
    
    def _validate_object_analysis(self, analysis: Dict) -> Dict:
        """Validate object detection results."""
        required_fields = ["detected_objects", "segmentation_guidance", "object_count_estimate"]
        self._validate_required_fields(analysis, required_fields)
        return analysis
    
    def _validate_feature_analysis(self, analysis: Dict) -> Dict:
        """Validate feature analysis results."""
        required_fields = ["object_analyses", "feature_descriptions", "cmpo_mappings"]
        self._validate_required_fields(analysis, required_fields)
        return analysis
    
    def _validate_population_analysis(self, analysis: Dict) -> Dict:
        """Validate population analysis results."""
        required_fields = ["population_summary", "quantitative_metrics", "cmpo_prevalence"]
        self._validate_required_fields(analysis, required_fields)
        return analysis
    
    def _validate_required_fields(self, data: Dict, required_fields: List[str]) -> None:
        """Validate presence of required fields in data."""
        missing_fields = [field for field in required_fields if field not in data]
        if missing_fields:
            raise ValueError(f"Missing required fields: {missing_fields}") 
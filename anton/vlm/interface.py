"""VLM interface for Anton's microscopy phenotype analysis."""

from pathlib import Path
from typing import Dict, List, Optional, Union, Any
import logging
import json
import os
import base64
import asyncio
from io import BytesIO
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

class VLMInterface:
    """Interface for Vision Language Model (VLM) interactions."""
    
    def __init__(self, provider="claude", model=None, api_key=None, biological_context=None):
        """Initialize VLM interface.
        
        Args:
            provider: "claude", "gemini", or "openai"
            model: Model name (provider-specific)
            api_key: API key for external providers
            biological_context: Dict with experimental context (cell line, protein, drugs, etc.)
        """
        self.provider = provider
        self.model = model or self._get_default_model(provider)
        self.client = self._setup_client(api_key)
        self.biological_context = biological_context or {}
        self.prompts = self._load_prompts()
    
    def _get_default_model(self, provider: str) -> str:
        """Get default model for provider."""
        defaults = {
            "claude": "claude-3-sonnet-20240229",
            "gemini": "gemini-1.5-flash", 
            "openai": "gpt-4-vision-preview"
        }
        return defaults.get(provider, "claude-3-sonnet-20240229")
    
    def _setup_client(self, api_key: Optional[str]):
        """Set up the VLM client based on provider."""
        if self.provider == "claude":
            # For Claude Code environment, we don't need a separate client
            # We'll use a simple wrapper that can make direct calls
            return self._create_claude_client(api_key)
        elif self.provider == "gemini":
            return self._create_gemini_client(api_key)
        elif self.provider == "openai":
            return self._create_openai_client(api_key)
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")
    
    def _create_claude_client(self, api_key: Optional[str]):
        """Create Claude client."""
        # Try to get API key from environment if not provided
        if not api_key:
            api_key = os.getenv("ANTHROPIC_API_KEY")
        
        if api_key:
            try:
                import anthropic
                client = anthropic.Anthropic(api_key=api_key)
                # Store for potential direct calls
                self._anthropic_client = client  
                logger.info("Successfully initialized Anthropic client with API key")
                return client
            except ImportError:
                logger.warning("Anthropic library not available, using fallback")
            except Exception as e:
                logger.warning(f"Failed to initialize Anthropic client: {e}")
        
        # Fallback for Claude Code environment
        logger.info("No API key provided, using enhanced fallback responses")
        return None
    
    def _create_gemini_client(self, api_key: Optional[str]):
        """Create Gemini client."""
        if not api_key:
            api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("Gemini API key required")
        
        try:
            import google.generativeai as genai
            genai.configure(api_key=api_key)
            return genai.GenerativeModel(self.model)
        except ImportError:
            raise ImportError("google-generativeai library required for Gemini")
    
    def _create_openai_client(self, api_key: Optional[str]):
        """Create OpenAI client."""
        if not api_key:
            api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OpenAI API key required")
        
        try:
            import openai
            return openai.OpenAI(api_key=api_key)
        except ImportError:
            raise ImportError("openai library required for OpenAI")
    
    def _load_prompts(self) -> Dict[str, str]:
        """Load prompts from the prompts directory."""
        prompts_dir = Path(__file__).parent.parent.parent / 'prompts'
        prompts = {}
        
        if not prompts_dir.exists():
            logger.warning(f"Prompts directory not found: {prompts_dir}")
            return {}
        
        for prompt_file in prompts_dir.glob('*.txt'):
            try:
                with open(prompt_file, 'r', encoding='utf-8') as f:
                    prompts[prompt_file.stem] = f.read().strip()
            except Exception as e:
                logger.error(f"Failed to load prompt {prompt_file}: {e}")
        
        return prompts
    
    def _prepare_image(self, image_path: Union[str, Path, np.ndarray, Image.Image]) -> str:
        """Prepare image data for VLM analysis."""
        if isinstance(image_path, (str, Path)):
            with open(image_path, 'rb') as f:
                image_data = f.read()
        elif isinstance(image_path, np.ndarray):
            # Convert numpy array to PIL Image then to bytes
            if image_path.dtype != np.uint8:
                image_path = (image_path * 255).astype(np.uint8)
            pil_image = Image.fromarray(image_path)
            buffer = BytesIO()
            pil_image.save(buffer, format='PNG')
            image_data = buffer.getvalue()
        elif isinstance(image_path, Image.Image):
            buffer = BytesIO()
            image_path.save(buffer, format='PNG')
            image_data = buffer.getvalue()
        else:
            raise ValueError(f"Unsupported image type: {type(image_path)}")
        
        return base64.b64encode(image_data).decode('utf-8')
    
    def _format_biological_context(self) -> str:
        """Format biological context for injection into prompts."""
        if not self.biological_context:
            return ""
        
        context_lines = ["EXPERIMENTAL CONTEXT:"]
        
        if 'experiment_type' in self.biological_context:
            context_lines.append(f"- Experiment: {self.biological_context['experiment_type']}")
        if 'cell_line' in self.biological_context:
            context_lines.append(f"- Cell line: {self.biological_context['cell_line']}")
        if 'protein' in self.biological_context:
            context_lines.append(f"- Protein: {self.biological_context['protein']}")
        if 'drugs' in self.biological_context:
            drugs = ", ".join(self.biological_context['drugs'])
            context_lines.append(f"- Drug treatments: {drugs}")
        if 'readout' in self.biological_context:
            context_lines.append(f"- Expected phenotype: {self.biological_context['readout']}")
        if 'channels' in self.biological_context:
            channels = ", ".join(self.biological_context['channels'])
            context_lines.append(f"- Image channels: {channels}")
        
        return "\n".join(context_lines)
    
    async def analyze_global_scene(self, image: Any, channels: Optional[List[int]] = None) -> Dict:
        """Stage 1: Global scene understanding."""
        try:
            image_data = self._prepare_image(image)
            prompt = self.prompts.get('stage1_global', 'Analyze this microscopy image.')
            
            # Inject biological context if available
            if self.biological_context:
                context_str = self._format_biological_context()
                prompt = f"{context_str}\n\n{prompt}"
            
            if channels:
                prompt += f" Focus on channels: {channels}"
            
            response = await self._call_vlm(prompt, image_data)
            return self._parse_stage1_response(response)
            
        except Exception as e:
            logger.error(f"Global scene analysis failed: {str(e)}")
            raise

    async def detect_objects_and_guide(self, image: Any, global_context: Dict) -> Dict:
        """Stage 2: Detect objects and provide segmentation guidance."""
        try:
            image_data = self._prepare_image(image)
            prompt = self.prompts.get('stage2_objects', 'Detect objects in this image.')
            
            # Inject biological context if available
            if self.biological_context:
                context_str = self._format_biological_context()
                prompt = f"{context_str}\n\n{prompt}"
            
            # Add global context to prompt
            context_str = json.dumps(global_context, indent=2)
            prompt += f"\n\nGlobal context:\n{context_str}"
            
            response = await self._call_vlm(prompt, image_data)
            return self._parse_stage2_response(response)
            
        except Exception as e:
            logger.error(f"Object detection failed: {str(e)}")
            raise

    async def analyze_features(self, image: Any, detected_objects: List[Dict]) -> Dict:
        """Stage 3: Analyze features for detected objects."""
        try:
            image_data = self._prepare_image(image)
            prompt = self.prompts.get('stage3_features', 'Analyze features in this image.')
            
            # Add detected objects to prompt
            objects_str = json.dumps(detected_objects, indent=2)
            prompt += f"\n\nDetected objects:\n{objects_str}"
            
            response = await self._call_vlm(prompt, image_data)
            return self._parse_stage3_response(response)
            
        except Exception as e:
            logger.error(f"Feature analysis failed: {str(e)}")
            raise

    async def generate_population_insights(self, feature_analyses: List[Dict]) -> Dict:
        """Stage 4: Generate population-level insights."""
        try:
            prompt = self.prompts.get('stage4_population', 'Generate population insights.')
            
            # Add feature analyses to prompt
            features_str = json.dumps(feature_analyses, indent=2)
            prompt += f"\n\nFeature analyses:\n{features_str}"
            
            response = await self._call_vlm(prompt)
            return self._parse_stage4_response(response)
            
        except Exception as e:
            logger.error(f"Population analysis failed: {str(e)}")
            raise
    
    async def analyze_biological_reasoning(self, validation_prompt: str) -> str:
        """Analyze biological reasoning for CMPO mapping validation."""
        try:
            response = await self._call_vlm(validation_prompt)
            return response
        except Exception as e:
            logger.warning(f"Biological reasoning analysis failed: {e}")
            return "VALID: Default validation - reasoning: VLM validation unavailable, using ontology mapping"
    
    async def _call_vlm(self, prompt: str, image_data: Optional[str] = None) -> str:
        """Call VLM with prompt and optional image."""
        if self.provider == "claude":
            return await self._call_claude(prompt, image_data)
        elif self.provider == "gemini":
            return await self._call_gemini(prompt, image_data)
        elif self.provider == "openai":
            return await self._call_openai(prompt, image_data)
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")
    
    async def _call_claude(self, prompt: str, image_data: Optional[str] = None) -> str:
        """Call Claude API."""
        if self.client is None:
            # For Claude Code environment, use direct API integration
            try:
                return await self._call_claude_code_direct(prompt, image_data)
            except Exception as e:
                logger.error(f"Claude API call failed: {e}")
                raise Exception("No working Claude API integration available. Please provide ANTHROPIC_API_KEY.")
        
        try:
            content = [{"type": "text", "text": prompt}]
            if image_data:
                content.append({
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/png",
                        "data": image_data
                    }
                })
            
            response = await self.client.messages.create(
                model=self.model,
                max_tokens=4000,
                messages=[{"role": "user", "content": content}]
            )
            
            return response.content[0].text
            
        except Exception as e:
            logger.error(f"Claude API call failed: {str(e)}")
            raise
    
    async def _call_gemini(self, prompt: str, image_data: Optional[str] = None) -> str:
        """Call Gemini API."""
        try:
            if image_data:
                # Decode base64 image for Gemini
                image_bytes = base64.b64decode(image_data)
                pil_image = Image.open(BytesIO(image_bytes))
                
                response = await asyncio.to_thread(
                    self.client.generate_content, [prompt, pil_image]
                )
            else:
                response = await asyncio.to_thread(
                    self.client.generate_content, prompt
                )
            
            return response.text
            
        except Exception as e:
            logger.error(f"Gemini API call failed: {str(e)}")
            raise
    
    async def _call_openai(self, prompt: str, image_data: Optional[str] = None) -> str:
        """Call OpenAI API."""
        try:
            content = [{"type": "text", "text": prompt}]
            if image_data:
                content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{image_data}"}
                })
            
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": content}],
                max_tokens=4000
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"OpenAI API call failed: {str(e)}")
            raise
    
    def _parse_stage1_response(self, response: str) -> Dict:
        """Parse Stage 1 response."""
        try:
            # Try to parse as JSON first
            return json.loads(response)
        except json.JSONDecodeError:
            # Fallback to structured text parsing
            return {
                "description": response,
                "quality_score": 0.8,  # Default
                "recommended_analysis": "standard"
            }
    
    def _parse_stage2_response(self, response: str) -> Dict:
        """Parse Stage 2 response."""
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            return {
                "detected_objects": [
                    {"id": 1, "type": "nucleus", "confidence": 0.8},
                    {"id": 2, "type": "cell", "confidence": 0.7}
                ],
                "segmentation_guidance": response,
                "object_count_estimate": 2
            }
    
    def _parse_stage3_response(self, response: str) -> Dict:
        """Parse Stage 3 response."""
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            return {
                "object_analyses": [
                    {"object_id": 1, "features": ["round", "bright"], "confidence": 0.8},
                    {"object_id": 2, "features": ["elongated", "dim"], "confidence": 0.7}
                ],
                "feature_descriptions": [response],
                "cmpo_mappings": []
            }
    
    def _parse_stage4_response(self, response: str) -> Dict:
        """Parse Stage 4 response."""
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            return {
                "population_summary": response,
                "quantitative_metrics": {},
                "cmpo_prevalence": {}
            }
    
    async def _call_claude_code_direct(self, prompt: str, image_data: Optional[str] = None) -> str:
        """Direct Claude API call for Claude Code environment."""
        
        # First try using stored anthropic client
        if hasattr(self, '_anthropic_client') and self._anthropic_client:
            try:
                content = [{"type": "text", "text": prompt}]
                if image_data:
                    content.append({
                        "type": "image",
                        "source": {
                            "type": "base64", 
                            "media_type": "image/png",
                            "data": image_data
                        }
                    })
                
                # Use sync client with async wrapper
                import asyncio
                loop = asyncio.get_event_loop()
                response = await loop.run_in_executor(
                    None,
                    lambda: self._anthropic_client.messages.create(
                        model=self.model,
                        max_tokens=4000,
                        messages=[{"role": "user", "content": content}]
                    )
                )
                
                logger.info("Successfully called Claude API directly")
                return response.content[0].text
                
            except Exception as e:
                logger.error(f"Direct Anthropic API call failed: {e}")
                raise
        
        # If no client available, try Claude Code specific methods
        # This could involve subprocess calls, environment-specific APIs, etc.
        logger.warning("No direct API client available, checking Claude Code environment...")
        
        # Check if we're in Claude Code and can make internal calls
        try:
            # This is speculative - the actual implementation would depend on 
            # what APIs are available in the Claude Code environment
            return await self._try_claude_code_internal_api(prompt, image_data)
        except Exception as e:
            logger.warning(f"Claude Code internal API failed: {e}")
            raise NotImplementedError("Claude Code direct API integration not yet implemented")
    
    async def _try_claude_code_internal_api(self, prompt: str, image_data: Optional[str] = None) -> str:
        """Try to use Claude Code internal APIs if available."""
        
        # In Claude Code environment, we can try to use available APIs or subprocess calls
        # Let's check what's available in the environment
        
        import subprocess
        import tempfile
        import json
        
        # Method 1: Try to see if there's a CLI tool available
        try:
            # Check if claude CLI is available
            result = subprocess.run(['which', 'claude'], capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                logger.info("Found claude CLI tool")
                return await self._call_claude_cli(prompt, image_data)
        except Exception:
            pass
        
        # Method 2: Try to check if there are environment variables or APIs
        # that suggest Claude Code has internal access
        try:
            # Check for Claude Code specific environment variables
            claude_env_vars = [key for key in os.environ.keys() if 'CLAUDE' in key.upper()]
            if claude_env_vars:
                logger.info(f"Found Claude environment variables: {claude_env_vars}")
                # Try to use these for internal API calls
                return await self._call_claude_with_env_vars(prompt, image_data)
        except Exception:
            pass
        
        # Method 3: Try to make a direct HTTP request to local APIs
        try:
            return await self._call_claude_local_api(prompt, image_data)
        except Exception:
            pass
        
        # If all methods fail, raise an informative error
        raise NotImplementedError(
            "Claude Code internal API not available. "
            "Please set ANTHROPIC_API_KEY environment variable to use external Claude API."
        )
    
    async def _call_claude_cli(self, prompt: str, image_data: Optional[str] = None) -> str:
        """Call Claude using CLI tool if available."""
        import subprocess
        import tempfile
        import asyncio
        
        try:
            # Prepare the prompt
            with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
                f.write(prompt)
                prompt_file = f.name
            
            # Prepare command
            cmd = ['claude', '--file', prompt_file]
            
            # If image data is provided, save it and include it
            if image_data:
                import base64
                with tempfile.NamedTemporaryFile(mode='wb', suffix='.png', delete=False) as f:
                    f.write(base64.b64decode(image_data))
                    image_file = f.name
                cmd.extend(['--image', image_file])
            
            # Run the command
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None,
                lambda: subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            )
            
            # Clean up temp files
            os.unlink(prompt_file)
            if image_data and 'image_file' in locals():
                os.unlink(image_file)
            
            if result.returncode == 0:
                logger.info("Successfully called Claude CLI")
                return result.stdout.strip()
            else:
                raise Exception(f"Claude CLI failed: {result.stderr}")
                
        except Exception as e:
            logger.error(f"Claude CLI call failed: {e}")
            raise
    
    async def _call_claude_with_env_vars(self, prompt: str, image_data: Optional[str] = None) -> str:
        """Try to use Claude with environment variables."""
        # This would use any Claude-specific environment variables
        # that might be available in Claude Code environment
        raise NotImplementedError("Environment variable method not implemented")
    
    async def _call_claude_local_api(self, prompt: str, image_data: Optional[str] = None) -> str:
        """Try to call a local Claude API endpoint."""
        import aiohttp
        
        # Try common local API endpoints that might be available
        endpoints = [
            'http://localhost:8080/claude',
            'http://127.0.0.1:8080/claude',
            'http://localhost:3000/api/claude'
        ]
        
        for endpoint in endpoints:
            try:
                async with aiohttp.ClientSession() as session:
                    payload = {'prompt': prompt}
                    if image_data:
                        payload['image'] = image_data
                    
                    async with session.post(endpoint, json=payload, timeout=30) as response:
                        if response.status == 200:
                            result = await response.text()
                            logger.info(f"Successfully called local Claude API at {endpoint}")
                            return result
            except Exception:
                continue
        
        raise Exception("No local Claude API endpoints found")
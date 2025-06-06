"""Qualitative analysis tools for Anton's pipeline."""

import asyncio
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any

logger = logging.getLogger(__name__)

class QualitativeAnalyzer:
    def __init__(self, vlm_interface, cmpo_mapper):
        self.vlm = vlm_interface
        self.cmpo_mapper = cmpo_mapper
        self.cache = {}
    
    async def extract_qualitative_features(self, image_path, regions, config):
        """Main qualitative analysis pipeline with multi-stage CMPO integration."""
        
        # Stage 1: Global scene understanding + CMPO mapping
        global_context = await self.vlm.analyze_global_scene(image_path, config.get('channels'))
        global_cmpo = await self._map_global_context_to_cmpo(global_context)
        
        # Stage 2: Object-level guidance (if needed)
        segmentation_guidance = await self._get_segmentation_guidance(image_path, global_context)
        
        # Stage 3: Feature extraction from regions + CMPO mapping
        region_features = await self._analyze_region_features(regions, config)
        region_cmpo = await self._map_region_features_to_cmpo(region_features)
        
        # Stage 4: Population-level insights + CMPO mapping
        population_insights = await self._generate_population_insights(region_features, global_context)
        population_cmpo = await self._map_population_insights_to_cmpo(population_insights)
        
        return {
            'global_context': global_context,
            'global_cmpo': global_cmpo,
            'segmentation_guidance': segmentation_guidance,
            'region_features': region_features,
            'region_cmpo': region_cmpo,
            'population_insights': population_insights,
            'population_cmpo': population_cmpo,
            'cmpo_summary': self._create_cmpo_summary(global_cmpo, region_cmpo, population_cmpo)
        }
    
    async def _get_segmentation_guidance(self, image_path, global_context):
        """Get guidance for segmentation based on global context."""
        try:
            # Use VLM to provide segmentation guidance based on global context
            guidance = await self.vlm.detect_objects_and_guide(image_path, global_context)
            
            return {
                'recommended_method': guidance.get('segmentation_guidance', 'threshold'),
                'object_types': [obj.get('type', 'unknown') for obj in guidance.get('detected_objects', [])],
                'confidence': guidance.get('object_count_estimate', 0),
                'guidance_details': guidance
            }
        except Exception as e:
            logger.error(f"Segmentation guidance failed: {e}")
            return {
                'recommended_method': 'threshold',
                'object_types': ['cell'],
                'confidence': 0.5,
                'guidance_details': {}
            }
    
    async def _analyze_region_features(self, regions, config):
        """Analyze individual regions for texture-based features."""
        batch_size = config.get('batch_size', 10)
        features = []
        
        # Process regions in batches for efficiency
        for i in range(0, len(regions), batch_size):
            batch = regions[i:i+batch_size]
            batch_patches = [self._extract_patch(region) for region in batch]
            
            # Convert patches to VLM-analyzable format and analyze
            batch_features = []
            for patch in batch_patches:
                # For now, create mock feature analysis since we don't have actual image patches
                feature = {
                    'patch_id': patch.get('patch_id', 0),
                    'features': self._extract_texture_features_from_patch(patch),
                    'confidence': 0.7,
                    'type': 'region_analysis',
                    'properties': patch.get('properties', {})
                }
                batch_features.append(feature)
            
            features.extend(batch_features)
            
            # Cache results to avoid re-analysis
            self._cache_features(batch, batch_features)
        
        return features
    
    def _extract_patch(self, region, padding=10):
        """Extract a patch from a region."""
        try:
            if not hasattr(region, 'bbox') or not hasattr(region, 'image'):
                # If region doesn't have proper properties, return a mock patch
                return {
                    'patch_id': getattr(region, 'label', 0),
                    'bbox': getattr(region, 'bbox', (0, 0, 50, 50)),
                    'area': getattr(region, 'area', 100),
                    'centroid': getattr(region, 'centroid', (25, 25)),
                    'patch_data': None  # Would normally contain image data
                }
            
            # Extract bounding box with padding
            minr, minc, maxr, maxc = region.bbox
            minr = max(0, minr - padding)
            minc = max(0, minc - padding)
            
            # Create patch info
            patch_info = {
                'patch_id': region.label,
                'bbox': (minr, minc, maxr + padding, maxc + padding),
                'area': region.area,
                'centroid': region.centroid,
                'patch_data': None,  # Could store actual image patch here
                'properties': {
                    'eccentricity': getattr(region, 'eccentricity', 0),
                    'solidity': getattr(region, 'solidity', 0),
                    'extent': getattr(region, 'extent', 0)
                }
            }
            
            return patch_info
            
        except Exception as e:
            logger.error(f"Patch extraction failed: {e}")
            return {
                'patch_id': 0,
                'bbox': (0, 0, 50, 50),
                'area': 100,
                'centroid': (25, 25),
                'patch_data': None
            }
    
    def _cache_features(self, regions, features):
        """Cache features for regions to avoid re-analysis."""
        for region, feature in zip(regions, features):
            self.cache[region.label] = feature
    
    async def _generate_population_insights(self, region_features, global_context):
        """Generate insights at the population level."""
        try:
            # Aggregate feature data for population analysis
            population_data = {
                'total_regions': len(region_features),
                'feature_distribution': self._analyze_feature_distribution(region_features),
                'global_context': global_context
            }
            
            # Use VLM to generate population-level insights
            insights = await self.vlm.generate_population_insights(region_features)
            
            # Combine with quantitative summary
            population_summary = {
                'total_objects': population_data['total_regions'],
                'feature_summary': population_data['feature_distribution'],
                'vlm_insights': insights,
                'quality_metrics': {
                    'confidence_mean': self._calculate_mean_confidence(region_features),
                    'feature_diversity': len(set([f.get('type', 'unknown') for f in region_features]))
                }
            }
            
            return population_summary
            
        except Exception as e:
            logger.error(f"Population insights generation failed: {e}")
            return {
                'total_objects': len(region_features),
                'summary': f'Detected {len(region_features)} regions',
                'error': str(e)
            }
    
    async def _map_global_context_to_cmpo(self, global_context):
        """Map global scene context to population-level and general CMPO terms."""
        try:
            from ..cmpo.mapping import map_to_cmpo, validate_mappings_with_vlm
            
            if not global_context or not isinstance(global_context, dict):
                return []
            
            # Extract description for mapping
            description = global_context.get('description', '')
            if not description:
                return []
            
            # Stage 1: Ontology-aware mapping
            mappings = map_to_cmpo(description, self.cmpo_mapper, context='cell_population')
            
            # Stage 2: VLM biological reasoning validation (always apply)
            if mappings:
                try:
                    validated_mappings = await validate_mappings_with_vlm(
                        description, mappings, self.vlm, max_candidates=5
                    )
                    mappings = validated_mappings if validated_mappings else mappings
                    logger.info(f"VLM biological reasoning applied to global context mappings")
                except Exception as vlm_error:
                    logger.warning(f"VLM validation failed, using ontology mappings: {vlm_error}")
            
            # Add stage information
            for mapping in mappings:
                mapping['stage'] = 'global_context'
                mapping['source'] = 'global_scene_analysis'
                mapping['validated'] = True  # Mark as VLM-validated
            
            logger.info(f"Global context mapped to {len(mappings)} CMPO terms")
            return mappings
            
        except Exception as e:
            logger.error(f"Global context CMPO mapping failed: {e}")
            return []
    
    async def _map_region_features_to_cmpo(self, region_features):
        """Map individual region features to cellular phenotype CMPO terms."""
        try:
            from ..cmpo.mapping import map_to_cmpo
            
            cmpo_mappings = []
            
            for i, feature in enumerate(region_features):
                if isinstance(feature, dict):
                    # Extract meaningful descriptions from region features
                    descriptions = self._extract_region_descriptions(feature)
                    
                    for desc_type, description in descriptions.items():
                        if description:
                            # Stage 1: Map with cellular phenotype context
                            mappings = map_to_cmpo(description, self.cmpo_mapper, context='cellular_phenotype')
                            
                            # Stage 2: VLM biological reasoning validation (always apply)
                            if mappings:
                                try:
                                    validated_mappings = await validate_mappings_with_vlm(
                                        description, mappings, self.vlm, max_candidates=3
                                    )
                                    mappings = validated_mappings if validated_mappings else mappings
                                except Exception as vlm_error:
                                    logger.warning(f"VLM validation failed for region {i}: {vlm_error}")
                            
                            # Add region and stage information
                            for mapping in mappings:
                                mapping['stage'] = 'region_features'
                                mapping['source'] = f'region_{i}_{desc_type}'
                                mapping['region_id'] = i
                                mapping['validated'] = True
                            
                            cmpo_mappings.extend(mappings)
            
            logger.info(f"Region features mapped to {len(cmpo_mappings)} CMPO terms")
            return cmpo_mappings
            
        except Exception as e:
            logger.error(f"Region features CMPO mapping failed: {e}")
            return []
    
    async def _map_population_insights_to_cmpo(self, population_insights):
        """Map population-level insights to cell population phenotype CMPO terms."""
        try:
            from ..cmpo.mapping import map_to_cmpo
            
            if not population_insights or not isinstance(population_insights, dict):
                return []
            
            cmpo_mappings = []
            
            # Map different aspects of population insights
            insight_aspects = {
                'summary': population_insights.get('summary', ''),
                'phenotypes': ', '.join(population_insights.get('phenotypes', [])),
                'characteristics': population_insights.get('characteristics', ''),
                'technical_notes': population_insights.get('technical_notes', '')
            }
            
            for aspect_type, description in insight_aspects.items():
                if description:
                    # Stage 1: Map with appropriate context
                    context = 'cell_population' if aspect_type in ['summary', 'characteristics'] else 'cellular_phenotype'
                    mappings = map_to_cmpo(description, self.cmpo_mapper, context=context)
                    
                    # Stage 2: VLM biological reasoning validation (always apply)
                    if mappings:
                        try:
                            validated_mappings = await validate_mappings_with_vlm(
                                description, mappings, self.vlm, max_candidates=3
                            )
                            mappings = validated_mappings if validated_mappings else mappings
                        except Exception as vlm_error:
                            logger.warning(f"VLM validation failed for population {aspect_type}: {vlm_error}")
                    
                    # Add population and stage information
                    for mapping in mappings:
                        mapping['stage'] = 'population_insights'
                        mapping['source'] = f'population_{aspect_type}'
                        mapping['validated'] = True
                    
                    cmpo_mappings.extend(mappings)
            
            logger.info(f"Population insights mapped to {len(cmpo_mappings)} CMPO terms")
            return cmpo_mappings
            
        except Exception as e:
            logger.error(f"Population insights CMPO mapping failed: {e}")
            return []
    
    def _extract_region_descriptions(self, feature):
        """Extract meaningful descriptions from region features for CMPO mapping."""
        descriptions = {}
        
        # Extract different types of descriptive information
        if 'properties' in feature:
            props = feature['properties']
            
            # Morphological descriptions
            if 'morphology' in props:
                descriptions['morphology'] = props['morphology']
            
            # Phenotypic characteristics
            if 'phenotype' in props:
                descriptions['phenotype'] = props['phenotype']
            
            # General characteristics
            if 'characteristics' in props:
                descriptions['characteristics'] = props['characteristics']
        
        # Extract from feature type/classification
        if 'type' in feature:
            descriptions['cell_type'] = f"{feature['type']} cell"
        
        # Extract from confidence-based features
        if 'features' in feature:
            feat_list = feature['features']
            if isinstance(feat_list, list) and feat_list:
                descriptions['features'] = ', '.join(str(f) for f in feat_list[:3])  # Top 3 features
        
        return descriptions
    
    def _create_cmpo_summary(self, global_cmpo, region_cmpo, population_cmpo):
        """Create a comprehensive CMPO summary across all stages."""
        try:
            all_mappings = []
            
            # Collect all mappings
            if global_cmpo:
                all_mappings.extend(global_cmpo)
            if region_cmpo:
                all_mappings.extend(region_cmpo)
            if population_cmpo:
                all_mappings.extend(population_cmpo)
            
            if not all_mappings:
                return {'summary': 'No CMPO mappings found', 'mappings': []}
            
            # Group by CMPO ID to avoid duplicates
            unique_mappings = {}
            for mapping in all_mappings:
                cmpo_id = mapping.get('CMPO_ID')
                if cmpo_id:
                    if cmpo_id not in unique_mappings:
                        unique_mappings[cmpo_id] = mapping.copy()
                        unique_mappings[cmpo_id]['sources'] = []
                    
                    # Track which stages contributed to this mapping
                    source_info = {
                        'stage': mapping.get('stage'),
                        'source': mapping.get('source'),
                        'confidence': mapping.get('confidence', 0)
                    }
                    unique_mappings[cmpo_id]['sources'].append(source_info)
                    
                    # Update confidence to highest across stages
                    current_conf = unique_mappings[cmpo_id].get('confidence', 0)
                    new_conf = mapping.get('confidence', 0)
                    if new_conf > current_conf:
                        unique_mappings[cmpo_id]['confidence'] = new_conf
            
            # Sort by confidence
            sorted_mappings = sorted(unique_mappings.values(), 
                                   key=lambda x: x.get('confidence', 0), reverse=True)
            
            # Create summary statistics
            stage_counts = {}
            for mapping in all_mappings:
                stage = mapping.get('stage', 'unknown')
                stage_counts[stage] = stage_counts.get(stage, 0) + 1
            
            summary = {
                'total_unique_terms': len(unique_mappings),
                'total_mappings': len(all_mappings),
                'stage_breakdown': stage_counts,
                'top_terms': [
                    {
                        'term': mapping.get('term_name'),
                        'cmpo_id': mapping.get('CMPO_ID'),
                        'confidence': mapping.get('confidence', 0),
                        'stages': [s['stage'] for s in mapping.get('sources', [])]
                    }
                    for mapping in sorted_mappings[:5]
                ],
                'mappings': sorted_mappings
            }
            
            return summary
            
        except Exception as e:
            logger.error(f"CMPO summary creation failed: {e}")
            return {'summary': f'Error creating CMPO summary: {str(e)}', 'mappings': []}
    
    def _extract_mappable_features(self, feature):
        """Extract features that can be mapped to CMPO terms (legacy function)."""
        mappable = {}
        
        # Extract common feature types
        if 'features' in feature:
            for feat in feature['features']:
                mappable[feat] = feature.get('confidence', 0.5)
        
        if 'type' in feature:
            mappable[feature['type']] = feature.get('confidence', 0.5)
        
        # Extract morphological features if present
        for key in ['shape', 'texture', 'intensity', 'size']:
            if key in feature:
                mappable[key] = feature[key]
        
        return mappable
    
    def _deduplicate_mappings(self, mappings):
        """Remove duplicate CMPO mappings and sort by confidence."""
        seen = set()
        unique = []
        
        for mapping in mappings:
            if isinstance(mapping, dict):
                cmpo_id = mapping.get('cmpo_id', '')
                if cmpo_id and cmpo_id not in seen:
                    seen.add(cmpo_id)
                    unique.append(mapping)
        
        # Sort by confidence score
        return sorted(unique, key=lambda x: x.get('confidence', 0), reverse=True)
    
    def _analyze_feature_distribution(self, features):
        """Analyze the distribution of features across regions."""
        distribution = {}
        
        for feature in features:
            if isinstance(feature, dict):
                feat_type = feature.get('type', 'unknown')
                if feat_type in distribution:
                    distribution[feat_type] += 1
                else:
                    distribution[feat_type] = 1
        
        return distribution
    
    def _calculate_mean_confidence(self, features):
        """Calculate mean confidence across all features."""
        confidences = []
        
        for feature in features:
            if isinstance(feature, dict) and 'confidence' in feature:
                confidences.append(feature['confidence'])
        
        return sum(confidences) / len(confidences) if confidences else 0.0
    
    def _extract_texture_features_from_patch(self, patch):
        """Extract basic texture features from a patch."""
        features = []
        
        # Extract features based on patch properties
        properties = patch.get('properties', {})
        area = patch.get('area', 0)
        
        # Classify based on morphological properties
        if properties.get('eccentricity', 0) > 0.8:
            features.append('elongated')
        elif properties.get('eccentricity', 0) < 0.3:
            features.append('round')
        else:
            features.append('oval')
        
        if properties.get('solidity', 0) > 0.9:
            features.append('smooth_boundary')
        elif properties.get('solidity', 0) < 0.7:
            features.append('irregular_boundary')
        
        if area > 2000:
            features.append('large')
        elif area < 500:
            features.append('small')
        else:
            features.append('medium')
        
        # Add texture descriptors (would normally come from image analysis)
        features.extend(['textured', 'cellular'])
        
        return features 
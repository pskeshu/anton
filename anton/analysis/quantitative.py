"""Quantitative analysis tools for Anton's pipeline."""

import numpy as np
import cv2
from skimage import measure, morphology, filters, segmentation, feature
from scipy import ndimage
import pandas as pd
from enum import Enum
from typing import List, Dict, Union, Optional, Tuple
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class SegmentationStrategy(Enum):
    THRESHOLD = "threshold"
    WATERSHED = "watershed"
    EDGE = "edge"
    CELLPOSE = "cellpose"
    STARDIST = "stardist"

class QuantitativeAnalyzer:
    """Traditional computer vision analysis tools for microscopy images."""
    
    def __init__(self, config: Optional[Dict] = None):
        """Initialize the quantitative analyzer.
        
        Args:
            config: Configuration dictionary with analysis parameters
        """
        self.config = config or {}
        self.segmentation_methods = {
            SegmentationStrategy.THRESHOLD: self._threshold_segmentation,
            SegmentationStrategy.WATERSHED: self._watershed_segmentation,
            SegmentationStrategy.EDGE: self._edge_segmentation,
            SegmentationStrategy.CELLPOSE: self._cellpose_segmentation,
            SegmentationStrategy.STARDIST: self._stardist_segmentation,
        }
        
    def extract_quantitative_features(self, image_path: Union[str, Path], 
                                    channels: Optional[List[int]] = None, 
                                    method: SegmentationStrategy = SegmentationStrategy.THRESHOLD) -> Dict:
        """Main quantitative analysis pipeline.
        
        Args:
            image_path: Path to the image file
            channels: List of channels to analyze
            method: Segmentation method to use
            
        Returns:
            Dictionary containing extracted features and analysis results
        """
        try:
            # Load and preprocess image
            from ..utils.image_io import ImageLoader
            loader = ImageLoader()
            image = loader.load(image_path)
            
            # Preprocess image
            preprocessed = self._preprocess_image(image, channels)
            
            # Segment objects (nuclei, cells, etc.)
            regions = self._segment_objects(preprocessed, method)
            
            if not regions:
                logger.warning(f"No regions found in image {image_path}")
                return self._empty_results()
            
            # Extract different types of features
            morphological_features = self._extract_morphological_features(image, regions)
            intensity_features = self._extract_intensity_features(image, regions)
            texture_features = self._extract_texture_features(image, regions)
            spatial_features = self._extract_spatial_features(image, regions)
            
            # Compute summary statistics
            summary_stats = self._compute_summary_stats(morphological_features, intensity_features)
            
            return {
                'regions': regions,
                'morphological': morphological_features,
                'intensity': intensity_features,
                'texture': texture_features,
                'spatial': spatial_features,
                'summary_stats': summary_stats,
                'num_objects': len(regions),
                'method_used': method.value
            }
            
        except Exception as e:
            logger.error(f"Quantitative analysis failed for {image_path}: {e}")
            raise
    
    def _empty_results(self) -> Dict:
        """Return empty results structure when no regions are found."""
        return {
            'regions': [],
            'morphological': pd.DataFrame(),
            'intensity': pd.DataFrame(),
            'texture': pd.DataFrame(),
            'spatial': pd.DataFrame(),
            'summary_stats': {},
            'num_objects': 0,
            'method_used': 'none'
        }
    
    def _preprocess_image(self, image: np.ndarray, channels: Optional[List[int]] = None) -> np.ndarray:
        """Preprocess image for analysis.
        
        Args:
            image: Input image array
            channels: Specific channels to use for segmentation
            
        Returns:
            Preprocessed image
        """
        try:
            # Extract specific channels if provided
            if channels and len(image.shape) == 3:
                if len(channels) == 1:
                    # Single channel for segmentation
                    processed = image[:, :, channels[0]]
                else:
                    # Multiple channels - use first for segmentation
                    processed = image[:, :, channels[0]]
            elif len(image.shape) == 3:
                # Convert RGB to grayscale
                processed = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            else:
                # Already grayscale
                processed = image.copy()
            
            # Ensure proper data type
            if processed.dtype != np.uint8:
                # Normalize to 0-255 range
                processed = ((processed - processed.min()) / (processed.max() - processed.min()) * 255).astype(np.uint8)
            
            return processed
            
        except Exception as e:
            logger.error(f"Image preprocessing failed: {e}")
            raise
    
    def _segment_objects(self, image: np.ndarray, method: SegmentationStrategy = SegmentationStrategy.THRESHOLD) -> List:
        """Segment objects using specified method.
        
        Args:
            image: Preprocessed image
            method: Segmentation strategy to use
            
        Returns:
            List of region properties
        """
        try:
            if method not in self.segmentation_methods:
                logger.warning(f"Unknown method {method}, using threshold")
                method = SegmentationStrategy.THRESHOLD
            
            return self.segmentation_methods[method](image)
            
        except Exception as e:
            logger.error(f"Object segmentation failed: {e}")
            return []
    
    def _threshold_segmentation(self, image: np.ndarray) -> List:
        """Simple threshold-based segmentation using Otsu's method.
        
        Args:
            image: Grayscale input image
            
        Returns:
            List of region properties
        """
        try:
            # Apply Gaussian blur to reduce noise
            blurred = cv2.GaussianBlur(image, (5, 5), 0)
            
            # Apply Otsu's threshold
            _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # Clean up with morphological operations
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            cleaned = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
            cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel, iterations=1)
            
            # Label connected components
            labeled = measure.label(cleaned)
            regions = measure.regionprops(labeled, intensity_image=image)
            
            # Filter by size
            min_area = self.config.get('min_object_area', 50)
            max_area = self.config.get('max_object_area', 10000)
            
            filtered_regions = [r for r in regions if min_area <= r.area <= max_area]
            
            logger.info(f"Threshold segmentation found {len(filtered_regions)} objects")
            return filtered_regions
            
        except Exception as e:
            logger.error(f"Threshold segmentation failed: {e}")
            return []
    
    def _watershed_segmentation(self, image: np.ndarray) -> List:
        """Watershed segmentation for overlapping objects.
        
        Args:
            image: Grayscale input image
            
        Returns:
            List of region properties
        """
        try:
            # Apply Gaussian filter
            blurred = cv2.GaussianBlur(image, (5, 5), 0)
            
            # Threshold to get binary image
            _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # Distance transform
            dist_transform = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
            
            # Find local maxima as markers
            _, markers = cv2.threshold(dist_transform, 0.4 * dist_transform.max(), 255, 0)
            markers = markers.astype(np.uint8)
            
            # Label markers
            _, markers = cv2.connectedComponents(markers)
            
            # Apply watershed
            markers = cv2.watershed(cv2.cvtColor(image, cv2.COLOR_GRAY2RGB), markers)
            
            # Extract regions
            regions = measure.regionprops(markers, intensity_image=image)
            
            # Filter by size
            min_area = self.config.get('min_object_area', 50)
            max_area = self.config.get('max_object_area', 10000)
            
            filtered_regions = [r for r in regions if min_area <= r.area <= max_area and r.label > 0]
            
            logger.info(f"Watershed segmentation found {len(filtered_regions)} objects")
            return filtered_regions
            
        except Exception as e:
            logger.error(f"Watershed segmentation failed: {e}")
            return []
    
    def _edge_segmentation(self, image: np.ndarray) -> List:
        """Edge-based segmentation using Canny edge detection.
        
        Args:
            image: Grayscale input image
            
        Returns:
            List of region properties
        """
        try:
            # Apply Gaussian blur
            blurred = cv2.GaussianBlur(image, (5, 5), 0)
            
            # Canny edge detection
            edges = cv2.Canny(blurred, 50, 150)
            
            # Close gaps in edges
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)
            
            # Fill holes
            filled = ndimage.binary_fill_holes(closed).astype(np.uint8) * 255
            
            # Label connected components
            labeled = measure.label(filled)
            regions = measure.regionprops(labeled, intensity_image=image)
            
            # Filter by size
            min_area = self.config.get('min_object_area', 50)
            max_area = self.config.get('max_object_area', 10000)
            
            filtered_regions = [r for r in regions if min_area <= r.area <= max_area]
            
            logger.info(f"Edge segmentation found {len(filtered_regions)} objects")
            return filtered_regions
            
        except Exception as e:
            logger.error(f"Edge segmentation failed: {e}")
            return []
    
    def _cellpose_segmentation(self, image: np.ndarray) -> List:
        """Cellpose segmentation (placeholder for future implementation).
        
        Args:
            image: Input image
            
        Returns:
            List of region properties
        """
        logger.warning("Cellpose segmentation not implemented, using threshold instead")
        return self._threshold_segmentation(image)
    
    def _stardist_segmentation(self, image: np.ndarray) -> List:
        """StarDist segmentation (placeholder for future implementation).
        
        Args:
            image: Input image
            
        Returns:
            List of region properties
        """
        logger.warning("StarDist segmentation not implemented, using threshold instead")
        return self._threshold_segmentation(image)
    
    def _extract_morphological_features(self, image: np.ndarray, regions: List) -> pd.DataFrame:
        """Extract morphological features from segmented regions.
        
        Args:
            image: Original image
            regions: List of region properties
            
        Returns:
            DataFrame with morphological features
        """
        try:
            features = []
            
            for i, region in enumerate(regions):
                feature_dict = {
                    'object_id': i,
                    'area': region.area,
                    'perimeter': region.perimeter,
                    'centroid_x': region.centroid[1],
                    'centroid_y': region.centroid[0],
                    'eccentricity': region.eccentricity,
                    'solidity': region.solidity,
                    'extent': region.extent,
                    'orientation': region.orientation,
                    'major_axis_length': region.major_axis_length,
                    'minor_axis_length': region.minor_axis_length,
                    'equivalent_diameter': region.equivalent_diameter,
                    'convex_area': region.convex_area,
                    'filled_area': region.filled_area,
                    'euler_number': region.euler_number
                }
                
                # Derived features
                if region.perimeter > 0:
                    feature_dict['compactness'] = (4 * np.pi * region.area) / (region.perimeter ** 2)
                else:
                    feature_dict['compactness'] = 0
                
                if region.minor_axis_length > 0:
                    feature_dict['aspect_ratio'] = region.major_axis_length / region.minor_axis_length
                else:
                    feature_dict['aspect_ratio'] = 1
                
                features.append(feature_dict)
            
            return pd.DataFrame(features)
            
        except Exception as e:
            logger.error(f"Morphological feature extraction failed: {e}")
            return pd.DataFrame()
    
    def _extract_intensity_features(self, image: np.ndarray, regions: List) -> pd.DataFrame:
        """Extract intensity-based features from segmented regions.
        
        Args:
            image: Original image
            regions: List of region properties
            
        Returns:
            DataFrame with intensity features
        """
        try:
            features = []
            
            # Convert to grayscale if needed
            if len(image.shape) == 3:
                gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            else:
                gray_image = image
            
            for i, region in enumerate(regions):
                # Get pixel intensities for this region
                coords = region.coords
                intensities = gray_image[coords[:, 0], coords[:, 1]]
                
                feature_dict = {
                    'object_id': i,
                    'mean_intensity': np.mean(intensities),
                    'median_intensity': np.median(intensities),
                    'std_intensity': np.std(intensities),
                    'min_intensity': np.min(intensities),
                    'max_intensity': np.max(intensities),
                    'intensity_range': np.max(intensities) - np.min(intensities),
                    'integrated_intensity': np.sum(intensities),
                    'weighted_centroid_x': region.weighted_centroid[1],
                    'weighted_centroid_y': region.weighted_centroid[0]
                }
                
                # Additional percentiles
                feature_dict['intensity_p25'] = np.percentile(intensities, 25)
                feature_dict['intensity_p75'] = np.percentile(intensities, 75)
                feature_dict['intensity_iqr'] = feature_dict['intensity_p75'] - feature_dict['intensity_p25']
                
                features.append(feature_dict)
            
            return pd.DataFrame(features)
            
        except Exception as e:
            logger.error(f"Intensity feature extraction failed: {e}")
            return pd.DataFrame()
    
    def _extract_texture_features(self, image: np.ndarray, regions: List) -> pd.DataFrame:
        """Extract texture features using Haralick features and Local Binary Patterns.
        
        Args:
            image: Original image
            regions: List of region properties
            
        Returns:
            DataFrame with texture features
        """
        try:
            features = []
            
            # Convert to grayscale if needed
            if len(image.shape) == 3:
                gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            else:
                gray_image = image
            
            for i, region in enumerate(regions):
                # Extract region of interest
                minr, minc, maxr, maxc = region.bbox
                roi = gray_image[minr:maxr, minc:maxc]
                mask = np.zeros_like(roi, dtype=bool)
                
                # Create mask for this region
                coords = region.coords
                local_coords = coords - [minr, minc]
                valid_coords = ((local_coords[:, 0] >= 0) & (local_coords[:, 0] < roi.shape[0]) & 
                               (local_coords[:, 1] >= 0) & (local_coords[:, 1] < roi.shape[1]))
                if np.any(valid_coords):
                    mask[local_coords[valid_coords, 0], local_coords[valid_coords, 1]] = True
                
                # Basic texture measures
                roi_masked = roi[mask] if np.any(mask) else roi.flatten()
                
                feature_dict = {
                    'object_id': i,
                    'texture_contrast': np.std(roi_masked) if len(roi_masked) > 1 else 0,
                    'texture_variance': np.var(roi_masked) if len(roi_masked) > 1 else 0,
                    'texture_skewness': self._compute_skewness(roi_masked),
                    'texture_kurtosis': self._compute_kurtosis(roi_masked),
                    'texture_energy': np.sum(roi_masked ** 2) if len(roi_masked) > 0 else 0
                }
                
                # Local Binary Pattern (simplified)
                if roi.size > 0:
                    lbp_var = self._compute_lbp_variance(roi)
                    feature_dict['lbp_variance'] = lbp_var
                else:
                    feature_dict['lbp_variance'] = 0
                
                features.append(feature_dict)
            
            return pd.DataFrame(features)
            
        except Exception as e:
            logger.error(f"Texture feature extraction failed: {e}")
            return pd.DataFrame()
    
    def _extract_spatial_features(self, image: np.ndarray, regions: List) -> pd.DataFrame:
        """Extract spatial and neighborhood features.
        
        Args:
            image: Original image
            regions: List of region properties
            
        Returns:
            DataFrame with spatial features
        """
        try:
            features = []
            
            # Compute centroids for distance calculations
            centroids = np.array([region.centroid for region in regions])
            
            for i, region in enumerate(regions):
                feature_dict = {
                    'object_id': i,
                    'distance_to_edge': self._distance_to_edge(region, image.shape),
                    'distance_to_center': self._distance_to_center(region, image.shape)
                }
                
                # Neighborhood analysis
                if len(centroids) > 1:
                    distances = np.linalg.norm(centroids - region.centroid, axis=1)
                    distances = distances[distances > 0]  # Exclude self
                    
                    if len(distances) > 0:
                        feature_dict['nearest_neighbor_distance'] = np.min(distances)
                        feature_dict['mean_neighbor_distance'] = np.mean(distances)
                        feature_dict['neighbor_count_50px'] = np.sum(distances < 50)
                        feature_dict['neighbor_count_100px'] = np.sum(distances < 100)
                    else:
                        feature_dict['nearest_neighbor_distance'] = np.inf
                        feature_dict['mean_neighbor_distance'] = np.inf
                        feature_dict['neighbor_count_50px'] = 0
                        feature_dict['neighbor_count_100px'] = 0
                else:
                    feature_dict['nearest_neighbor_distance'] = np.inf
                    feature_dict['mean_neighbor_distance'] = np.inf
                    feature_dict['neighbor_count_50px'] = 0
                    feature_dict['neighbor_count_100px'] = 0
                
                features.append(feature_dict)
            
            return pd.DataFrame(features)
            
        except Exception as e:
            logger.error(f"Spatial feature extraction failed: {e}")
            return pd.DataFrame()
    
    def _compute_summary_stats(self, morphological_features: pd.DataFrame, 
                             intensity_features: pd.DataFrame) -> Dict:
        """Compute summary statistics across all objects.
        
        Args:
            morphological_features: DataFrame with morphological features
            intensity_features: DataFrame with intensity features
            
        Returns:
            Dictionary with summary statistics
        """
        try:
            summary = {}
            
            if not morphological_features.empty:
                summary['morphological'] = {
                    'total_objects': len(morphological_features),
                    'mean_area': float(morphological_features['area'].mean()),
                    'std_area': float(morphological_features['area'].std()),
                    'mean_perimeter': float(morphological_features['perimeter'].mean()),
                    'mean_eccentricity': float(morphological_features['eccentricity'].mean()),
                    'mean_solidity': float(morphological_features['solidity'].mean())
                }
            
            if not intensity_features.empty:
                summary['intensity'] = {
                    'mean_intensity': float(intensity_features['mean_intensity'].mean()),
                    'overall_integrated_intensity': float(intensity_features['integrated_intensity'].sum()),
                    'intensity_cv': float(intensity_features['mean_intensity'].std() / intensity_features['mean_intensity'].mean())
                        if intensity_features['mean_intensity'].mean() > 0 else 0
                }
            
            return summary
            
        except Exception as e:
            logger.error(f"Summary statistics computation failed: {e}")
            return {}
    
    def _compute_skewness(self, data: np.ndarray) -> float:
        """Compute skewness of data."""
        if len(data) < 3:
            return 0.0
        mean_val = np.mean(data)
        std_val = np.std(data)
        if std_val == 0:
            return 0.0
        return np.mean(((data - mean_val) / std_val) ** 3)
    
    def _compute_kurtosis(self, data: np.ndarray) -> float:
        """Compute kurtosis of data."""
        if len(data) < 4:
            return 0.0
        mean_val = np.mean(data)
        std_val = np.std(data)
        if std_val == 0:
            return 0.0
        return np.mean(((data - mean_val) / std_val) ** 4) - 3
    
    def _compute_lbp_variance(self, image: np.ndarray) -> float:
        """Compute Local Binary Pattern variance (simplified version)."""
        if image.size < 9:
            return 0.0
        try:
            # Simple LBP calculation for center pixels
            center = image[1:-1, 1:-1]
            patterns = []
            
            offsets = [(-1, -1), (-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1)]
            
            for i in range(center.shape[0]):
                for j in range(center.shape[1]):
                    pattern = 0
                    center_val = center[i, j]
                    for k, (di, dj) in enumerate(offsets):
                        if image[i + 1 + di, j + 1 + dj] >= center_val:
                            pattern |= (1 << k)
                    patterns.append(pattern)
            
            return float(np.var(patterns)) if patterns else 0.0
        except:
            return 0.0
    
    def _distance_to_edge(self, region, image_shape: Tuple[int, int]) -> float:
        """Compute minimum distance from region centroid to image edge."""
        cy, cx = region.centroid
        height, width = image_shape[:2]
        
        distances = [cy, height - cy, cx, width - cx]
        return float(min(distances))
    
    def _distance_to_center(self, region, image_shape: Tuple[int, int]) -> float:
        """Compute distance from region centroid to image center."""
        cy, cx = region.centroid
        height, width = image_shape[:2]
        center_y, center_x = height / 2, width / 2
        
        return float(np.sqrt((cy - center_y) ** 2 + (cx - center_x) ** 2))
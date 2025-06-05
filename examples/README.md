# Anton Examples

This directory contains example scripts demonstrating how to use Anton for microscopy image analysis.

## Basic Analysis Example

The `basic_analysis.py` script demonstrates Anton's core functionality:

1. Full pipeline analysis combining quantitative and qualitative approaches
2. Standalone quantitative analysis using traditional computer vision
3. Qualitative analysis with CMPO phenotype mapping

### Running the Example

1. First, ensure you have the required dependencies installed:
```bash
pip install -r requirements.txt
```

2. Place your microscopy image in the `data/sample_images` directory:
```bash
mkdir -p data/sample_images
# Copy your image to data/sample_images/cells.tif
```

3. Run the example:
```bash
python examples/basic_analysis.py
```

### Expected Output

The script will:
1. Run a full analysis pipeline on your image
2. Perform quantitative analysis using Cellpose segmentation
3. Conduct qualitative analysis with CMPO phenotype mapping
4. Save results to the `results` directory:
   - `full_analysis.json`: Complete analysis results
   - `morphological_features.csv`: Quantitative morphological measurements
   - `intensity_features.csv`: Intensity-based features
   - `texture_features.csv`: Texture analysis results

### Configuration

The example uses a default configuration that you can modify:
- Segmentation method: Cellpose
- Analysis channels: [0, 1] (DAPI and GFP)
- Batch size: 10
- Confidence threshold: 0.5

You can modify these parameters in the script to suit your needs.

## Sample Images

The example expects a multi-channel microscopy image with:
- Channel 0: DAPI (nuclear stain)
- Channel 1: GFP (or other marker)

If your image has different channels, modify the `channels` parameter in the configuration. 
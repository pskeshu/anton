 `anton` is a minimal Python framework for Vision-Language Model (VLM)-driven fluorescence microscopy image analysis, using the Cellular Microscopy Phenotype Ontology (CMPO). It interprets microscopist language (e.g., “Identify apoptotic cells in DAPI channel”) and structures abstract pipelines for CMPO phenotype identification.
 
 ## Installation
 ```bash
 pip install -r requirements.txt
 ```
 
 ## Usage
 Run the main function with a goal and image path:
 ```bash
 python -m anton.main --goal "Identify apoptotic cells in DAPI-stained channel 1" --image path/to/image.tiff --metadata path/to/metadata.xml
 ```
 
 ## Example
 See `examples/phenotype_detection.py` (coming soon).
 
 ## Dependencies
 - Python 3.9+
 - Pandas
 
 ## License
 MIT
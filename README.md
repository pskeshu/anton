# Anton - Advanced Microscopy Analysis Tool

Anton is a sophisticated microscopy image analysis tool that combines traditional computer vision techniques with Vision Language Models (VLM) for comprehensive cellular phenotype analysis.

## Features

- **Hybrid Analysis Pipeline**
  - Quantitative analysis using traditional computer vision
  - Qualitative analysis using Vision Language Models
  - CMPO (Cellular Microscopy Phenotype Ontology) integration
  - Cross-validation between quantitative and qualitative results

- **Segmentation Capabilities**
  - Multiple segmentation strategies:
    - Traditional (Otsu, Watershed)
    - Deep Learning (Cellpose, StarDist)
  - Configurable segmentation parameters
  - Region-based analysis

- **Feature Extraction**
  - Morphological features (area, perimeter, eccentricity, etc.)
  - Intensity-based features (mean, std, min, max, etc.)
  - Texture analysis (Haralick features, LBP)
  - Spatial relationships (neighborhood analysis)
  - Population-level insights

- **CMPO Integration**
  - Automatic phenotype mapping
  - Ontology-based validation
  - Confidence scoring
  - Export capabilities (CSV, JSON)

## Architecture Overview

```mermaid
graph TB
    %% User Input
    User[👤 User Input<br/>Goal + Image + Config] --> Pipeline[🔄 AnalysisPipeline<br/>Core Orchestration]
    
    %% Core Pipeline branching to two main analysis paths
    Pipeline --> VLM_Path[🧠 VLM Analysis Path]
    Pipeline --> CV_Path[🔬 Computer Vision Path]
    
    %% VLM Analysis Pipeline (4 stages)
    VLM_Path --> Stage1[🌍 Stage 1: Global Scene<br/>Image quality, staining, characteristics]
    Stage1 --> Stage2[🎯 Stage 2: Object Detection<br/>Structure identification, segmentation guidance]
    Stage2 --> Stage3[🔍 Stage 3: Feature Analysis<br/>Texture patterns, CMPO mapping]
    Stage3 --> Stage4[📊 Stage 4: Population Insights<br/>Prevalence, spatial patterns]
    
    %% VLM Interface
    Stage1 -.-> VLM[🤖 VLM Interface<br/>Claude-3/GPT-4V]
    Stage2 -.-> VLM
    Stage3 -.-> VLM
    Stage4 -.-> VLM
    
    %% Computer Vision Path
    CV_Path --> Segmentation[⚙️ Segmentation Strategies]
    Segmentation --> Cellpose[🧬 Cellpose<br/>Deep Learning]
    Segmentation --> StarDist[⭐ StarDist<br/>Deep Learning]
    Segmentation --> Traditional[📐 Traditional CV<br/>Otsu, Watershed, Edge]
    
    %% Feature Extraction
    Cellpose --> Features[📏 Feature Extraction]
    StarDist --> Features
    Traditional --> Features
    
    Features --> Morphological[📊 Morphological<br/>Area, perimeter, shape]
    Features --> Intensity[💡 Intensity<br/>Mean, std, distribution]
    Features --> Texture[🌐 Texture<br/>Haralick, LBP]
    Features --> Spatial[📍 Spatial<br/>Neighborhood analysis]
    
    %% CMPO Integration
    Stage3 --> CMPO[🗃️ CMPO Ontology]
    Features --> CMPO
    
    CMPO --> OLS[🌐 OLS API Download]
    CMPO --> OWL[📄 OWL File Parsing]
    CMPO --> Cache[💾 Intelligent Caching]
    CMPO --> Phenotypes[🏷️ Phenotype Database<br/>CMPO_0000094: Apoptosis<br/>CMPO_0000140: Mitosis<br/>CMPO_0000077: Abnormal morphology]
    
    %% Hybrid Analysis & Cross-validation
    Stage4 --> Hybrid[🔀 Hybrid Analysis]
    Spatial --> Hybrid
    
    Hybrid --> CrossVal[✅ Cross-Validation<br/>Quantitative ↔ Qualitative]
    Hybrid --> Confidence[📈 Confidence Scoring]
    Hybrid --> Agreement[🤝 Agreement Metrics]
    
    %% Results & Output
    CrossVal --> Results[📋 Results]
    Confidence --> Results
    Agreement --> Results
    
    Results --> JSON[📄 JSON Export<br/>Full analysis results]
    Results --> CSV[📊 CSV Export<br/>Feature tables]
    Results --> Pandas[🐼 Pandas DataFrames<br/>Structured data]
    
    %% Configuration & Prompts
    Config[⚙️ Configuration<br/>JSON-based settings] --> Pipeline
    Prompts[📝 Structured Prompts<br/>Stage-specific templates] --> VLM
    
    %% Async Processing
    VLM --> Async[⚡ Async Processing<br/>Batch optimization<br/>Intelligent caching]
    
    %% Styling
    classDef userInput fill:#e1f5fe
    classDef pipeline fill:#f3e5f5
    classDef vlm fill:#fff3e0
    classDef cv fill:#e8f5e8
    classDef cmpo fill:#e3f2fd
    classDef hybrid fill:#fce4ec
    classDef results fill:#f1f8e9
    classDef config fill:#fafafa
    
    class User,Config userInput
    class Pipeline pipeline
    class VLM_Path,Stage1,Stage2,Stage3,Stage4,VLM,Async,Prompts vlm
    class CV_Path,Segmentation,Cellpose,StarDist,Traditional,Features,Morphological,Intensity,Texture,Spatial cv
    class CMPO,OLS,OWL,Cache,Phenotypes cmpo
    class Hybrid,CrossVal,Confidence,Agreement hybrid
    class Results,JSON,CSV,Pandas results
```

## Project Structure

```
anton/
├── analysis/
│   ├── quantitative.py  # Traditional CV analysis with multiple segmentation methods
│   └── qualitative.py   # VLM-driven analysis with feature extraction
├── cmpo/
│   ├── ontology.py      # CMPO ontology management and lookup
│   └── mapping.py       # Phenotype mapping and validation
├── vlm/
│   └── interface.py     # VLM integration for qualitative analysis
├── core/
│   ├── pipeline.py      # Main analysis pipeline orchestration
│   └── config.py        # Configuration management
├── utils/              # Utility functions
└── examples/           # Usage examples
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/pskeshu/anton.git
cd anton
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

Basic usage example:

```python
from anton.core.pipeline import AntonPipeline
from anton.vlm.interface import VLMInterface

# Initialize components
vlm = VLMInterface()
pipeline = AntonPipeline(vlm, config={})

# Run analysis
results = await pipeline.analyze_image("path/to/image.tif", "detect_cell_death")
```

### Quantitative Analysis

```python
from anton.analysis.quantitative import QuantitativeAnalyzer

# Initialize analyzer with specific segmentation method
analyzer = QuantitativeAnalyzer(config={
    'segmentation_method': 'cellpose'
})

# Extract features
features = analyzer.extract_quantitative_features(
    image_path="path/to/image.tif",
    channels=[0, 1]
)
```

### Qualitative Analysis

```python
from anton.analysis.qualitative import QualitativeAnalyzer
from anton.cmpo.ontology import CMPOOntology

# Initialize components
ontology = CMPOOntology()
analyzer = QualitativeAnalyzer(vlm_interface, ontology)

# Extract qualitative features
features = await analyzer.extract_qualitative_features(
    image_path="path/to/image.tif",
    regions=quantitative_results['regions'],
    config={'batch_size': 10}
)
```

## Configuration

The analysis pipeline can be configured through the `config` parameter:

```python
config = {
    'segmentation': {
        'method': 'cellpose',  # or 'stardist', 'threshold', 'watershed'
        'parameters': {
            'batch_size': 8,
            'diameter': 30
        }
    },
    'analysis': {
        'channels': [0, 1],
        'batch_size': 10,
        'confidence_threshold': 0.5
    },
    'cmpo': {
        'ontology_path': 'data/cmpo.json',
        'cache_path': 'data/cmpo_cache.pkl'
    }
}
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

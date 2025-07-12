---
title: Anton Microscopy Analysis
emoji: 🔬
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
license: mit
---

# Anton 🔬

**AI-powered microscopy analysis that speaks biology's language**

Anton combines traditional computer vision with Vision Language Models (VLMs) to analyze microscopy images and automatically classify cellular phenotypes using standardized biological terminology (CMPO ontology).

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## ✨ What makes Anton special?

- **Hybrid Analysis**: Combines quantitative computer vision with qualitative AI interpretation
- **Biological Intelligence**: Uses Vision Language Models to understand cellular phenotypes like a biologist would
- **Standardized Output**: Automatically maps findings to CMPO (Cellular Microscopy Phenotype Ontology) terms
- **Multiple Segmentation Methods**: Supports Cellpose, StarDist, traditional thresholding, and more
- **Web Interface**: Interactive Streamlit UI for easy analysis

## 🚀 Quick Start

### 1. Installation

```bash
git clone https://github.com/pskeshu/anton.git
cd anton
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. API Setup

Anton needs at least one Vision Language Model API key:

**Option A: Google Gemini (Free tier available)**
1. Get API key from [Google AI Studio](https://aistudio.google.com/app/apikey)
2. Create `.env` file: `cp .env.example .env`
3. Add your key: `GOOGLE_API_KEY="your-key-here"`

**Option B: Anthropic Claude** 
1. Get API key from [Anthropic Console](https://console.anthropic.com/)
2. Add to `.env`: `ANTHROPIC_API_KEY="your-key-here"`

### 3. Test Installation

```bash
# Quick 30-second demo
python scripts/utilities/quick_demo.py

# Interactive web interface
streamlit run scripts/demos/anton_simple_ui.py
```

## 📊 See Anton in Action

### Web Interface Demo
Launch the interactive interface to upload images and see real-time analysis:
```bash
streamlit run scripts/demos/anton_simple_ui.py
```

### Comprehensive Analysis Demo
Run a complete analysis pipeline on sample data:
```bash
python scripts/demos/demo.py
```

### What You'll See
Anton analyzes images through a 4-stage pipeline:

1. **🌍 Global Scene Understanding** - Overall image quality and content assessment
2. **🎯 Object Detection & Segmentation** - Identifies cellular structures and guides segmentation
3. **🔍 Feature-Level Analysis** - Detailed analysis of individual detected objects
4. **📊 Population-Level Insights** - Summary statistics and phenotype classification

## 🧬 CMPO Integration

Anton's standout feature is its integration with the Cellular Microscopy Phenotype Ontology (CMPO):

- **399 Official CMPO Terms**: Complete ontology with semantic relationships
- **Biological Accuracy**: Prevents impossible phenotype mappings
- **Two-Stage Validation**: Candidate generation followed by biological reasoning
- **Evidence Tracking**: Full provenance for all mapping decisions

## 💻 Basic Usage

### Python API
```python
from anton.core.pipeline import AnalysisPipeline

# Initialize with your preferred VLM provider
config = {'vlm_provider': 'gemini', 'channels': [0]}
pipeline = AnalysisPipeline(config)

# Run analysis
results = await pipeline.run_pipeline("path/to/image.tif")
# Or synchronously:
results = pipeline.run_pipeline_sync("path/to/image.tif")
```

### Quantitative Analysis Only
```python
from anton.analysis.quantitative import QuantitativeAnalyzer

analyzer = QuantitativeAnalyzer(config={
    'segmentation_method': 'cellpose'
})

features = analyzer.extract_quantitative_features(
    image_path="path/to/image.tif",
    channels=[0, 1]
)
```

## 🏗️ Architecture

```
📁 anton/
├── 🧬 analysis/      # Quantitative & qualitative analysis engines
├── 🎯 cmpo/          # CMPO ontology integration
├── ⚙️ core/          # Pipeline orchestration
├── 🤖 vlm/           # Vision Language Model interface
└── 🛠️ utils/         # Image I/O and validation

📁 scripts/
├── 🎮 demos/         # Interactive demos and UI
├── 📊 bbbc013/       # BBBC013 dataset analysis
└── 🔧 utilities/     # General utilities

📁 examples/          # Usage tutorials and examples
📁 tests/             # Test suite
```

## 🔧 Configuration

Customize Anton's behavior through configuration:

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
        'semantic_validation': True
    }
}
```

## 📚 Documentation

- **[Getting Started](examples/README.md)**: Tutorials and examples
- **[API Reference](docs/)**: Complete API documentation
- **[Directory Structure](DIRECTORY_STRUCTURE.md)**: Full project navigation guide

## 🛠️ Supported Segmentation Methods

- **Cellpose**: Deep learning-based cell segmentation
- **StarDist**: Star-convex polygon detection
- **Traditional Methods**: Otsu thresholding, watershed, edge detection
- **Configurable Parameters**: Customize for your specific image types

## 📋 Requirements

- Python 3.8+ (recommended: Python 3.10+)
- At least one VLM API key (Gemini, Claude, or OpenAI)
- Standard scientific Python packages (numpy, pandas, scikit-image, etc.)

## 🤝 Contributing

We welcome contributions! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙋 Support

Having issues? Check out our [troubleshooting guide](docs/troubleshooting.md) or open an issue on GitHub.

---

**Ready to analyze your microscopy data?** Start with the [quick demo](scripts/utilities/quick_demo.py) or try the [web interface](scripts/demos/anton_simple_ui.py)!
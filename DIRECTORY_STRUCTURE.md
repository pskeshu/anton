# Anton Repository Structure

This document describes the organized structure of the Anton microscopy analysis framework repository.

## 📁 Directory Organization

```
anton/
├── 🧬 anton/                    # Core Anton Framework
│   ├── analysis/               # Analysis engines
│   │   ├── qualitative.py      # VLM-based phenotype analysis
│   │   └── quantitative.py     # Statistical analysis
│   ├── cmpo/                   # CMPO ontology integration
│   │   ├── data/cmpo.json      # CMPO ontology data (399+ terms)
│   │   ├── mapping.py          # Semantic CMPO mapping
│   │   ├── ontology.py         # Ontology management
│   │   └── examples.py         # CMPO usage examples
│   ├── core/                   # Core pipeline
│   │   ├── config.py           # Configuration management
│   │   └── pipeline.py         # Main analysis pipeline
│   ├── utils/                  # Utilities
│   │   ├── image_io.py         # Image loading/processing
│   │   └── validation.py       # Input validation
│   ├── vlm/                    # Vision Language Model interface
│   │   └── interface.py        # VLM provider abstraction
│   └── main.py                 # Main entry point
│
├── 📊 scripts/                  # Analysis Scripts
│   ├── bbbc013/                # BBBC013 dataset scripts
│   │   ├── bbbc013_demo.py                    # Basic demo
│   │   ├── bbbc013_interactive_explorer.py    # Interactive analysis
│   │   └── bbbc013_phenotype_analysis.py      # Phenotype detection
│   ├── bbbc021/                # BBBC021 dataset scripts
│   │   ├── bbbc021_connection_discovery.py    # Connection analysis
│   │   ├── bbbc021_downloader.py              # Dataset downloader
│   │   ├── bbbc021_multichannel_discovery.py  # Multi-channel analysis
│   │   ├── bbbc021_quick_multichannel.py      # Quick analysis
│   │   ├── bbbc021_simple_multichannel.py     # Simple analysis
│   │   └── run_multichannel_discovery.py      # Full pipeline runner
│   ├── demos/                  # Demo and test scripts
│   │   ├── anton_simple_ui.py              # Simple UI demo
│   │   ├── anton_visual_analysis.ipynb     # Jupyter notebook demo
│   │   ├── demo.py                         # Basic demo
│   │   ├── demo_semantic_mapping.py        # Semantic mapping demo
│   │   ├── run_semantic_discovery.py       # Semantic analysis pipeline
│   │   ├── semantic_cmpo_mapper.py         # Semantic mapper
│   │   ├── show_vlm_analysis.py            # VLM analysis viewer
│   │   ├── test_one_compound.py            # Single compound test
│   │   ├── test_single_composite.py        # Single image test
│   │   ├── test_single_multichannel.py     # Multi-channel test
│   │   └── test_vlm_providers.py           # VLM provider tests
│   └── utilities/              # Utility scripts
│       ├── create_multichannel_composites.py  # Composite image creation
│       ├── debug_filenames.py                 # Filename debugging
│       ├── extract_all_zips.py                # Bulk extraction
│       ├── phrase_aware_cmpo.py               # Legacy CMPO mapping
│       ├── quick_demo.py                      # Quick demo
│       └── quick_discovery_test.py            # Quick test
│
├── 📈 results/                  # Analysis Results
│   ├── bbbc013/                # BBBC013 results
│   │   ├── bbbc013_cmpo_analysis.png       # Analysis visualization
│   │   └── bbbc013_phenotype_results.json  # Phenotype results
│   ├── bbbc021/                # BBBC021 results
│   │   ├── bbbc021_connection_discovery_results.json
│   │   ├── bbbc021_mock_metadata.csv
│   │   └── bbbc021_real_discovery_results.json
│   └── semantic_mapping/       # Semantic mapping results
│       ├── detailed_vlm_analysis.json
│       ├── semantic_mapping_demo.json
│       ├── test_one_compound_result.json
│       └── test_single_multichannel_results.json
│
├── 🗄️ data/                    # Datasets
│   ├── bbbc013/                # BBBC013 fluorescence images
│   │   ├── BBBC013_v1_images_bmp/         # Raw BMP images (192 files)
│   │   ├── BBBC013_v1_images_bmp.zip      # Original archive
│   │   └── BBBC013_v1_platemap.txt        # Plate mapping
│   ├── bbbc021/                # BBBC021 compound screen
│   │   ├── images/                        # Raw TIF images (2,170+ files)
│   │   ├── composites/                    # RGB composite images (20 files)
│   │   ├── analysis_subset.csv            # Analysis subset metadata
│   │   ├── compound_metadata.csv          # Compound information
│   │   ├── image_metadata.csv             # Image metadata
│   │   ├── moa_metadata.csv               # Mechanism of action data
│   │   └── multichannel_analysis_dataset.csv
│   ├── sample_images/          # Demo images
│   │   ├── demo_images/                   # Sample images (23 files)
│   │   └── metadata.csv                   # Sample metadata
│   └── cmpo.json               # CMPO ontology backup
│
├── 📚 docs/                    # Documentation
│   ├── BBBC013_ANALYSIS_SUMMARY.md    # BBBC013 analysis summary
│   ├── SESSION_NOTES.md               # Development session notes
│   └── multi_channel_design.md        # Multi-channel design docs
│
├── 🧪 examples/                # Example usage
│   ├── basic_analysis.py          # Basic usage example
│   ├── phenotype_detection.py     # Phenotype detection example
│   └── README.md                  # Examples documentation
│
├── 🎯 prompts/                 # VLM prompts
│   ├── stage1_global.txt          # Global scene analysis
│   ├── stage2_objects.txt          # Object detection
│   ├── stage3_features.txt         # Feature analysis
│   ├── stage4_population.txt       # Population analysis
│   └── cmpo_mapping.txt           # CMPO mapping prompt
│
├── 🧪 tests/                   # Test suite
│   ├── test_pipeline.py           # Pipeline tests
│   ├── test_qualitative_analysis.py
│   ├── test_vlm_interface.py      # VLM interface tests
│   ├── conftest.py               # Test configuration
│   └── run_tests.py              # Test runner
│
├── 🔧 Configuration Files
│   ├── requirements.txt          # Python dependencies
│   ├── LICENCE                   # License
│   ├── README.md                 # Main documentation
│   └── DIRECTORY_STRUCTURE.md    # This file
│
└── 🐍 venv/                    # Virtual environment
```

## 🚀 Quick Start Paths

### Core Framework
- **Main Entry**: `anton/main.py`
- **Pipeline**: `anton/core/pipeline.py`
- **CMPO Mapping**: `anton/cmpo/mapping.py`

### Analysis Scripts
- **BBBC021 Multi-Channel**: `scripts/bbbc021/bbbc021_multichannel_discovery.py`
- **Semantic Mapping Demo**: `scripts/demos/demo_semantic_mapping.py`
- **Quick Test**: `scripts/demos/test_single_composite.py`

### Results
- **Latest Semantic Results**: `results/semantic_mapping/`
- **BBBC021 Analysis**: `results/bbbc021/`

## 📋 File Categories

### 🟢 Active/Current Files
- All files in `anton/` (core framework)
- Multi-channel analysis scripts in `scripts/bbbc021/`
- Semantic mapping scripts in `scripts/demos/`
- Recent results in `results/semantic_mapping/`

### 🟡 Legacy/Reference Files
- `scripts/utilities/phrase_aware_cmpo.py` (old bag-of-words mapping)
- Some early analysis scripts in utilities

### 🔵 Data Files
- Raw images in `data/bbbc013/` and `data/bbbc021/`
- Composite images in `data/bbbc021/composites/`
- Metadata CSV files

## 🎯 Key Improvements Made

1. **Separated Code from Results**: Scripts in `scripts/`, outputs in `results/`
2. **Dataset Organization**: Clear data structure with subdirectories
3. **Functional Grouping**: Scripts grouped by purpose (demos, utilities, dataset-specific)
4. **Documentation Centralized**: All docs in `docs/` directory
5. **Clean Root**: Only essential files in root directory
6. **Clear Hierarchy**: Logical nested structure for easy navigation

## 🔍 Finding Files

Use this guide to locate specific functionality:

- **Want to run analysis?** → `scripts/`
- **Looking for results?** → `results/`
- **Need example code?** → `examples/`
- **Working with data?** → `data/`
- **Reading documentation?** → `docs/`
- **Understanding the code?** → `anton/`
# Anton Framework Tests

This directory contains the test suite for the Anton microscopy phenotype analysis framework.

## Test Structure

- **`run_tests.py`**: Main test runner for basic functionality tests
- **`test_pipeline.py`**: Pytest-based tests for the core analysis pipeline
- **`test_qualitative_analysis.py`**: Tests for the qualitative analysis components
- **`test_vlm_interface.py`**: Tests for the VLM (Vision Language Model) interface
- **`conftest.py`**: Pytest configuration and shared fixtures
- **`__init__.py`**: Makes tests a proper Python package

## Running Tests

### Basic Test Runner
```bash
# Activate virtual environment
source venv/bin/activate

# Run basic functionality tests
python tests/run_tests.py
```

### Pytest (if installed)
```bash
# Install pytest if not available
pip install pytest pytest-asyncio

# Run all tests
pytest tests/

# Run specific test file
pytest tests/test_pipeline.py

# Run with verbose output
pytest -v tests/
```

## Test Coverage

The current test suite covers:

✅ **Core Pipeline**
- Pipeline initialization
- Async/sync execution
- All 4 analysis stages
- Error handling

✅ **Qualitative Analysis**
- Feature extraction
- Patch processing
- Population insights
- CMPO mapping

✅ **VLM Interface**
- Multiple providers (Claude, Gemini)
- Mock response handling
- Image processing

## Test Data

Tests use sample images from `data/sample_images/demo_images/` directory. If these images are not available, relevant tests will be skipped.

## Mock Data

Tests currently use mock VLM responses for consistent testing. Real API integration tests can be enabled by providing actual API keys in environment variables:

- `ANTHROPIC_API_KEY` for Claude
- `GOOGLE_API_KEY` for Gemini
- `OPENAI_API_KEY` for OpenAI

## Adding New Tests

When adding new functionality:

1. Add unit tests to the appropriate test file
2. Update the basic test runner if needed
3. Add integration tests for end-to-end workflows
4. Update this README if new test categories are added
# Test Suite Documentation

## Overview
Comprehensive test suite for the Enterprise Policy Chatbot with support for both Gemini and OpenAI providers.

## Test Structure

```
tests/
├── __init__.py                 # Test package initialization
├── conftest.py                # Shared fixtures and configuration
├── test_settings.py           # Configuration and provider detection tests
├── test_embeddings.py         # Embeddings provider tests
├── test_chat.py              # Chat LLM provider tests
└── test_api.py               # API endpoint tests
```

## Running Tests

### Install Test Dependencies
```bash
pip install -r requirements.txt
```

### Run All Tests
```bash
pytest
```

### Run with Coverage
```bash
pytest --cov=app --cov=config --cov-report=html
```

### Run Specific Test Files
```bash
# Configuration tests
pytest tests/test_settings.py

# Embeddings tests
pytest tests/test_embeddings.py

# Chat LLM tests
pytest tests/test_chat.py

# API tests
pytest tests/test_api.py
```

### Run Tests by Marker
```bash
# Unit tests only
pytest -m unit

# API tests only
pytest -m api

# Embeddings tests
pytest -m embeddings

# Chat tests
pytest -m chat
```

### Run Specific Test Class or Function
```bash
# Run specific test class
pytest tests/test_settings.py::TestProviderDetection

# Run specific test function
pytest tests/test_embeddings.py::TestEmbeddingsFactory::test_get_embeddings_returns_gemini
```

### Verbose Output
```bash
pytest -v
```

### Show Print Statements
```bash
pytest -s
```

### Stop on First Failure
```bash
pytest -x
```

### Run Last Failed Tests
```bash
pytest --lf
```

## Test Categories

### Unit Tests (`-m unit`)
- **test_settings.py**: Tests configuration and provider auto-detection
- **test_embeddings.py**: Tests embedding providers and factory
- **test_chat.py**: Tests chat LLM providers and factory

### API Tests (`-m api`)
- **test_api.py**: Tests FastAPI endpoints

## Test Coverage

### Current Coverage
Run `pytest --cov` to see current coverage report.

### Coverage Report
After running tests with coverage, view the HTML report:
```bash
open htmlcov/index.html
```

### Minimum Coverage
- Target: 70% (configured in pytest.ini)
- Tests will fail if coverage drops below this threshold

## Test Fixtures

### Environment Fixtures (from conftest.py)
- `mock_env_vars`: Auto-applied test environment variables
- `gemini_only_env`: Environment with only Gemini API key
- `openai_only_env`: Environment with only OpenAI API key
- `no_api_keys_env`: Environment with no API keys

### Mock Fixtures
- `mock_openai_response`: Mock OpenAI API response
- `mock_gemini_response`: Mock Gemini API response
- `mock_vector_store`: Mock vector store
- `mock_knowledge_graph`: Mock knowledge graph
- `mock_document_processor`: Mock document processor

### Sample Data Fixtures
- `sample_document`: Sample document for testing
- `sample_chat_request`: Sample chat request
- `mock_pdf_content`: Mock PDF content

## Writing New Tests

### Test Naming Convention
```python
class TestFeatureName:
    """Test description"""

    def test_specific_behavior(self):
        """Should do X when Y"""
        # Arrange
        # Act
        # Assert
```

### Using Fixtures
```python
def test_with_fixture(self, mock_env_vars, sample_document):
    """Test using fixtures"""
    # Fixtures are automatically injected
    assert sample_document["document_type"] == "policy"
```

### Adding Markers
```python
@pytest.mark.slow
def test_slow_operation(self):
    """Test that takes a long time"""
    pass

@pytest.mark.integration
def test_integration_flow(self):
    """Test end-to-end flow"""
    pass
```

## Test Best Practices

### 1. Arrange-Act-Assert Pattern
```python
def test_example(self):
    # Arrange: Set up test data
    data = {"key": "value"}

    # Act: Execute the code being tested
    result = process_data(data)

    # Assert: Verify the results
    assert result["processed"] is True
```

### 2. Mock External Dependencies
```python
@patch('app.services.external_api')
def test_with_mock(self, mock_api):
    mock_api.call.return_value = "expected"
    result = my_function()
    assert result == "expected"
```

### 3. Test Both Success and Failure Cases
```python
def test_success_case(self):
    """Test normal operation"""
    assert function(valid_input) == expected_output

def test_error_case(self):
    """Test error handling"""
    with pytest.raises(ValueError):
        function(invalid_input)
```

### 4. Use Descriptive Test Names
```python
# Good ✅
def test_gemini_provider_selected_when_gemini_key_present(self):
    pass

# Bad ❌
def test_provider(self):
    pass
```

### 5. Keep Tests Independent
- Each test should be able to run independently
- Don't rely on test execution order
- Use fixtures for shared setup

## Continuous Integration

### GitHub Actions Example
```yaml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - uses: actions/setup-python@v2
        with:
          python-version: '3.9'
      - run: pip install -r requirements.txt
      - run: pytest --cov --cov-report=xml
      - uses: codecov/codecov-action@v2
```

## Debugging Tests

### Run with PDB
```bash
pytest --pdb
```

### Run with more detail
```bash
pytest -vv --tb=long
```

### Show all variables on failure
```bash
pytest --showlocals
```

## Common Issues

### Import Errors
- Ensure PYTHONPATH includes project root
- Check pytest.ini configuration

### Fixture Not Found
- Check conftest.py is in the correct location
- Verify fixture name spelling

### Mock Not Working
- Ensure patch path is correct (where it's used, not where it's defined)
- Check if patch is applied before function is imported

## Test Maintenance

### Keep Tests Fast
- Mock external API calls
- Use in-memory databases for tests
- Avoid unnecessary sleep/wait statements

### Update Tests When Code Changes
- Tests should be updated along with code
- Maintain test coverage

### Review Test Failures
- Don't ignore failing tests
- Fix or update tests immediately

## Resources

- [Pytest Documentation](https://docs.pytest.org/)
- [pytest-mock Documentation](https://pytest-mock.readthedocs.io/)
- [Coverage.py Documentation](https://coverage.readthedocs.io/)

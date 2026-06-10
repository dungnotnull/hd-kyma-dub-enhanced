# Contributing to kyma-dub-enhanced

Thank you for your interest in contributing to kyma-dub-enhanced!

## Setting Up Development Environment

```bash
# Clone the repository
git clone https://github.com/your-org/kyma-dub-enhanced.git
cd kyma-dub-enhanced

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install dev dependencies
pip install pytest pytest-asyncio black isort flake8 mypy

# Install pre-commit hooks (optional)
pip install pre-commit
pre-commit install
```

## Code Style

We follow PEP 8 with these modifications:

- Maximum line length: 100 characters
- Use `snake_case` for functions and variables
- Use `PascalCase` for classes
- Add type hints to all function signatures
- Use `from __future__ import annotations` for forward references

### Formatting

```bash
# Format code with black
black agent/ tools/

# Sort imports with isort
isort agent/ tools/

# Check linting with flake8
flake8 agent/ tools/

# Type checking with mypy (optional)
mypy agent/
```

## Running Tests

```bash
# Run all tests
pytest tests/test_agent.py -v

# Run specific test class
pytest tests/test_agent.py::TestASRTranscriber -v

# Run with coverage
pytest tests/test_agent.py --cov=agent --cov=tools --cov-report=html

# Run specific test
pytest tests/test_agent.py::TestASRTranscriber::test_transcript_result_to_srt -v
```

## Development Workflow

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/my-feature`
3. Make your changes
4. Run tests: `pytest tests/test_agent.py -v`
5. Format code: `black agent/ tools/`
6. Commit changes: `git commit -m "Add my feature"`
7. Push to fork: `git push origin feature/my-feature`
8. Open a pull request

## Commit Message Convention

We follow conventional commits:

- `feat:` New feature
- `fix:` Bug fix
- `docs:` Documentation changes
- `test:` Test additions/changes
- `refactor:` Code refactoring
- `chore:` Maintenance tasks

Examples:
- `feat(translator): add support for Swedish language`
- `fix(tts): correct audio concatenation bug`
- `docs: update installation instructions`

## Project Structure

```
kyma-dub-enhanced/
├── agent/
│   ├── main.py              # CLI and FastAPI server
│   ├── orchestrator.py      # E2E pipeline orchestration
│   ├── modules/             # AI pipeline modules
│   │   ├── asr_transcriber.py
│   │   ├── script_translator.py
│   │   ├── tts_synthesizer.py
│   │   ├── lipsync_engine.py
│   │   └── mos_evaluator.py
│   ├── tools/               # Utilities
│   │   └── video_processor.py
│   └── memory/              # Persistent memory
│       └── memory_manager.py
├── tools/                   # Shared tools
│   ├── llm_client.py
│   ├── hf_model_manager.py
│   └── knowledge_updater.py
├── tests/                   # Test suite
│   ├── test_agent.py
│   └── test-scenarios.md
├── docker/                  # Docker deployment
│   ├── Dockerfile
│   └── docker-compose.yml
├── config/                  # Configuration files
│   ├── agent_config.yaml
│   └── .env.example
└── docs/                    # Documentation
    ├── PROJECT-DEVELOPMENT-PHASE-TRACKING.md
    ├── PROJECT-detail.md
    └── SECOND-KNOWLEDGE-BRAIN.md
```

## Adding a New Language

1. Add language code to `SUPPORTED_LANGUAGES` in `agent/modules/script_translator.py`
2. Add SeamlessM4T language mapping in `_to_seamless_lang()`
3. Add Bark voice preset in `BARK_LANG_PREFIXES` in `agent/modules/tts_synthesizer.py`
4. Add language name to `SUPPORTED_LANGS` in `agent/modules/mos_evaluator.py`
5. Update `config/agent_config.yaml` supported_languages list
6. Add tests for the new language

## Testing Guidelines

- Write unit tests for all new functions
- Use `AsyncMock` for mocking async LLM calls
- Use `MagicMock` for mocking HuggingFace models
- Create temporary files for file I/O tests
- Test both success and failure paths

### Example Test

```python
@pytest.mark.asyncio
async def test_translate_spanish(mock_llm):
    from agent.modules.script_translator import ScriptTranslator
    from agent.modules.asr_transcriber import TranscriptResult, Segment

    t = ScriptTranslator(llm_client=mock_llm)
    transcript = TranscriptResult(
        segments=[Segment(id=0, start=0.0, end=1.0, text="Hello")],
        language="en",
        duration_seconds=1.0,
    )

    result = await t.translate(transcript, target_language="es")
    assert result.target_language == "es"
    assert len(result.segments) == 1
```

## Documentation Updates

- Update `PROJECT-detail.md` for architectural changes
- Update `SECOND-KNOWLEDGE-BRAIN.md` for new research findings
- Update `README.md` for user-facing changes
- Add docstrings to all public functions

## Issues and Bug Reports

When reporting issues, please include:

- Python version
- Operating system
- Steps to reproduce
- Expected vs actual behavior
- Relevant logs

## Feature Requests

For feature requests:

- Check if similar issue exists
- Describe the use case
- Propose API/interface if applicable
- Consider if it fits project scope

## Code Review Process

1. All PRs must pass tests
2. At least one maintainer approval required
3. Auto-merge after approval if all checks pass
4. Squash commits for clean history

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

## Questions?

- Open an issue for questions
- Check existing documentation first
- Be patient with volunteer maintainers

Thank you for contributing!

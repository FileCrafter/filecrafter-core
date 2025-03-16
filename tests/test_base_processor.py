import pytest
from pathlib import Path
from ..src.filecrafter.core.processors.base import BaseProcessor, ProcessingResult

class DummyProcessor(BaseProcessor):
    """Test implementation of BaseProcessor."""
    
    async def process(self, file_path: Path) -> ProcessingResult:
        return ProcessingResult(success=True)
    
    def get_supported_formats(self) -> list[str]:
        return ['.txt']

@pytest.fixture
def processor():
    return DummyProcessor()

def test_processor_initialization():
    config = {'test': 'value'}
    processor = DummyProcessor(config=config)
    assert processor.config == config

def test_validate_file(processor):
    valid_file = Path('test.txt')
    invalid_file = Path('test.jpg')
    
    assert processor.validate_file(valid_file) is True
    assert processor.validate_file(invalid_file) is False
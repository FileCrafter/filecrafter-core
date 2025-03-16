from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Optional
from pydantic import BaseModel

class ProcessingError(Exception):
    """Base exception for processing errors."""
    pass

class ProcessingResult(BaseModel):
    """Standard result format for all processors."""
    success: bool
    error_message: Optional[str] = None
    output_path: Optional[str] = None
    metadata: Dict[str, Any] = {}

class BaseProcessor(ABC):
    """Base class for all file processors."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
    
    @abstractmethod
    async def process(self, file_path: Path) -> ProcessingResult:
        """
        Process a file and return the result.
        
        Args:
            file_path: Path to the file to process
            
        Returns:
            ProcessingResult containing success status and metadata
            
        Raises:
            ProcessingError: If processing fails
        """
        pass
    
    @abstractmethod
    def get_supported_formats(self) -> list[str]:
        """Return list of supported file extensions (e.g., ['.jpg', '.png'])."""
        pass
    
    def validate_file(self, file_path: Path) -> bool:
        """
        Validate if the file can be processed by this processor.
        
        Args:
            file_path: Path to the file to validate
            
        Returns:
            bool: True if file is valid for this processor
        """
        return file_path.suffix.lower() in self.get_supported_formats()
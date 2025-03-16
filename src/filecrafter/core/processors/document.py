from pathlib import Path
from typing import Optional, Dict, List, Tuple, Any, Union, Literal
from dataclasses import dataclass
from pydantic import BaseModel, Field, validator
from enum import Enum

# PDF Processing
from PyPDF2 import PdfReader, PdfWriter, Transformation
from PyPDF2.generic import PdfObject, RectangleObject, ArrayObject, NumberObject
import fitz  # PyMuPDF for advanced compression

# Image Processing
import io
from PIL import Image, ImageDraw, ImageFont

# Base Classes and Types
from .base import BaseProcessor, ProcessingResult, ProcessingError

# Optional type hints for better code completion
PdfPage = Any  # PyPDF2 page type
FitzPage = Any  # PyMuPDF page type

class CompressionLevel(str, Enum):
    LOW = 'low'
    MEDIUM = 'medium'
    HIGH = 'high'

class ScaleType(str, Enum):
    PROPORTIONAL = 'proportional'
    FIT = 'fit'

class CompressConfig(BaseModel):
    compression_level: CompressionLevel = Field(
        default=CompressionLevel.MEDIUM,
        description="Compression level for PDF optimization"
    )

class FlattenConfig(BaseModel):
    flatten_forms: bool = Field(
        default=True,
        description="Whether to flatten form fields"
    )
    flatten_annotations: bool = Field(
        default=True,
        description="Whether to flatten annotations"
    )

class ScaleConfig(BaseModel):
    scale_type: ScaleType = Field(
        default=ScaleType.PROPORTIONAL,
        description="Type of scaling to apply"
    )
    target_width: Optional[float] = Field(
        default=None,
        description="Target width in points"
    )
    target_height: Optional[float] = Field(
        default=None,
        description="Target height in points"
    )
    scale_factor: float = Field(
        default=1.0,
        ge=0.1,
        le=10.0,
        description="Uniform scale factor"
    )

    @validator('target_width', 'target_height')
    def validate_dimensions(cls, v):
        if v is not None and v <= 0:
            raise ValueError("Dimensions must be positive")
        return v

class ExtractPagesConfig(BaseModel):
    pages: List[int] = Field(
        ...,
        description="Page numbers to extract (0-based)"
    )

    @validator('pages')
    def validate_pages(cls, v):
        if not v:
            raise ValueError("At least one page number must be specified")
        if any(p < 0 for p in v):
            raise ValueError("Page numbers must be non-negative")
        return v

class MergeConfig(BaseModel):
    input_files: List[Path] = Field(
        ...,
        description="List of PDF files to merge"
    )

    @validator('input_files')
    def validate_input_files(cls, v):
        if len(v) < 2:
            raise ValueError("At least two PDF files are required for merging")
        for path in v:
            if not path.exists():
                raise ValueError(f"File not found: {path}")
            if path.suffix.lower() != '.pdf':
                raise ValueError(f"Not a PDF file: {path}")
        return v

class SplitConfig(BaseModel):
    split_size: int = Field(
        default=1,
        ge=1,
        description="Number of pages per split file"
    )

class MetadataConfig(BaseModel):
    metadata: Dict[str, str] = Field(
        ...,
        description="Metadata key-value pairs to set"
    )

class EncryptConfig(BaseModel):
    user_password: Optional[str] = Field(
        default=None,
        description="Password for opening the PDF"
    )
    owner_password: Optional[str] = Field(
        default=None,
        description="Password for full permissions"
    )

    @validator('user_password', 'owner_password')
    def validate_passwords(cls, v, values):
        if 'user_password' in values and not values['user_password'] and not v:
            raise ValueError("At least one password must be specified")
        return v

class RotateConfig(BaseModel):
    rotation: int = Field(
        default=90,
        description="Rotation angle in degrees"
    )
    pages: Optional[List[int]] = Field(
        default=None,
        description="Specific pages to rotate (None means all pages)"
    )

    @validator('rotation')
    def validate_rotation(cls, v):
        if v % 90 != 0:
            raise ValueError("Rotation must be a multiple of 90 degrees")
        return v

class RemovePagesConfig(BaseModel):
    pages: List[int] = Field(
        ...,
        description="Page numbers to remove (0-based)"
    )

    @validator('pages')
    def validate_pages(cls, v):
        if not v:
            raise ValueError("At least one page number must be specified")
        if any(p < 0 for p in v):
            raise ValueError("Page numbers must be non-negative")
        return v

class ReorderConfig(BaseModel):
    page_order: List[int] = Field(
        ...,
        description="New order of pages (0-based)"
    )

    @validator('page_order')
    def validate_page_order(cls, v):
        if not v:
            raise ValueError("Page order must not be empty")
        if any(p < 0 for p in v):
            raise ValueError("Page numbers must be non-negative")
        if len(set(v)) != len(v):
            raise ValueError("Page numbers must not be repeated")
        return v

class DocumentProcessorConfig(BaseModel):
    operation: Literal[
        'optimize',
        'extract_text',
        'extract_pages',
        'merge',
        'split',
        'set_metadata',
        'encrypt',
        'rotate',
        'compress',
        'remove_pages',
        'reorder',
        'flatten',
        'scale'
    ] = Field(
        ...,
        description="Operation to perform on the PDF"
    )
    compress: Optional[CompressConfig] = None
    flatten: Optional[FlattenConfig] = None
    scale: Optional[ScaleConfig] = None
    extract_pages: Optional[ExtractPagesConfig] = None
    merge: Optional[MergeConfig] = None
    split: Optional[SplitConfig] = None
    metadata: Optional[MetadataConfig] = None
    encrypt: Optional[EncryptConfig] = None
    rotate: Optional[RotateConfig] = None
    remove_pages: Optional[RemovePagesConfig] = None
    reorder: Optional[ReorderConfig] = None

    @validator('*')
    def validate_operation_config(cls, v, values):
        if 'operation' in values:
            operation = values['operation']
            field_name = v.__class__.__name__.lower().replace('config', '')
            if operation == field_name and v is None:
                raise ValueError(f"Configuration required for operation '{operation}'")
        return v

class DocumentProcessingError(ProcessingError):
    """Specific exception for document processing errors."""
    pass

class DocumentProcessor(BaseProcessor):
    """Handles document processing operations, primarily PDFs."""
    
    SUPPORTED_FORMATS = ['.pdf']
    COMPRESSION_SETTINGS = {
        CompressionLevel.LOW: {'image_quality': 85, 'dpi': 150},
        CompressionLevel.MEDIUM: {'image_quality': 60, 'dpi': 120},
        CompressionLevel.HIGH: {'image_quality': 30, 'dpi': 72}
    }

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        try:
            self.validated_config = DocumentProcessorConfig(**(config or {}))
        except Exception as e:
            raise DocumentProcessingError(f"Invalid configuration: {str(e)}")

    def _compress_image(self, image: Image.Image, quality: int, max_dpi: int) -> Image.Image:
        """Compress an image with specified quality and DPI."""
        # Calculate new size based on DPI
        original_dpi = image.info.get('dpi', (300, 300))
        if original_dpi[0] > max_dpi:
            scale_factor = max_dpi / original_dpi[0]
            new_size = (
                int(image.width * scale_factor),
                int(image.height * scale_factor)
            )
            image = image.resize(new_size, Image.Resampling.LANCZOS)
        
        # Convert to RGB if RGBA
        if image.mode == 'RGBA':
            background = Image.new('RGB', image.size, (255, 255, 255))
            background.paste(image, mask=image.split()[3])
            image = background
        
        # Compress
        buffer = io.BytesIO()
        image.save(buffer, format='JPEG', quality=quality, optimize=True)
        return Image.open(buffer)

    def _get_page_dimensions(self, page) -> Tuple[float, float]:
        """Get page dimensions in points."""
        mediabox = page.mediabox
        return (
            float(mediabox.width),
            float(mediabox.height)
        )

    def _calculate_scale_factors(
        self,
        page_width: float,
        page_height: float,
        target_width: float,
        target_height: float,
        scale_type: str
    ) -> Tuple[float, float]:
        """Calculate scaling factors based on target dimensions and scale type."""
        if scale_type == 'proportional':
            scale_factor = min(
                target_width / page_width,
                target_height / page_height
            )
            return scale_factor, scale_factor
        else:  # 'fit'
            return (
                target_width / page_width,
                target_height / page_height
            )

    async def _advanced_compress_pdf(self, reader: PdfReader, compression_level: CompressionLevel) -> PdfWriter:
        """Implement advanced PDF compression using PyMuPDF."""
        try:
            writer = PdfWriter()
            settings = self.COMPRESSION_SETTINGS[compression_level]
            
            # Open with PyMuPDF for advanced compression
            pdf_bytes = io.BytesIO()
            writer_temp = PdfWriter()
            for page in reader.pages:
                writer_temp.add_page(page)
            writer_temp.write(pdf_bytes)
            pdf_bytes.seek(0)
            
            try:
                doc = fitz.open(stream=pdf_bytes, filetype="pdf")
            except Exception as e:
                raise DocumentProcessingError(f"Failed to open PDF with PyMuPDF: {str(e)}")
            
            for page_num, page in enumerate(doc):
                try:
                    # Process images
                    for img_index, img in enumerate(page.get_images()):
                        try:
                            xref = img[0]
                            base_image = doc.extract_image(xref)
                            image_data = base_image["image"]
                            
                            image = Image.open(io.BytesIO(image_data))
                            compressed_image = self._compress_image(
                                image,
                                settings['image_quality'],
                                settings['dpi']
                            )
                            
                            img_bytes = io.BytesIO()
                            compressed_image.save(img_bytes, format='JPEG')
                            page.replace_image(xref, stream=img_bytes.getvalue())
                        except Exception as e:
                            raise DocumentProcessingError(
                                f"Failed to process image {img_index} on page {page_num + 1}: {str(e)}"
                            )
                    
                    page.clean_contents()
                    page.set_compression(1)
                except Exception as e:
                    raise DocumentProcessingError(f"Failed to process page {page_num + 1}: {str(e)}")
            
            pdf_bytes = io.BytesIO()
            doc.save(pdf_bytes, garbage=4, clean=True, deflate=True)
            pdf_bytes.seek(0)
            
            compressed_reader = PdfReader(pdf_bytes)
            for page in compressed_reader.pages:
                writer.add_page(page)
            
            return writer
            
        except DocumentProcessingError:
            raise
        except Exception as e:
            raise DocumentProcessingError(f"PDF compression failed: {str(e)}")

    async def _flatten_form_fields(self, page) -> None:
        """Flatten form fields on a page."""
        if '/Annots' not in page:
            return
        
        annotations = page['/Annots']
        if annotations is None:
            return
            
        flattened_annotations = []
        
        for annotation in annotations:
            if annotation.get('/Subtype') == '/Widget':  # Form field
                if '/V' in annotation:  # If field has a value
                    # Create appearance stream if it doesn't exist
                    if '/AP' not in annotation:
                        # Create basic appearance stream based on value
                        value = annotation['/V']
                        rect = annotation['/Rect']
                        page.merge_page(self._create_text_layer(
                            str(value),
                            RectangleObject(rect)
                        ))
            else:
                flattened_annotations.append(annotation)
        
        # Update annotations
        page['/Annots'] = flattened_annotations

    async def _flatten_annotations(self, page) -> None:
        """Flatten annotations on a page."""
        if '/Annots' not in page:
            return
            
        annotations = page['/Annots']
        if annotations is None:
            return
        
        for annotation in annotations:
            if annotation.get('/Subtype') in ['/Highlight', '/Underline', '/StrikeOut']:
                # Convert annotation to direct appearance
                rect = annotation['/Rect']
                color = annotation.get('/C', [0, 0, 0])
                
                # Create appearance stream
                appearance = self._create_annotation_appearance(
                    annotation['/Subtype'],
                    RectangleObject(rect),
                    color
                )
                page.merge_page(appearance)
        
        # Remove original annotations
        page['/Annots'] = []

    async def _scale_page(self, page, scale_x: float, scale_y: float) -> None:
        """Scale page content."""
        # Get original page dimensions
        orig_width, orig_height = self._get_page_dimensions(page)
        
        # Create transformation matrix
        transform = Transformation().scale(scale_x, scale_y)
        
        # Apply transformation to page
        page.add_transformation(transform)
        
        # Update page dimensions
        new_width = orig_width * scale_x
        new_height = orig_height * scale_y
        page.mediabox = RectangleObject([0, 0, new_width, new_height])
        page.cropbox = RectangleObject([0, 0, new_width, new_height])

    async def process(self, file_path: Path) -> ProcessingResult:
        """Process a document file based on configuration."""
        try:
            if not self.validate_file(file_path):
                return ProcessingResult(
                    success=False,
                    error_message=f"Unsupported file format: {file_path.suffix}"
                )

            if not file_path.exists():
                return ProcessingResult(
                    success=False,
                    error_message=f"File not found: {file_path}"
                )

            operation = self.validated_config.operation
            
            try:
                reader = PdfReader(file_path)
            except Exception as e:
                return ProcessingResult(
                    success=False,
                    error_message=f"Failed to read PDF: {str(e)}"
                )

            writer = PdfWriter()
            
            try:
                if operation == 'compress':
                    config = self.validated_config.compress or CompressConfig()
                    writer = await self._advanced_compress_pdf(reader, config.compression_level)
                    output_path = file_path.with_stem(f"{file_path.stem}_compressed")

                elif operation == 'flatten':
                    config = self.validated_config.flatten or FlattenConfig()
                    for page in reader.pages:
                        if config.flatten_forms:
                            await self._flatten_form_fields(page)
                        if config.flatten_annotations:
                            await self._flatten_annotations(page)
                        writer.add_page(page)
                    output_path = file_path.with_stem(f"{file_path.stem}_flattened")

                elif operation == 'scale':
                    config = self.validated_config.scale or ScaleConfig()
                    for page in reader.pages:
                        if config.target_width and config.target_height:
                            page_width, page_height = self._get_page_dimensions(page)
                            scale_x, scale_y = self._calculate_scale_factors(
                                page_width, page_height,
                                config.target_width, config.target_height,
                                config.scale_type
                            )
                        else:
                            scale_x = scale_y = config.scale_factor
                        
                        await self._scale_page(page, scale_x, scale_y)
                        writer.add_page(page)
                    output_path = file_path.with_stem(f"{file_path.stem}_scaled")

                else:
                    return ProcessingResult(
                        success=False,
                        error_message=f"Unknown operation: {operation}"
                    )

                # Write the output file
                try:
                    with open(output_path, 'wb') as output_file:
                        writer.write(output_file)
                except Exception as e:
                    return ProcessingResult(
                        success=False,
                        error_message=f"Failed to write output file: {str(e)}"
                    )

                return ProcessingResult(
                    success=True,
                    output_path=str(output_path),
                    metadata={
                        'page_count': len(reader.pages),
                        'original_size': file_path.stat().st_size,
                        'processed_size': output_path.stat().st_size
                    }
                )

            except DocumentProcessingError as e:
                return ProcessingResult(
                    success=False,
                    error_message=str(e)
                )
            except Exception as e:
                return ProcessingResult(
                    success=False,
                    error_message=f"Processing failed: {str(e)}"
                )

        except Exception as e:
            return ProcessingResult(
                success=False,
                error_message=f"Unexpected error: {str(e)}"
            )
    
    def get_supported_formats(self) -> list[str]:
        """Return list of supported document formats."""
        return self.SUPPORTED_FORMATS
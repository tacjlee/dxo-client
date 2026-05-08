"""
Identification utilities for workflow documents.

Provides content-hash based identification for screens and other
workflow entities. Supports multiple input types: .md, .json, .xls, .xlsx
"""

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union


@dataclass
class ScreenIdentity:
    """Identity information for a screen document."""
    id: str  # Content hash (unique identifier)
    screen_name: str  # Original filename without extension
    mode: Optional[str] = None  # CREATE, EDIT, VIEW, DELETE
    content_hash: str = ""  # Full content hash for change detection

    def __post_init__(self):
        if not self.content_hash:
            self.content_hash = self.id


def generate_content_hash(content: Union[str, bytes], length: int = 32) -> str:
    """
    Generate a SHA256 hash of content, truncated to specified length.

    Args:
        content: String or bytes content to hash
        length: Number of hex characters to return (default 32)

    Returns:
        Truncated hex hash string
    """
    if isinstance(content, str):
        content = content.encode('utf-8')
    return hashlib.sha256(content).hexdigest()[:length]


def generate_content_id(content: Union[str, bytes], prefix: str = "") -> str:
    """
    Generate a content-based ID with optional prefix.

    Args:
        content: String or bytes content to hash
        prefix: Optional prefix to prepend (e.g., "SC005_")

    Returns:
        ID string in format "{prefix}{hash}" or just "{hash}"
    """
    content_hash = generate_content_hash(content)
    if prefix:
        return f"{prefix}_{content_hash}" if not prefix.endswith("_") else f"{prefix}{content_hash}"
    return content_hash


def extract_mode(text: str) -> Optional[str]:
    """
    Extract operation mode from text.

    Args:
        text: Text containing potential mode indicator

    Returns:
        Mode string (CREATE, EDIT, VIEW, DELETE) or None
    """
    text_upper = text.upper()
    if "CREATE" in text_upper or "THÊM MỚI" in text_upper or "ADD" in text_upper:
        return "CREATE"
    if "EDIT" in text_upper or "SỬA" in text_upper or "UPDATE" in text_upper:
        return "EDIT"
    if "DELETE" in text_upper or "XÓA" in text_upper:
        return "DELETE"
    if "VIEW" in text_upper or "XEM" in text_upper or "LIST" in text_upper or "DANH SÁCH" in text_upper:
        return "VIEW"
    return None


def read_file_content(file_path: Union[str, Path]) -> bytes:
    """
    Read file content based on file type.

    Supports: .md, .json, .xls, .xlsx

    Args:
        file_path: Path to the file

    Returns:
        File content as bytes

    Raises:
        ValueError: If file type is not supported
        FileNotFoundError: If file doesn't exist
    """
    path = Path(file_path)
    suffix = path.suffix.lower()

    if suffix in ('.md', '.json', '.txt'):
        return path.read_bytes()

    if suffix in ('.xls', '.xlsx'):
        # For Excel files, read the binary content
        return path.read_bytes()

    raise ValueError(f"Unsupported file type: {suffix}. Supported: .md, .json, .txt, .xls, .xlsx")


def normalize_content_for_hash(content: bytes, file_type: str) -> bytes:
    """
    Normalize content before hashing for consistency across formats.

    For text formats, strips whitespace and normalizes line endings.
    For binary formats (Excel), uses raw bytes.

    Args:
        content: Raw file content
        file_type: File extension (e.g., '.md', '.xlsx')

    Returns:
        Normalized content for hashing
    """
    file_type = file_type.lower()

    if file_type in ('.md', '.json', '.txt'):
        # Normalize text: strip, normalize line endings
        text = content.decode('utf-8', errors='replace')
        text = text.strip()
        text = text.replace('\r\n', '\n').replace('\r', '\n')
        return text.encode('utf-8')

    # For binary files, use as-is
    return content


def generate_screen_identity(
    file_path: Optional[Union[str, Path]] = None,
    content: Optional[Union[str, bytes]] = None,
    screen_name: Optional[str] = None,
) -> ScreenIdentity:
    """
    Generate screen identity from file or content.

    Args:
        file_path: Path to the source file
        content: Content to hash (if file_path not provided)
        screen_name: Override screen name (defaults to filename without ext)

    Returns:
        ScreenIdentity with id, screen_name, mode

    Raises:
        ValueError: If neither file_path nor content is provided
    """
    if file_path:
        path = Path(file_path)
        file_content = read_file_content(path)
        normalized_content = normalize_content_for_hash(file_content, path.suffix)

        if screen_name is None:
            screen_name = path.stem  # filename without extension

        # Try to extract text for mode detection
        try:
            text_for_analysis = file_content.decode('utf-8', errors='replace')
        except:
            text_for_analysis = screen_name

    elif content:
        if isinstance(content, str):
            normalized_content = content.strip().encode('utf-8')
            text_for_analysis = content
        else:
            normalized_content = content
            try:
                text_for_analysis = content.decode('utf-8', errors='replace')
            except:
                text_for_analysis = ""

        if screen_name is None:
            screen_name = "unknown"

    else:
        raise ValueError("Either file_path or content must be provided")

    # Generate hash
    content_hash = generate_content_hash(normalized_content)

    # Extract mode
    mode = extract_mode(text_for_analysis) or extract_mode(screen_name)

    return ScreenIdentity(
        id=content_hash,
        screen_name=screen_name,
        mode=mode,
        content_hash=content_hash,
    )

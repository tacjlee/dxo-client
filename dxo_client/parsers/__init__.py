"""
Parsers for converting various file formats to structured data.
"""

from .golden_parser import (
    parse_golden_file,
    GoldenFileData,
)

__all__ = [
    "parse_golden_file",
    "GoldenFileData",
]

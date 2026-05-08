"""
Test extraction API functionality via dxo-client.
"""

import pytest
from unittest.mock import MagicMock, patch

from dxo_client import (
    KnowledgeClient,
    ExtractionResult,
    SupportedFormats,
    KnowledgeValidationError,
    KnowledgeAPIError,
)


class TestExtractionAPI:
    """Test text extraction API methods."""

    def test_extract_text_success(self):
        """Test successful text extraction."""
        with patch.object(KnowledgeClient, '_get_client') as mock_get_client:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "content": "This is extracted text from the document.",
                "file_type": "pdf",
                "char_count": 42,
                "filename": "test.pdf"
            }

            mock_client = MagicMock()
            mock_client.post.return_value = mock_response
            mock_get_client.return_value = mock_client

            client = KnowledgeClient(base_url="http://test:8000")
            result = client.extract_text(b"fake pdf content", "test.pdf")

            assert isinstance(result, ExtractionResult)
            assert result.content == "This is extracted text from the document."
            assert result.file_type == "pdf"
            assert result.char_count == 42
            assert result.filename == "test.pdf"

    def test_extract_text_unsupported_format(self):
        """Test extraction with unsupported file format."""
        with patch.object(KnowledgeClient, '_get_client') as mock_get_client:
            mock_response = MagicMock()
            mock_response.status_code = 422
            mock_response.text = "Unsupported file format: .xyz"

            mock_client = MagicMock()
            mock_client.post.return_value = mock_response
            mock_get_client.return_value = mock_client

            client = KnowledgeClient(base_url="http://test:8000")

            with pytest.raises(KnowledgeValidationError):
                client.extract_text(b"some content", "test.xyz")

    def test_get_supported_formats(self):
        """Test getting supported file formats."""
        with patch.object(KnowledgeClient, '_make_request') as mock_request:
            mock_request.return_value = {
                "extensions": [".txt", ".md", ".json", ".pdf", ".docx", ".xlsx", ".xls", ".pptx", ".html", ".htm"]
            }

            client = KnowledgeClient(base_url="http://test:8000")
            result = client.get_supported_formats()

            assert isinstance(result, SupportedFormats)
            assert ".pdf" in result.extensions
            assert ".docx" in result.extensions
            assert ".txt" in result.extensions

    def test_is_format_supported_true(self):
        """Test checking if a format is supported (true case)."""
        with patch.object(KnowledgeClient, '_make_request') as mock_request:
            mock_request.return_value = {
                "filename": "document.pdf",
                "supported": True,
                "supported_extensions": [".pdf", ".docx", ".txt"]
            }

            client = KnowledgeClient(base_url="http://test:8000")
            result = client.is_format_supported("document.pdf")

            assert result is True

    def test_is_format_supported_false(self):
        """Test checking if a format is supported (false case)."""
        with patch.object(KnowledgeClient, '_make_request') as mock_request:
            mock_request.return_value = {
                "filename": "document.xyz",
                "supported": False,
                "supported_extensions": [".pdf", ".docx", ".txt"]
            }

            client = KnowledgeClient(base_url="http://test:8000")
            result = client.is_format_supported("document.xyz")

            assert result is False


class TestExtractionModels:
    """Test extraction-related models."""

    def test_extraction_result_model(self):
        """Test ExtractionResult model."""
        result = ExtractionResult(
            content="Hello World",
            file_type="txt",
            char_count=11,
            filename="test.txt"
        )

        assert result.content == "Hello World"
        assert result.file_type == "txt"
        assert result.char_count == 11
        assert result.filename == "test.txt"

    def test_supported_formats_model(self):
        """Test SupportedFormats model."""
        formats = SupportedFormats(
            extensions=[".txt", ".pdf", ".docx"]
        )

        assert len(formats.extensions) == 3
        assert ".txt" in formats.extensions
        assert ".pdf" in formats.extensions

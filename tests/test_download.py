"""
Test module for document downloader functionality
"""

import os
import sys
from pathlib import Path
import tempfile
import unittest
from unittest import mock
import pathlib

import requests
import responses

project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))
import group4py
from download import download_pdf, _is_pdf_file


class TestDocumentDownloader(unittest.TestCase):
    """Test cases for document downloader module"""

    def setUp(self):
        """Set up test fixtures"""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.output_dir = self.temp_dir.name
        
        # Valid test URLs for different document types
        self.pdf_url = "https://example.com/document.pdf"
        self.doc_url = "https://example.com/document.doc"
        self.docx_url = "https://example.com/document.docx"
        
        # URL without extension
        self.no_ext_url = "https://example.com/document"
        
        # Sample PDF header
        self.pdf_header = b"%PDF-1.5"
        
        # Create mock content for various file types
        self.pdf_content = self.pdf_header + b"\nSample PDF content"
        self.doc_content = b"Sample DOC content"
        self.docx_content = b"Sample DOCX content"

    def tearDown(self):
        """Clean up after tests"""
        self.temp_dir.cleanup()

    @responses.activate
    def test_download_pdf_file(self):
        """Test downloading a PDF file"""
        # Mock the HTTP response
        responses.add(
            responses.GET,
            self.pdf_url,
            body=self.pdf_content,
            status=200,
            content_type="application/pdf"
        )
        
        # Download the document
        result = download_pdf(self.pdf_url, self.output_dir)
        
        # Check the result
        self.assertIsNotNone(result)
        self.assertTrue(os.path.exists(result))
        self.assertTrue(result.endswith(".pdf"))
        
        # Verify content
        with open(result, "rb") as f:
            content = f.read()
            self.assertEqual(content, self.pdf_content)

    @responses.activate
    def test_download_doc_file(self):
        """Test downloading a DOC file"""
        # Mock the HTTP response
        responses.add(
            responses.GET,
            self.doc_url,
            body=self.doc_content,
            status=200,
            content_type="application/msword"
        )
        
        # Download the document
        result = download_pdf(self.doc_url, self.output_dir)
        
        # Check the result
        self.assertIsNotNone(result)
        self.assertTrue(os.path.exists(result))
        self.assertTrue(result.endswith(".doc"))
        
        # Verify content
        with open(result, "rb") as f:
            content = f.read()
            self.assertEqual(content, self.doc_content)

    @responses.activate
    def test_download_docx_file(self):
        """Test downloading a DOCX file"""
        # Mock the HTTP response
        responses.add(
            responses.GET,
            self.docx_url,
            body=self.docx_content,
            status=200,
            content_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        )
        
        # Download the document
        result = download_pdf(self.docx_url, self.output_dir)
        
        # Check the result
        self.assertIsNotNone(result)
        self.assertTrue(os.path.exists(result))
        self.assertTrue(result.endswith(".docx"))
        
        # Verify content
        with open(result, "rb") as f:
            content = f.read()
            self.assertEqual(content, self.docx_content)

    @responses.activate
    def test_download_with_no_extension(self):
        """Test downloading a file with no extension"""
        # Mock the HTTP response
        responses.add(
            responses.GET,
            self.no_ext_url,
            body=self.pdf_content,
            status=200,
            content_type="application/pdf"
        )
        
        # Download the document
        result = download_pdf(self.no_ext_url, self.output_dir)
        
        # Check the result
        self.assertIsNotNone(result)
        self.assertTrue(os.path.exists(result))
        self.assertTrue(result.endswith(".pdf"))
        
        # Verify content
        with open(result, "rb") as f:
            content = f.read()
            self.assertEqual(content, self.pdf_content)

    @responses.activate
    def test_download_with_incorrect_content_type(self):
        """Test downloading a file with incorrect content type but valid extension"""
        # Mock the HTTP response with wrong content type
        responses.add(
            responses.GET,
            self.pdf_url,
            body=self.pdf_content,
            status=200,
            content_type="text/plain"  # Wrong content type
        )
        
        # Download the document (should succeed due to .pdf extension)
        result = download_pdf(self.pdf_url, self.output_dir)
        
        # Check the result
        self.assertIsNotNone(result)
        self.assertTrue(os.path.exists(result))

    @responses.activate
    def test_download_with_force_download(self):
        """Test downloading a file with force_download=True"""
        # Mock the HTTP response with wrong content type
        responses.add(
            responses.GET,
            self.no_ext_url,
            body=self.pdf_content,
            status=200,
            content_type="text/plain"  # Wrong content type
        )
        
        # Download the document with force_download=True
        result = download_pdf(self.no_ext_url, self.output_dir, force_download=True)
        
        # Check the result
        self.assertIsNotNone(result)
        self.assertTrue(os.path.exists(result))

    @responses.activate
    def test_download_with_retry(self):
        """Test downloading a file with retry mechanism"""
        # Add a failing response first
        responses.add(
            responses.GET,
            self.pdf_url,
            status=500
        )
        
        # Add a successful response that will be used after retry
        responses.add(
            responses.GET,
            self.pdf_url,
            body=self.pdf_content,
            status=200,
            content_type="application/pdf"
        )
        
        # Download the document
        result = download_pdf(self.pdf_url, self.output_dir)
        
        # Check the result
        self.assertIsNotNone(result)
        self.assertTrue(os.path.exists(result))

    def test_invalid_url(self):
        """Test downloading with invalid URL"""
        # Try with an invalid URL
        result = download_pdf("not-a-url", self.output_dir)
        
        # Should return None
        self.assertIsNone(result)

    @responses.activate
    def test_empty_file(self):
        """Test downloading an empty file"""
        # Mock the HTTP response with empty content
        responses.add(
            responses.GET,
            self.pdf_url,
            body=b"",
            status=200,
            content_type="application/pdf"
        )
        
        # Download the document
        result = download_pdf(self.pdf_url, self.output_dir)
        
        # Should return None for empty file
        self.assertIsNone(result)

    def test_is_pdf_file(self):
        """Test PDF validation function"""
        # Create a temporary PDF file
        temp_pdf = os.path.join(self.output_dir, "test.pdf")
        with open(temp_pdf, "wb") as f:
            f.write(self.pdf_header)
        
        # Test the function
        self.assertTrue(_is_pdf_file(temp_pdf))
        
        # Create a non-PDF file
        temp_non_pdf = os.path.join(self.output_dir, "test.txt")
        with open(temp_non_pdf, "wb") as f:
            f.write(b"Not a PDF file")
        
        # Test the function
        self.assertFalse(_is_pdf_file(temp_non_pdf))

    @responses.activate
    def test_backward_compatibility(self):
        """Test backward compatibility with download_pdf function"""
        # Mock the HTTP response
        responses.add(
            responses.GET,
            self.pdf_url,
            body=self.pdf_content,
            status=200,
            content_type="application/pdf"
        )
        
        # Download using the backward compatibility function
        result = download_pdf(self.pdf_url, self.output_dir)
        
        # Check the result
        self.assertIsNotNone(result)
        self.assertTrue(os.path.exists(result))
        self.assertTrue(result.endswith(".pdf"))


if __name__ == "__main__":
    unittest.main()

import sys
import pytest
from pathlib import Path
from typing import List
from pprint import pprint

project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))
import group4py
from group4py.src.scrape import Detector, DocUpdater
from group4py.src.schema import NDCDocumentBase

@pytest.fixture
def detector():
    """Fixture that returns a Detector instance."""
    return Detector()

@pytest.fixture
def doc_updater():
    """Fixture that returns a DocUpdater instance."""
    return DocUpdater()

@pytest.fixture
def containers(detector):
    """Fixture that returns the extracted containers."""
    containers = detector.extract_all_containers()
    assert containers, "No containers were extracted"
    return containers

@pytest.fixture
def pdf_links(detector, containers):
    """Fixture that returns extracted PDF links."""
    links = detector.extract_all_pdf_links(containers)
    assert links, "No PDF links were extracted"
    return links

def test_extract_all_containers(detector):
    """Test that containers are extracted correctly."""
    containers = detector.extract_all_containers()
    assert containers is not None, "Containers should not be None"
    assert len(containers) > 0, "Should extract at least one container"
    print(f"Extracted {len(containers)} containers")
    assert True

def test_extract_all_pdf_links(detector, containers):
    """Test that PDF links are extracted correctly."""
    links = detector.extract_all_pdf_links(containers)
    assert links is not None, "PDF links should not be None"
    assert len(links) > 0, "Should extract at least one PDF link"
    print(f"Extracted {len(links)} PDF links")
    
    # URLs might be relative paths or full URLs, so we check for a common pattern
    assert all('/sites/default/files/' in link for link in links), "All links should contain /sites/default/files/"
    assert True

def test_check_new_links(detector):
    """Test the link comparison functionality."""
    old_links = ["/sites/default/files/NDC/2022-06/old_doc1.pdf", 
                "/sites/default/files/NDC/2022-10/common_doc.pdf"]
    new_links = ["/sites/default/files/NDC/2022-10/common_doc.pdf", 
                "/sites/default/files/NDC/2022-06/new_doc1.pdf"]
    
    result = detector.check_new_links(new_links, old_links)
    
    assert len(result['new']) == 1, "Should have 1 new link"
    assert len(result['old']) == 1, "Should have 1 old link"
    assert len(result['current']) == 1, "Should have 1 current link"
    assert result['new'][0] == "/sites/default/files/NDC/2022-06/new_doc1.pdf"
    assert result['old'][0] == "/sites/default/files/NDC/2022-06/old_doc1.pdf"
    assert result['current'][0] == "/sites/default/files/NDC/2022-10/common_doc.pdf"

def test_extract_metadata_from_containers(doc_updater, containers):
    """Test that metadata is extracted and parsed correctly."""
    documents = doc_updater.extract_metadata_from_containers(containers)
    assert documents is not None, "Documents should not be None"
    assert len(documents) > 0, "Should extract at least one document"
    
    # Sample a few fields to ensure data structure is as expected
    for doc in documents:
        # Check basic attributes that should be present
        assert hasattr(doc, 'country'), "Document should have country attribute"
        assert hasattr(doc, 'url'), "Document should have url attribute"
        
        # Check that URL contains the expected pattern
        assert '/sites/default/files/' in doc.url, f"URL should contain /sites/default/files/ but got {doc.url}"
    
    print(f"Extracted metadata for {len(documents)} documents")
    assert True

if __name__ == "__main__":
    """Run tests if script is executed directly."""
    detector = Detector()
    doc_updater = DocUpdater()
    
    print("Testing container extraction...")
    test_extract_all_containers(detector)
    
    print("\nTesting PDF link extraction...")
    containers = detector.extract_all_containers()
    test_extract_all_pdf_links(detector, containers)
    
    print("\nTesting link comparison...")
    test_check_new_links(detector)
    
    print("\nTesting metadata extraction...")
    test_extract_metadata_from_containers(doc_updater, containers)
    
    documents = doc_updater.extract_metadata_from_containers(containers)
    if documents:
        print("\nSample document metadata:")
        document = documents[0]
        print(f"Country: {document.country}")
        print(f"Title: {document.title}")
        print(f"URL: {document.url}")
        print(f"Language: {document.language}")
        print(f"Submission date: {document.submission_date}")
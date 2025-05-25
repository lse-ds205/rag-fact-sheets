"""
Debug script to examine PDF extraction issues
"""
import sys
from pathlib import Path
import logging

project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

from group4py.src.extract_document import extract_text_from_pdf, _has_character_corruption
from unstructured.partition.pdf import partition_pdf

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def debug_pdf_extraction(pdf_path: str):
    """
    Debug PDF extraction to understand why elements are being filtered out
    """
    print(f"\n=== Debugging PDF Extraction: {pdf_path} ===")
    
    try:
        # Direct unstructured partition
        print("\n1. Raw unstructured partition_pdf results:")
        elements = partition_pdf(filename=pdf_path, strategy="fast")
        print(f"   Total elements extracted: {len(elements)}")
        
        # Examine first 10 elements
        print("\n2. Sample of first 10 elements:")
        for i, element in enumerate(elements[:10]):
            text = str(element).strip()
            print(f"   Element {i}:")
            print(f"     Type: {type(element).__name__}")
            print(f"     Text length: {len(text)}")
            print(f"     Text preview: {repr(text[:100])}")
            print(f"     Has corruption: {_has_character_corruption(text)}")
            print(f"     Has metadata: {hasattr(element, 'metadata')}")
            if hasattr(element, 'metadata') and element.metadata:
                try:
                    if hasattr(element.metadata, '__dict__'):
                        print(f"     Metadata: {element.metadata.__dict__}")
                    else:
                        print(f"     Metadata: {element.metadata}")
                except:
                    print(f"     Metadata: <error accessing metadata>")
            print()
        
        # Check filtering criteria
        print("\n3. Filtering analysis:")
        empty_count = 0
        short_count = 0
        corruption_count = 0
        artifact_count = 0
        valid_count = 0
        
        import re
        skip_patterns = [
            r'^\d+$',  # Just page numbers
            r'^[ivxlcdm]+$',  # Roman numerals only
            r'^[\s\-_=\.]+$',  # Just punctuation/whitespace
            r'^(page|pg)\s*\d+$',  # Page indicators
        ]
        
        for i, element in enumerate(elements):
            text = str(element).strip()
            
            if not text:
                empty_count += 1
                continue
                
            if len(text) < 10:
                short_count += 1
                if i < 5:  # Show first few short texts
                    print(f"     Short text {i}: {repr(text)}")
                continue
                
            if _has_character_corruption(text):
                corruption_count += 1
                if corruption_count <= 3:  # Show first few corrupted texts
                    print(f"     Corrupted text {i}: {text[:100]}...")
                continue
                
            if any(re.match(pattern, text.lower()) for pattern in skip_patterns):
                artifact_count += 1
                if artifact_count <= 3:  # Show first few artifacts
                    print(f"     Artifact {i}: {repr(text)}")
                continue
                
            valid_count += 1
            if valid_count <= 3:  # Show first few valid texts
                print(f"     Valid text {i}: {text[:100]}...")
        
        print(f"\n   Filtering results:")
        print(f"     Empty elements: {empty_count}")
        print(f"     Too short (< 10 chars): {short_count}")
        print(f"     Corrupted elements: {corruption_count}")
        print(f"     Artifacts (page numbers, etc.): {artifact_count}")
        print(f"     Valid elements: {valid_count}")
        
        # Test with our extraction function
        print(f"\n4. Testing extract_text_from_pdf function:")
        extracted_elements = extract_text_from_pdf(pdf_path, strategy="fast")
        print(f"   Function returned: {len(extracted_elements)} elements")
        
        if extracted_elements:
            print(f"   Sample extracted element:")
            sample = extracted_elements[0]
            print(f"     Text: {sample.get('text', '')[:100]}...")
            print(f"     Metadata: {sample.get('metadata', {})}")
        
    except Exception as e:
        print(f"Error during debugging: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # Test with the problematic China PDF
    china_pdf = project_root / "data" / "pdfs" / "china_english_20220601.pdf"
    
    if china_pdf.exists():
        debug_pdf_extraction(str(china_pdf))
    else:
        print(f"PDF not found: {china_pdf}")
        
        # List available PDFs
        pdf_dir = project_root / "data" / "pdfs"
        if pdf_dir.exists():
            pdfs = list(pdf_dir.glob("*.pdf"))
            print(f"Available PDFs: {[p.name for p in pdfs[:5]]}")
            if pdfs:
                debug_pdf_extraction(str(pdfs[0]))

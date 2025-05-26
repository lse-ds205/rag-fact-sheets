import json
import os
import sys
from pathlib import Path
from datetime import datetime
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

# Get project root and add to path
project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

# Constants
OUTPUT_DIR = project_root / "outputs" / "factsheets"
DATA_DIR = project_root / "data"
FONT_NAME = "Helvetica"
FONT_SIZE_TITLE = 16
FONT_SIZE_HEADING = 14
FONT_SIZE_BODY = 12
LINE_SPACING = 0.5

def ensure_output_directory():
    """Create output directory if it doesn't exist."""
    if not OUTPUT_DIR.exists():
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def load_llm_response_data():
    """Load and parse LLM response data from data folder."""
    json_path = DATA_DIR / "llm_response.json"
    
    if not json_path.exists():
        raise FileNotFoundError(f"LLM response file not found: {json_path}")
    
    with open(json_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def create_pdf_style():
    """Create and return ReportLab styles."""
    styles = getSampleStyleSheet()
    
    # Add custom styles
    styles.add(ParagraphStyle(
        name='CustomTitle',
        parent=styles['Heading1'],
        fontSize=FONT_SIZE_TITLE,
        spaceAfter=30,
        alignment=1  # Center alignment
    ))
    
    styles.add(ParagraphStyle(
        name='CustomHeading',
        parent=styles['Heading2'],
        fontSize=FONT_SIZE_HEADING,
        spaceAfter=12,
        spaceBefore=20
    ))
    
    styles.add(ParagraphStyle(
        name='CustomBody',
        parent=styles['Normal'],
        fontSize=FONT_SIZE_BODY,
        spaceAfter=12
    ))
    
    return styles

def create_metadata_table(response_data, styles):
    """Create a table for metadata information from LLM response."""
    # Extract metadata from LLM response structure
    question = response_data.get('question', 'N/A')
    citations_count = len(response_data.get('citations', []))
    
    # Get countries from citations
    countries = set()
    for citation in response_data.get('citations', []):
        if citation.get('country'):
            countries.add(citation['country'])
    
    countries_str = ', '.join(sorted(countries)) if countries else 'N/A'
    
    data = [
        ['Question', question[:100] + '...' if len(question) > 100 else question],
        ['Countries Analyzed', countries_str],
        ['Citations Count', str(citations_count)],
        ['Generated', datetime.now().strftime('%Y-%m-%d %H:%M:%S')]
    ]
    
    table = Table(data, colWidths=[2*inch, 4*inch])
    table.setStyle(TableStyle([
        ('FONTNAME', (0, 0), (-1, -1), FONT_NAME),
        ('FONTSIZE', (0, 0), (-1, -1), FONT_SIZE_BODY),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
        ('BACKGROUND', (0, 0), (0, -1), colors.lightgrey),
        ('TEXTCOLOR', (0, 0), (0, -1), colors.black),
        ('ALIGN', (0, 0), (0, -1), 'RIGHT'),
        ('ALIGN', (1, 0), (1, -1), 'LEFT'),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
    ]))
    return table

def create_answer_section(response_data, styles):
    """Create answer section content from LLM response."""
    elements = []
    
    # Add Answer heading
    elements.append(Paragraph("ANALYSIS", styles['CustomHeading']))
    elements.append(Spacer(1, 12))
    
    # Get answer content
    answer = response_data.get('answer', {})
    
    if isinstance(answer, dict):
        # Structured answer format
        summary = answer.get('summary', '')
        detailed = answer.get('detailed_response', '')
        
        if summary:
            elements.append(Paragraph("<b>Summary:</b>", styles['CustomBody']))
            elements.append(Spacer(1, 6))
            elements.append(Paragraph(summary, styles['CustomBody']))
            elements.append(Spacer(1, 12))
        
        if detailed:
            elements.append(Paragraph("<b>Detailed Analysis:</b>", styles['CustomBody']))
            elements.append(Spacer(1, 6))
            elements.append(Paragraph(detailed, styles['CustomBody']))
            elements.append(Spacer(1, 12))
    else:
        # Simple string answer
        elements.append(Paragraph(str(answer), styles['CustomBody']))
        elements.append(Spacer(1, 12))
    
    return elements

def create_citations_section(response_data, styles):
    """Create citations section from LLM response."""
    elements = []
    
    citations = response_data.get('citations', [])
    if not citations:
        return elements
    
    # Add Citations heading
    elements.append(Paragraph("SOURCES & CITATIONS", styles['CustomHeading']))
    elements.append(Spacer(1, 12))
    
    # Group citations by country
    citations_by_country = {}
    for citation in citations:
        country = citation.get('country', 'Unknown')
        if country not in citations_by_country:
            citations_by_country[country] = []
        citations_by_country[country].append(citation)
    
    # Add citations for each country
    for country, country_citations in citations_by_country.items():
        elements.append(Paragraph(f"<b>{country}:</b>", styles['CustomBody']))
        elements.append(Spacer(1, 6))
        
        for i, citation in enumerate(country_citations, 1):
            content = citation.get('content', 'No content available')
            how_used = citation.get('how_used', 'Referenced in analysis')
            similarity = citation.get('cos_similarity_score', 0.0)
            
            citation_text = f"[{i}] {content[:200]}{'...' if len(content) > 200 else ''}"
            elements.append(Paragraph(citation_text, styles['CustomBody']))
            
            if how_used:
                elements.append(Paragraph(f"<i>Used for: {how_used}</i>", styles['CustomBody']))
            
            elements.append(Paragraph(f"<i>Relevance Score: {similarity:.3f}</i>", styles['CustomBody']))
            elements.append(Spacer(1, 8))
        
        elements.append(Spacer(1, 12))
    
    return elements

def generate_pdf_report(response_data, output_path, styles):
    """Generate a PDF report from LLM response data."""
    doc = SimpleDocTemplate(
        str(output_path),  # Convert Path object to string for ReportLab
        pagesize=letter,
        rightMargin=72,
        leftMargin=72,
        topMargin=72,
        bottomMargin=72
    )
    
    elements = []
    
    # Add title
    question = response_data.get('question', 'Climate Policy Analysis')
    title = "Climate Policy Fact Sheet"
    elements.append(Paragraph(title, styles['CustomTitle']))
    elements.append(Spacer(1, 20))
    
    # Add metadata section
    elements.append(Paragraph("METADATA", styles['CustomHeading']))
    elements.append(Spacer(1, 12))
    elements.append(create_metadata_table(response_data, styles))
    elements.append(Spacer(1, 20))
    
    # Add answer section
    elements.extend(create_answer_section(response_data, styles))
    
    # Add citations section
    elements.extend(create_citations_section(response_data, styles))
    
    # Build PDF
    doc.build(elements)

def main():
    """Main function to orchestrate the PDF generation process."""
    try:
        print("[6_OUTPUT] Starting PDF generation from LLM response...")
        
        # Ensure output directory exists
        ensure_output_directory()
        
        # Load LLM response data
        print(f"[6_OUTPUT] Loading LLM response from: {DATA_DIR / 'llm_response.json'}")
        response_data = load_llm_response_data()
        
        # Create styles
        styles = create_pdf_style()
        
        # Generate output filename
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_filename = f"climate_policy_factsheet_{timestamp}.pdf"
        output_path = OUTPUT_DIR / output_filename
        
        # Generate PDF
        print(f"[6_OUTPUT] Generating PDF report...")
        generate_pdf_report(response_data, output_path, styles)
        
        print(f"[6_OUTPUT] PDF report generated successfully: {output_path}")
        print(f"[6_OUTPUT] Report contains analysis based on {len(response_data.get('citations', []))} citations")
        
        # Print summary
        countries = set()
        for citation in response_data.get('citations', []):
            if citation.get('country'):
                countries.add(citation['country'])
        
        if countries:
            print(f"[6_OUTPUT] Countries analyzed: {', '.join(sorted(countries))}")
        
        return str(output_path)
        
    except FileNotFoundError as e:
        print(f"[6_OUTPUT] Error: {e}")
        print(f"[6_OUTPUT] Please run the LLM response generation script first (5_llm_response.py)")
        return None
    except Exception as e:
        print(f"[6_OUTPUT] Error generating PDF: {e}")
        return None

if __name__ == "__main__":
    main()
import json
import os
import sys
import glob
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
DATA_DIR = project_root / "data" / "llm"  # Updated to data/llm directory
FONT_NAME = "Helvetica"
FONT_SIZE_TITLE = 16
FONT_SIZE_HEADING = 14
FONT_SIZE_BODY = 12
LINE_SPACING = 0.5

# Color scheme
COLORS = {
    'title': colors.HexColor('#2C3E50'),      # Dark blue for title
    'heading': colors.HexColor('#2980B9'),    # Medium blue for headings
    'subheading': colors.HexColor('#3498DB'), # Light blue for subheadings
    'metadata_bg': colors.HexColor('#ECF0F1'), # Light gray for metadata background
    'metadata_header': colors.HexColor('#BDC3C7'), # Darker gray for metadata header
    'summary_bg': colors.HexColor('#E8F6F3'),  # Light teal for summary background
    'detail_bg': colors.HexColor('#EBF5FB'),   # Light blue for detailed background
    'citation_bg': colors.HexColor('#FEF9E7'),  # Light yellow for citations
    'question_bg': colors.HexColor('#F4F6F7'),  # Light gray for question background
    'border': colors.HexColor('#D6DBDF'),      # Border color
}

def ensure_output_directory():
    """Create output directory if it doesn't exist."""
    if not OUTPUT_DIR.exists():
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def get_available_llm_files():
    """Get list of all available LLM response JSON files in the data/llm directory."""
    file_pattern = str(DATA_DIR / "*.json")
    return glob.glob(file_pattern)

def load_llm_response_data(json_path):
    """Load and parse LLM response data from the specified JSON file."""
    if not Path(json_path).exists():
        raise FileNotFoundError(f"LLM response file not found: {json_path}")
    
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Extract country name from metadata if available
    country_name = data.get('metadata', {}).get('country_name', 'Unknown')
    
    return country_name, data

def create_pdf_style():
    """Create and return ReportLab styles with enhanced colors."""
    styles = getSampleStyleSheet()
    
    # Add custom styles with colors
    styles.add(ParagraphStyle(
        name='CustomTitle',
        parent=styles['Heading1'],
        fontSize=FONT_SIZE_TITLE,
        spaceAfter=30,
        alignment=1,  # Center alignment
        textColor=COLORS['title'],
        borderWidth=1,
        borderColor=COLORS['border'],
        borderPadding=10,
        backColor=colors.white
    ))
    
    styles.add(ParagraphStyle(
        name='CustomHeading',
        parent=styles['Heading2'],
        fontSize=FONT_SIZE_HEADING,
        spaceAfter=12,
        spaceBefore=20,
        textColor=COLORS['heading'],
        borderWidth=0,
        borderRadius=5,
        borderPadding=6,
        backColor=COLORS['question_bg']
    ))
    
    styles.add(ParagraphStyle(
        name='CustomBody',
        parent=styles['Normal'],
        fontSize=FONT_SIZE_BODY,
        spaceAfter=12,
        textColor=colors.black
    ))
    
    # Add more specific styles
    styles.add(ParagraphStyle(
        name='SummaryStyle',
        parent=styles['CustomBody'],
        backColor=COLORS['summary_bg'],
        borderPadding=10,
        borderRadius=5,
        borderWidth=1,
        borderColor=COLORS['border']
    ))
    
    styles.add(ParagraphStyle(
        name='DetailStyle',
        parent=styles['CustomBody'],
        backColor=COLORS['detail_bg'],
        borderPadding=10,
        borderRadius=5,
        borderWidth=1,
        borderColor=COLORS['border']
    ))
    
    styles.add(ParagraphStyle(
        name='CitationStyle',
        parent=styles['CustomBody'],
        backColor=COLORS['citation_bg'],
        borderPadding=6,
        borderRadius=3,
        borderWidth=1,
        borderColor=COLORS['border'],
        fontSize=FONT_SIZE_BODY-1
    ))
    
    return styles

def create_metadata_table(response_data, styles, country_name):
    """Create a table for metadata information from LLM response with enhanced colors."""
    # Get the first question's data to extract common metadata
    first_question_key = next(iter(response_data.get('questions', {})), None)
    question_data = response_data.get('questions', {}).get(first_question_key, {})
    llm_response = question_data.get('llm_response', {})
    
    # Extract metadata
    question = llm_response.get('question', 'N/A')
    citations_count = len(llm_response.get('citations', []))
    
    # Get countries from citations
    countries = set([country_name])  # Include the main country
    for citation in llm_response.get('citations', []):
        if citation.get('country'):
            countries.add(citation['country'])
    
    countries_str = ', '.join(sorted(countries)) if countries else country_name
    
    # Get timestamp from metadata or current time
    timestamp = response_data.get('metadata', {}).get('timestamp', 
                                                    datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    
    data = [
        ['Country', country_name],
        ['Question Count', str(len(response_data.get('questions', {})))],
        ['Citations Count', str(citations_count)],
        ['Generated', timestamp]
    ]
    
    # Update table styling with colors
    table = Table(data, colWidths=[2*inch, 4*inch])
    table.setStyle(TableStyle([
        ('FONTNAME', (0, 0), (-1, -1), FONT_NAME),
        ('FONTSIZE', (0, 0), (-1, -1), FONT_SIZE_BODY),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
        ('BACKGROUND', (0, 0), (0, -1), COLORS['metadata_header']),
        ('BACKGROUND', (1, 0), (1, -1), COLORS['metadata_bg']),
        ('TEXTCOLOR', (0, 0), (0, -1), colors.black),
        ('ALIGN', (0, 0), (0, -1), 'RIGHT'),
        ('ALIGN', (1, 0), (1, -1), 'LEFT'),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('GRID', (0, 0), (-1, -1), 0.5, COLORS['border']),
        ('ROUNDEDCORNERS', [5, 5, 5, 5]),
    ]))
    return table

def create_answer_section(response_data, styles):
    """Create answer section content from LLM response with enhanced colors."""
    elements = []
    
    # Process each question in the data
    for question_num, question_data in sorted(response_data.get('questions', {}).items()):
        llm_response = question_data.get('llm_response', {})
        question_text = llm_response.get('question', f"Question {question_num}")
        
        # Extract question number safely from question_num
        if isinstance(question_num, str):
            # Handle formats like 'question_1' or just '1'
            if '_' in question_num:
                display_num = question_num.split('_')[-1]
            else:
                display_num = question_num
        else:
            display_num = str(question_num)
        
        # Add question heading with background color
        elements.append(Paragraph(f"QUESTION {display_num}: {question_text}", styles['CustomHeading']))
        elements.append(Spacer(1, 12))
        
        # Get answer content with error handling for malformed structures
        answer = llm_response.get('answer', {})
        
        try:
            if isinstance(answer, dict):
                # Check if this is a malformed JSON string structure
                summary = answer.get('summary', '')
                detailed = answer.get('detailed_response', '')
                
                # Handle case where summary/detailed contain JSON strings
                if isinstance(summary, str) and summary.strip().startswith('{'):
                    try:
                        # Try to parse and extract the actual content
                        import json
                        parsed_summary = json.loads(summary)
                        if isinstance(parsed_summary, dict) and 'answer' in parsed_summary:
                            summary = parsed_summary['answer'].get('summary', summary)
                    except (json.JSONDecodeError, KeyError):
                        # If parsing fails, use the raw string but clean it up
                        summary = summary.replace('\\n', ' ').replace('\\"', '"')
                
                if isinstance(detailed, str) and detailed.strip().startswith('{'):
                    try:
                        # Try to parse and extract the actual content
                        import json
                        parsed_detailed = json.loads(detailed)
                        if isinstance(parsed_detailed, dict) and 'answer' in parsed_detailed:
                            detailed = parsed_detailed['answer'].get('detailed_response', detailed)
                    except (json.JSONDecodeError, KeyError):
                        # If parsing fails, use the raw string but clean it up
                        detailed = detailed.replace('\\n', ' ').replace('\\"', '"')
                
                # Structured answer format
                if summary and summary != '{}':
                    elements.append(Paragraph("<b>Summary:</b>", styles['CustomBody']))
                    elements.append(Spacer(1, 6))
                    # Clean up any remaining JSON artifacts
                    clean_summary = str(summary).replace('\\n', '\n').replace('\\"', '"')
                    elements.append(Paragraph(clean_summary, styles['SummaryStyle']))
                    elements.append(Spacer(1, 12))
                
                if detailed and detailed != '{}':
                    elements.append(Paragraph("<b>Detailed Analysis:</b>", styles['CustomBody']))
                    elements.append(Spacer(1, 6))
                    # Clean up any remaining JSON artifacts
                    clean_detailed = str(detailed).replace('\\n', '\n').replace('\\"', '"')
                    elements.append(Paragraph(clean_detailed, styles['DetailStyle']))
                    elements.append(Spacer(1, 12))
                
                # If neither summary nor detailed content is available, show a message
                if (not summary or summary == '{}') and (not detailed or detailed == '{}'):
                    elements.append(Paragraph("Content not available in readable format.", styles['DetailStyle']))
                    elements.append(Spacer(1, 12))
            else:
                # Simple string answer
                clean_answer = str(answer).replace('\\n', '\n').replace('\\"', '"')
                elements.append(Paragraph(clean_answer, styles['DetailStyle']))
                elements.append(Spacer(1, 12))
        
        except Exception as e:
            # Fallback for any processing errors
            elements.append(Paragraph(f"Error processing answer content: {str(e)}", styles['DetailStyle']))
            elements.append(Spacer(1, 12))
        
        # Add spacer between questions
        elements.append(Spacer(1, 20))
    
    return elements

def create_citations_section(response_data, styles):
    """Create citations section from LLM response with enhanced colors."""
    elements = []
    
    # Collect all citations from all questions
    all_citations = []
    for question_data in response_data.get('questions', {}).values():
        llm_response = question_data.get('llm_response', {})
        all_citations.extend(llm_response.get('citations', []))
    
    if not all_citations:
        return elements
    
    # Add Citations heading
    elements.append(Paragraph("SOURCES & CITATIONS", styles['CustomHeading']))
    elements.append(Spacer(1, 12))
    
    # Group citations by country
    citations_by_country = {}
    for citation in all_citations:
        country = citation.get('country', 'Unknown')
        if country not in citations_by_country:
            citations_by_country[country] = []
        citations_by_country[country].append(citation)
    
    # Add citations for each country
    for country, country_citations in citations_by_country.items():
        elements.append(Paragraph(f"<b>{country}:</b>", styles['CustomBody']))
        elements.append(Spacer(1, 6))
        
        # Create a set to track unique citations by content
        unique_contents = set()
        unique_citations = []
        
        # Filter to unique citations only
        for citation in country_citations:
            content = citation.get('content', '')
            if content and content not in unique_contents:
                unique_contents.add(content)
                unique_citations.append(citation)
        
        for i, citation in enumerate(unique_citations, 1):
            content = citation.get('content', 'No content available')
            how_used = citation.get('how_used', 'Referenced in analysis')
            similarity = citation.get('cos_similarity_score', 0.0)
            
            citation_text = f"[{i}] {content[:200]}{'...' if len(content) > 200 else ''}"
            elements.append(Paragraph(citation_text, styles['CitationStyle']))
            
            if how_used:
                elements.append(Paragraph(f"<i>Used for: {how_used}</i>", styles['CustomBody']))
            
            if similarity:
                elements.append(Paragraph(f"<i>Relevance Score: {similarity:.3f}</i>", styles['CustomBody']))
            
            elements.append(Spacer(1, 8))
        
        elements.append(Spacer(1, 12))
    
    return elements

def generate_pdf_report(response_data, output_path, styles, country_name):
    """Generate a PDF report from LLM response data."""
    doc = SimpleDocTemplate(
        str(output_path),
        pagesize=letter,
        rightMargin=72,
        leftMargin=72,
        topMargin=72,
        bottomMargin=72
    )
    
    elements = []
    
    # Add title with country name
    title = f"{country_name} Climate Policy Fact Sheet"
    elements.append(Paragraph(title, styles['CustomTitle']))
    elements.append(Spacer(1, 20))
    
    # Add metadata section
    elements.append(Paragraph("METADATA", styles['CustomHeading']))
    elements.append(Spacer(1, 12))
    elements.append(create_metadata_table(response_data, styles, country_name))
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
        print("[6_OUTPUT] Starting PDF generation from LLM response files...")
        
        # Ensure output directory exists
        ensure_output_directory()
        
        # Get list of all LLM response files
        llm_files = get_available_llm_files()
        
        if not llm_files:
            print(f"[6_OUTPUT] No LLM response files found in {DATA_DIR}")
            return None
        
        print(f"[6_OUTPUT] Found {len(llm_files)} LLM response files")
        
        # Create styles
        styles = create_pdf_style()
        
        # Process each file
        for json_path in llm_files:
            try:
                print(f"[6_OUTPUT] Processing: {json_path}")
                
                # Load data and extract country name
                country_name, response_data = load_llm_response_data(json_path)
                
                # Generate output filename with country name
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                output_filename = f"{country_name}_climate_policy_factsheet_{timestamp}.pdf"
                output_path = OUTPUT_DIR / output_filename
                
                # Generate PDF
                print(f"[6_OUTPUT] Generating PDF report for {country_name}...")
                generate_pdf_report(response_data, output_path, styles, country_name)
                
                print(f"[6_OUTPUT] PDF report generated: {output_path}")
                print(f"[6_OUTPUT] Report contains analysis of {len(response_data.get('questions', {}))} questions for {country_name}")
                
            except Exception as e:
                print(f"[6_OUTPUT] Error processing {json_path}: {e}")
        
        return str(OUTPUT_DIR)
        
    except Exception as e:
        print(f"[6_OUTPUT] Error in PDF generation process: {e}")
        return None

if __name__ == "__main__":
    main()
import json
import os
from datetime import datetime
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

# Constants
OUTPUT_DIR = "outputs/factsheets"
FONT_NAME = "Helvetica"
FONT_SIZE_TITLE = 16
FONT_SIZE_HEADING = 14
FONT_SIZE_BODY = 12
LINE_SPACING = 0.5

def ensure_output_directory():
    """Create output directory if it doesn't exist."""
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

def load_json_data(json_path):
    """Load and parse JSON data from file."""
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

def create_metadata_table(metadata, styles):
    """Create a table for metadata information."""
    data = [
        ['Country', metadata.get('country', 'N/A')],
        ['Submission Date', metadata.get('submission_date', 'N/A')],
        ['Target Year', metadata.get('target_year', 'N/A')]
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
    ]))
    return table

def create_qa_section(qa_data, styles):
    """Create Q&A section content."""
    elements = []
    
    # Add Q&A heading
    elements.append(Paragraph("REPORT", styles['CustomHeading']))
    elements.append(Spacer(1, 12))
    
    # Add each Q&A pair
    for q, a in qa_data.items():
        elements.append(Paragraph(f"<b>{q}</b>", styles['CustomBody']))
        elements.append(Spacer(1, 6))
        elements.append(Paragraph(a, styles['CustomBody']))
        elements.append(Spacer(1, 12))
    
    return elements

def generate_pdf_report(entry, output_path, styles):
    """Generate a PDF report for a single entry."""
    doc = SimpleDocTemplate(
        output_path,
        pagesize=letter,
        rightMargin=72,
        leftMargin=72,
        topMargin=72,
        bottomMargin=72
    )
    
    elements = []
    
    # Add title
    elements.append(Paragraph("NDC Fact Sheet", styles['CustomTitle']))
    elements.append(Spacer(1, 20))
    
    # Add metadata section
    elements.append(Paragraph("METADATA", styles['CustomHeading']))
    elements.append(Spacer(1, 12))
    elements.append(create_metadata_table(entry['metadata'], styles))
    elements.append(Spacer(1, 20))
    
    # Add Q&A section
    elements.extend(create_qa_section(entry['qa'], styles))
    
    # Build PDF
    doc.build(elements)

def process_json_data(json_data):
    """Process JSON data and generate PDF reports."""
    styles = create_pdf_style()
    
    for entry in json_data:
        # TODO: Update if JSON schema changes: adjust key path to match actual metadata structure
        country = entry['metadata']['country']
        output_path = os.path.join(OUTPUT_DIR, f"{country}_factsheet.pdf")
        
        generate_pdf_report(entry, output_path, styles)
        print(f"Generated PDF report for {country}")

def main():
    """Main function to orchestrate the PDF generation process."""
    # TODO: Update path to match actual JSON file location
    json_path = "path/to/your/qa_data.json"
    
    ensure_output_directory()
    json_data = load_json_data(json_path)
    process_json_data(json_data)

if __name__ == "__main__":
    main()
import sys
from pathlib import Path
import logging
import csv
import io
from datetime import datetime
from typing import List, Dict, Any, Optional, Union

project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root))
import group4py
from constants.styles import EMAIL_CSS_STYLE
from constants.questions import DEFAULT_QUESTIONS

logger = logging.getLogger(__name__)


class HTMLFormatter:
    """
    HTML formatter class to generate PDF-style HTML documents from JSON data
    """
    
    @staticmethod
    def format_to_html(
        data: Optional[Dict[str, Any]] = None,
        country: str = "Unknown Country",
        submission_date: str = "Not specified",
        target_years: Union[List[str], str] = "2030",
        questions_answers: Optional[Dict[str, str]] = None
    ) -> str:
        """
        Generate a PDF-style HTML document from JSON data
        
        Args:
            data: Optional JSON data containing all information
            country: Country name
            submission_date: Submission date
            target_years: Target year(s) as string or list
            questions_answers: Dictionary of questions and answers
            
        Returns:
            str: Formatted HTML document
        """
        
        # Extract data from JSON if provided
        if data:
            country = data.get('country', country)
            submission_date = data.get('submission_date', submission_date)
            target_years = data.get('target_years', target_years)
            questions_answers = data.get('questions_answers', questions_answers)
        
        # Use default questions if none provided
        if not questions_answers:
            questions_answers = DEFAULT_QUESTIONS.copy()
            logger.info("Using default question set for NDC report")
        
        # Format target years
        if isinstance(target_years, str):
            target_years_list = [target_years]
        elif isinstance(target_years, list):
            target_years_list = target_years
        else:
            target_years_list = ["2030"]
        
        # Generate target year badges
        target_year_badges = ''.join([
            f'<span class="target-year">{year}</span>' 
            for year in target_years_list
        ])
        
        # Generate Q&A section
        qa_html = ""
        for i, (question, answer) in enumerate(questions_answers.items(), 1):
            # Clean and format the answer
            if not answer or answer.strip() == "":
                answer_content = '<div class="no-data">No data available</div>'
            else:
                # Split answer into paragraphs if it contains line breaks
                paragraphs = answer.split('\n')
                answer_content = ''.join([f'<p>{para.strip()}</p>' for para in paragraphs if para.strip()])
                if not answer_content:
                    answer_content = f'<p>{answer}</p>'
            
            qa_html += f"""
            <div class="qa-container">
                <div class="question">
                    Q{i}: {question}
                </div>
                <div class="answer">
                    {answer_content}
                </div>
            </div>
            """
        
        # Current timestamp for footer

        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Build complete HTML document
        html_content = f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>TPI X DS205 NDC Report - {country}</title>
            {EMAIL_CSS_STYLE}
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>TPI X DS205 NDC REPORT</h1>
                    <div class="subtitle">Nationally Determined Contributions Analysis</div>
                </div>
                
                <div class="metadata">
                    <div class="metadata-grid">
                        <div class="metadata-item">
                            <div class="metadata-label">Country</div>
                            <div class="metadata-value">{country}</div>
                        </div>
                        <div class="metadata-item">
                            <div class="metadata-label">Submission Date</div>
                            <div class="metadata-value">{submission_date}</div>
                        </div>
                        <div class="metadata-item">
                            <div class="metadata-label">Target Year(s)</div>
                            <div class="metadata-value">
                                <div class="target-years">
                                    {target_year_badges}
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
                
                <div class="content">
                    <h2 class="section-title">Questions & Answers</h2>
                    {qa_html}
                </div>
                
                <div class="footer">
                    Generated on {current_time} | TPI X DS205 NDC Analysis System
                </div>
            </div>
        </body>
        </html>
        """
        
        logger.info(f"Generated HTML report for {country} with {len(questions_answers)} Q&A pairs")
        return html_content
    
    @staticmethod
    def save_html_to_file(html_content: str, filename: str = None) -> str:
        """
        Save HTML content to file
        
        Args:
            html_content: HTML content to save
            filename: Optional filename, will generate one if not provided
            
        Returns:
            str: Path to saved file
        """
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"ndc_report_{timestamp}.html"
        
        # Ensure .html extension
        if not filename.endswith('.html'):
            filename += '.html'
        
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(html_content)
            
            logger.info(f"HTML report saved to: {filename}")
            return filename
            
        except Exception as e:
            logger = logging.getLogger(__name__)
            logger.error(f"Error saving HTML file: {e}")
            raise
    
    @staticmethod
    def generate_sample_report() -> str:
        """
        Generate a sample report using default data
        
        Returns:
            str: Sample HTML report
        """
        sample_data = {
            "country": "Singapore",
            "submission_date": "2022-06-01",
            "target_years": ["2030", "2050"],
            "questions_answers": {
                "What does the country promise as their 2030/2035 NDC target?": "Singapore commits to reduce emissions intensity by 36% from 2005 levels by 2030, and achieve net-zero emissions by 2050.",
                "What years are these countries using as their baseline?": "2005 is used as the baseline year for emissions intensity calculations.",
                "Are they reporting a business as usual (BAU) target rather than a base year target?": "No, Singapore is reporting against a base year target using 2005 as the baseline.",
                "What sectors are covered by the target?": "All sectors including Energy, Industrial Processes, Agriculture, and Waste. LULUCF is reported separately.",
                "What greenhouse gasses are covered by the target?": "All greenhouse gases covered under the Kyoto Protocol: CO2, CH4, N2O, HFCs, PFCs, SF6, and NF3."
            }
        }
        
        return HTMLFormatter.format_to_html(data=sample_data)


class CSVFormatter:
    """
    CSV formatter class to generate structured CSV reports from JSON data
    """
    
    @staticmethod
    def format_to_csv(
        data: Optional[Dict[str, Any]] = None,
        country: str = "Unknown Country",
        submission_date: str = "Not specified",
        target_years: Union[List[str], str] = "2030",
        questions_answers: Optional[Dict[str, str]] = None,
        include_headers: bool = True
    ) -> str:
        """
        Generate a CSV document from JSON data
        
        Args:
            data: Optional JSON data containing all information
            country: Country name
            submission_date: Submission date
            target_years: Target year(s) as string or list
            questions_answers: Dictionary of questions and answers
            include_headers: Whether to include CSV headers
            
        Returns:
            str: CSV formatted content
        """
        
        # Extract data from JSON if provided
        if data:
            country = data.get('country', country)
            submission_date = data.get('submission_date', submission_date)
            target_years = data.get('target_years', target_years)
            questions_answers = data.get('questions_answers', questions_answers)
        
        # Use default questions if none provided
        if not questions_answers:
            questions_answers = DEFAULT_QUESTIONS.copy()
            logger.info("Using default question set for CSV report")
        
        # Format target years as comma-separated string
        if isinstance(target_years, list):
            target_years_str = ", ".join(target_years)
        else:
            target_years_str = str(target_years)
        
        # Create CSV content using StringIO
        output = io.StringIO()
        writer = csv.writer(output, quoting=csv.QUOTE_ALL)
        
        # Write headers if requested
        if include_headers:
            headers = [
                "Question",
                "Answer", 
                "Page Number",
                "Paragraph Number",
                "Country",
                "Submission Date",
                "Target Years"
            ]
            writer.writerow(headers)
        
        # Write data rows
        for question, answer in questions_answers.items():
            # Clean answer text - remove newlines and extra whitespace
            cleaned_answer = answer.replace('\n', ' ').replace('\r', ' ').strip()
            if not cleaned_answer:
                cleaned_answer = "No data available"
            
            # Extract page and paragraph numbers from metadata if available
            # These would typically come from the chunk data
            page_number = data.get('page_number', '') if data else ''
            paragraph_number = data.get('paragraph_number', '') if data else ''
            
            row = [
                question,
                cleaned_answer,
                page_number,
                paragraph_number,
                country,
                submission_date,
                target_years_str
            ]
            writer.writerow(row)
        
        csv_content = output.getvalue()
        output.close()
        
        logger.info(f"Generated CSV report for {country} with {len(questions_answers)} Q&A pairs")
        return csv_content
    
    @staticmethod
    def format_hoprag_results_to_csv(
        hoprag_results: Dict[str, Any],
        country: str = "Unknown Country",
        submission_date: str = "Not specified",
        target_years: Union[List[str], str] = "2030",
        include_headers: bool = True
    ) -> str:
        """
        Convert HopRAG classification results to CSV format with chunk information
        
        Args:
            hoprag_results: Results from HopRAGClassifier.classify_nodes()
            country: Country name
            submission_date: Submission date
            target_years: Target year(s) as string or list
            include_headers: Whether to include CSV headers
            
        Returns:
            str: CSV formatted content with chunk analysis
        """
        
        # Format target years as comma-separated string
        if isinstance(target_years, list):
            target_years_str = ", ".join(target_years)
        else:
            target_years_str = str(target_years)
        
        # Create CSV content using StringIO
        output = io.StringIO()
        writer = csv.writer(output, quoting=csv.QUOTE_ALL)
        
        # Write headers if requested
        if include_headers:
            headers = [
                "Question",
                "Answer",
                "Page Number",
                "Paragraph Number", 
                "Country",
                "Submission Date",
                "Target Years",
                "Chunk Classification",
                "Centrality Score",
                "Confidence Level"
            ]
            writer.writerow(headers)
        
        # Extract chunk information from HopRAG results
        category_details = hoprag_results.get('category_details', {})
        
        question_counter = 1
        for category, details in category_details.items():
            nodes = details.get('nodes', [])
            
            for node in nodes:
                # Use chunk content as the "answer"
                content = node.get('content', 'No content available')
                cleaned_content = content.replace('\n', ' ').replace('\r', ' ').strip()
                
                # Generate a question based on the chunk classification
                question = f"Q{question_counter}: What information does this {category.lower()} chunk provide?"
                
                # Extract metadata if available (would come from database chunk data)
                page_number = ''
                paragraph_number = ''
                
                row = [
                    question,
                    cleaned_content,
                    page_number,
                    paragraph_number,
                    country,
                    submission_date,
                    target_years_str,
                    category,
                    round(node.get('combined_score', 0), 4),
                    node.get('confidence_level', 'UNKNOWN')
                ]
                writer.writerow(row)
                question_counter += 1
        
        csv_content = output.getvalue()
        output.close()
        
        logger.info(f"Generated HopRAG CSV report for {country} with {question_counter-1} chunks")
        return csv_content
    
    @staticmethod
    def save_csv_to_file(csv_content: str, filename: str = None) -> str:
        """
        Save CSV content to file
        
        Args:
            csv_content: CSV content to save
            filename: Optional filename, will generate one if not provided
            
        Returns:
            str: Path to saved file
        """
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"ndc_report_{timestamp}.csv"
        
        # Ensure .csv extension
        if not filename.endswith('.csv'):
            filename += '.csv'
        
        try:
            with open(filename, 'w', encoding='utf-8', newline='') as f:
                f.write(csv_content)
            
            logger = logging.getLogger(__name__)
            logger.info(f"CSV report saved to: {filename}")
            return filename
            
        except Exception as e:
            logger = logging.getLogger(__name__)
            logger.error(f"Error saving CSV file: {e}")
            raise
    
    @staticmethod
    def generate_sample_csv() -> str:
        """
        Generate a sample CSV using default data
        
        Returns:
            str: Sample CSV content
        """
        sample_data = {
            "country": "Singapore",
            "submission_date": "2022-06-01",
            "target_years": ["2030", "2050"],
            "questions_answers": {
                "What does the country promise as their 2030/2035 NDC target?": "Singapore commits to reduce emissions intensity by 36% from 2005 levels by 2030, and achieve net-zero emissions by 2050.",
                "What years are these countries using as their baseline?": "2005 is used as the baseline year for emissions intensity calculations.",
                "Are they reporting a business as usual (BAU) target rather than a base year target?": "No, Singapore is reporting against a base year target using 2005 as the baseline.",
                "What sectors are covered by the target?": "All sectors including Energy, Industrial Processes, Agriculture, and Waste. LULUCF is reported separately.",
                "What greenhouse gasses are covered by the target?": "All greenhouse gases covered under the Kyoto Protocol: CO2, CH4, N2O, HFCs, PFCs, SF6, and NF3."
            }
        }
        
        return CSVFormatter.format_to_csv(data=sample_data)
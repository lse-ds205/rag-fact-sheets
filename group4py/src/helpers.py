"""
Helper functions for the project - used for miscellaneous, non-core, furnishing/production type of tasks
"""

import logging
import colorlog
from pathlib import Path
import time
import random
from typing import List, Dict, Any, Optional, Union, Tuple
import logging
import csv
import io


class Logger:
    """
    Logger class
    """
    @staticmethod
    def setup_logging(log_file: Path, log_level: str = "INFO") -> None:
        color_formatter = colorlog.ColoredFormatter(
            "%(log_color)s%(asctime)s - %(name)s - %(levelname)s - %(message)s%(reset)s",
            datefmt="%Y-%m-%d %H:%M:%S",
            log_colors={
                'DEBUG': 'white',                 
                'INFO': 'green',                 
                'WARNING': 'yellow',          
                'ERROR': 'red',                 
                'CRITICAL': 'bold_red',   
            }
        )

        if not log_file.parent.exists():
            log_file.parent.mkdir(parents=True, exist_ok=True)

        console_handler = colorlog.StreamHandler()
        console_handler.setFormatter(color_formatter)

        file_handler = logging.FileHandler(log_file, mode="a")
        file_handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
        
        root_logger = logging.getLogger()
        root_logger.setLevel(log_level)

        root_logger.handlers.clear()
        
        root_logger.addHandler(console_handler)
        root_logger.addHandler(file_handler)
        
        # Set higher log levels for noisy third-party libraries
        logging.getLogger('pdfminer').setLevel(logging.WARNING)
        logging.getLogger('unstructured').setLevel(logging.INFO)
        logging.getLogger('pikepdf').setLevel(logging.WARNING)
        logging.getLogger('PIL').setLevel(logging.WARNING)
        logging.getLogger('matplotlib').setLevel(logging.WARNING)
        logging.getLogger('asyncio').setLevel(logging.INFO)

    @staticmethod
    def log(log_file: Path, log_level: str = "INFO"):
        """
        Decorator, typically only wrapped around main entrypoint functions (not the sub-functions), with two objectives: 
        (A) set up color logging
        (B) direct all logs to a specific file
        """
        def decorator(func):
            def wrapper(*args, **kwargs):
                Logger.setup_logging(log_file, log_level)
                result = func(*args, **kwargs)
                return result
            return wrapper
        return decorator
    
    @staticmethod
    def debug_log():
        """
        Decorator to log a debug message when a function starts running.
        """
        def decorator(func):
            def wrapper(*args, **kwargs):
                logging.debug(f"<library> {func.__name__} started running...")
                return func(*args, **kwargs)
            return wrapper
        return decorator


class TaskInfo:
    """
    Harmless decorators. To ease communication between groupmates - can use it if you want to / find that it makes collaborative development easier.
    """
    def completed():
        """
        Decorator to indicate that a task has been completed.
        """
        def decorator(func):
            def wrapper(*args, **kwargs):
                return func(*args, **kwargs)
            return wrapper
        return decorator
    
    def bryan():
        """
        Decorator to indicate that this task is being worked on by Bryan.
        """
        def decorator(func):
            def wrapper(*args, **kwargs):
                return func(*args, **kwargs)
            return wrapper
        return decorator
    
    def zicheng():
        """
        Decorator to indicate that this task is being worked on by Zicheng.
        """
        def decorator(func):
            def wrapper(*args, **kwargs):
                return func(*args, **kwargs)
            return wrapper
        return decorator

    def michele():
        """
        Decorator to indicate that this task is being worked on by Michele.
        """
        def decorator(func):
            def wrapper(*args, **kwargs):
                return func(*args, **kwargs)
            return wrapper
        return decorator
    
    def ruikai():
        """
        Decorator to indicate that this task is being worked on by Rui Kai.
        """
        def decorator(func):
            def wrapper(*args, **kwargs):
                return func(*args, **kwargs)
            return wrapper
        return decorator


class Test:
    """
    Test class - decorators to be used for testing/production phase. Removed for actual deployment.
    """
    logger = logging.getLogger(__name__)

    @staticmethod
    def sleep(duration: float):
        """
        Decorator to introduce a sleep time between function calls - simulate actual running of scripts for ease of reading logs.
        """
        def decorator(func):
            def wrapper(*args, **kwargs):
                result = func(*args, **kwargs)
                time.sleep(duration)
                return result
            return wrapper
        return decorator
    
    @staticmethod
    def dummy(dummy: Any) -> Any:
        def decorator(func):
            def wrapper(*args, **kwargs):
                result = func(*args, **kwargs)
                return dummy
            return wrapper
        return decorator
    
    @staticmethod
    def force_input(*forced_args, **forced_kwargs):
        """
        Decorator to force specific input arguments and keyword arguments to a function.
        """
        def decorator(func):
            def wrapper(*args, **kwargs):
                args = forced_args if forced_args else args
                kwargs.update(forced_kwargs)
                return func(*args, **kwargs)
            return wrapper
        return decorator

    @staticmethod
    def dummy_chunk() -> List[str]:
        dummy_chunks = ["I am dummy chunk 1", "I am dummy chunk 2", "I am dummy chunk 3"]
        def decorator(func):
            def wrapper(*args, **kwargs):
                result = func(*args, **kwargs)
                return dummy_chunks
            return wrapper
        return decorator
    
    @staticmethod
    def dummy_embedding() -> List[float]:
        dummy_embedding = [random.uniform(-1, 1) for _ in range(128)]    # Gives 128-dimensional embedding, randomly generated
        def decorator(func):
            def wrapper(*args, **kwargs):
                Test.logger.warning("Dummy embedding decorator used - not actual embedding!")
                result = func(*args, **kwargs)
                return dummy_embedding
            return wrapper
        return decorator
    
    @staticmethod
    def dummy_json() -> Dict[str, Any]:
        dummy_dict = {
            "key1": "value1",
            "key2": 42,
            "key3": [1, 2, 3],
            "key4": {"nested_key": "nested_value"}
        }
        dummy_json = [dummy_dict, dummy_dict, dummy_dict]
        def decorator(func):
            def wrapper(*args, **kwargs):
                result = func(*args, **kwargs)
                return dummy_json
            return wrapper
        return decorator
    
    @staticmethod
    def dummy_prompt() -> str:
        dummy_prompt = "I am a dummy prompt"
        def decorator(func):
            def wrapper(*args, **kwargs):
                result = func(*args, **kwargs)
                return dummy_prompt
            return wrapper
        return decorator
    
    @staticmethod
    def dummy_answer() -> str:
        dummy_answer = "I am a dummy answer"
        def decorator(func):
            def wrapper(*args, **kwargs):
                result = func(*args, **kwargs)
                return dummy_answer
            return wrapper
        return decorator


class HTMLFormatter:
    """
    HTML formatter class to generate PDF-style HTML documents from JSON data
    """
    
    # Default question set for NDC reports
    DEFAULT_QUESTIONS = {
        "What does the country promise as their 2030/2035 NDC target?": "Answer 1",
        "What years are these countries using as their baseline?": "Answer 2", 
        "Are they reporting a business as usual (BAU) target rather than a base year target?": "Answer 3",
        "What sectors are covered by the target (Ex. Energy, Industrial Processes, Land use, land use change and forestry (LULUCF), etc.)?": "Answer 4",
        "What greenhouse gasses are covered by the target?": "Answer 5",
        "What are the emissions in the baseline year (if reported)?": "Answer 6",
        "What are the emissions levels under the BAU scenario if relevant (may require getting data from tables/graphs)?": "Answer 7",
        "What promises under this new version of the document are different from the previous version of their NDC?": "Answer 8",
        "What policies or strategies does the country propose to meet its targets?": "Answer 9",
        "Do they specify what sectors of their economy will be the hardest to reduce emissions in?": "Answer 10"
    }
    
    @staticmethod
    def _get_css_styles() -> str:
        """Generate CSS styles for the HTML document"""
        return """
        <style>
            * {
                margin: 0;
                padding: 0;
                box-sizing: border-box;
            }
            
            body {
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                line-height: 1.6;
                color: #333;
                background-color: #f8f9fa;
                padding: 20px;
            }
            
            .container {
                max-width: 800px;
                margin: 0 auto;
                background-color: white;
                box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
                border-radius: 8px;
                overflow: hidden;
            }
            
            .header {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                padding: 30px;
                text-align: center;
            }
            
            .header h1 {
                font-size: 2.2em;
                font-weight: 700;
                margin-bottom: 10px;
                text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
            }
            
            .header .subtitle {
                font-size: 1.1em;
                opacity: 0.9;
                font-weight: 300;
            }
            
            .metadata {
                background-color: #f1f3f4;
                border-left: 5px solid #667eea;
                padding: 25px;
                margin: 0;
            }
            
            .metadata-grid {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
                gap: 20px;
            }
            
            .metadata-item {
                background-color: white;
                padding: 15px;
                border-radius: 6px;
                border: 1px solid #e1e5e9;
            }
            
            .metadata-label {
                font-weight: 600;
                color: #495057;
                font-size: 0.9em;
                text-transform: uppercase;
                letter-spacing: 0.5px;
                margin-bottom: 5px;
            }
            
            .metadata-value {
                font-size: 1.1em;
                color: #212529;
                font-weight: 500;
            }
            
            .target-years {
                display: inline-flex;
                flex-wrap: wrap;
                gap: 8px;
            }
            
            .target-year {
                background-color: #667eea;
                color: white;
                padding: 4px 12px;
                border-radius: 20px;
                font-size: 0.9em;
                font-weight: 500;
            }
            
            .content {
                padding: 30px;
            }
            
            .section-title {
                font-size: 1.8em;
                color: #495057;
                margin-bottom: 25px;
                padding-bottom: 10px;
                border-bottom: 3px solid #667eea;
                font-weight: 600;
            }
            
            .qa-container {
                margin-bottom: 30px;
            }
            
            .question {
                background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
                border-left: 4px solid #28a745;
                padding: 18px;
                margin-bottom: 15px;
                border-radius: 0 6px 6px 0;
                font-weight: 600;
                color: #495057;
                font-size: 1.05em;
            }
            
            .answer {
                background-color: #fff;
                border: 1px solid #dee2e6;
                border-radius: 6px;
                padding: 20px;
                margin-bottom: 25px;
                line-height: 1.7;
                color: #212529;
            }
            
            .answer p {
                margin-bottom: 10px;
            }
            
            .no-data {
                color: #6c757d;
                font-style: italic;
                padding: 15px;
                background-color: #f8f9fa;
                border-radius: 4px;
                text-align: center;
            }
            
            .footer {
                background-color: #495057;
                color: white;
                text-align: center;
                padding: 20px;
                font-size: 0.9em;
            }
            
            @media print {
                body {
                    background-color: white;
                    padding: 0;
                }
                
                .container {
                    box-shadow: none;
                    border-radius: 0;
                }
                
                .header {
                    background: #667eea !important;
                    -webkit-print-color-adjust: exact;
                }
                
                .target-year {
                    background-color: #667eea !important;
                    -webkit-print-color-adjust: exact;
                }
            }
        </style>
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
        logger = logging.getLogger(__name__)
        
        # Extract data from JSON if provided
        if data:
            country = data.get('country', country)
            submission_date = data.get('submission_date', submission_date)
            target_years = data.get('target_years', target_years)
            questions_answers = data.get('questions_answers', questions_answers)
        
        # Use default questions if none provided
        if not questions_answers:
            questions_answers = HTMLFormatter.DEFAULT_QUESTIONS.copy()
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
        from datetime import datetime
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Build complete HTML document
        html_content = f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>TPI X DS205 NDC Report - {country}</title>
            {HTMLFormatter._get_css_styles()}
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
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"ndc_report_{timestamp}.html"
        
        # Ensure .html extension
        if not filename.endswith('.html'):
            filename += '.html'
        
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(html_content)
            
            logger = logging.getLogger(__name__)
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
    
    # Default question set for NDC reports (same as HTMLFormatter)
    DEFAULT_QUESTIONS = HTMLFormatter.DEFAULT_QUESTIONS
    
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
        logger = logging.getLogger(__name__)
        
        # Extract data from JSON if provided
        if data:
            country = data.get('country', country)
            submission_date = data.get('submission_date', submission_date)
            target_years = data.get('target_years', target_years)
            questions_answers = data.get('questions_answers', questions_answers)
        
        # Use default questions if none provided
        if not questions_answers:
            questions_answers = CSVFormatter.DEFAULT_QUESTIONS.copy()
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
        logger = logging.getLogger(__name__)
        
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
            from datetime import datetime
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
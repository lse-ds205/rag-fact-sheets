#!/usr/bin/env python3
"""
Email sending script using Supabase Edge Functions.
This script invokes a Supabase Edge Function that handles email sending via Resend.
"""

import os
import sys
import json
import argparse
from typing import Dict, Any, Optional, List, Union
from dotenv import load_dotenv
from supabase import create_client, Client
import base64
from pathlib import Path


project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))
import group4py
from helpers.formatter import HTMLFormatter, CSVFormatter
from constants.questions import DEFAULT_QUESTIONS

load_dotenv()


class SupabaseEmailSender:
    """
    A class to handle email sending through Supabase Edge Functions.
    All sensitive configuration is loaded from environment variables/.env file.
    """
    
    def __init__(self):
        """Initialize the Supabase client with environment variables from .env file."""
        # Load Supabase configuration from .env
        self.supabase_url = os.getenv('SUPABASE_URL')
        self.supabase_key = os.getenv('SUPABASE_ANON_KEY')
        self.edge_function_name = os.getenv('EDGE_FUNCTION_NAME', 'send-email')
        
        # Validate required environment variables
        if not self.supabase_url or not self.supabase_key:
            raise ValueError(
                "Missing required environment variables in .env file:\n"
                "- SUPABASE_URL\n"
                "- SUPABASE_ANON_KEY\n"
                "Please check your .env file configuration."
            )
        
        # Initialize Supabase client
        try:
            self.supabase: Client = create_client(self.supabase_url, self.supabase_key)
            print("✅ Supabase client initialized successfully")
        except Exception as e:
            raise ConnectionError(f"Failed to initialize Supabase client: {str(e)}")
    
    def get_default_sender(self) -> str:
        """Get default sender email from environment variables."""
        return os.getenv('DEFAULT_EMAIL_SENDER', 'noreply@yourapp.com')
    
    def get_default_recipient(self) -> str:
        """Get default recipient email from environment variables."""
        default_recipients = "B.W.Q.Tan@lse.ac.uk,Z.Liu116@lse.ac.uk,R.Liu48@lse.ac.uk,M.Silvestri1@lse.ac.uk"
        return os.getenv('DEFAULT_EMAIL_RECIPIENT', default_recipients)
    
    def send_email(
        self,
        to_email: str,
        subject: str,
        html_content: str,
        from_email: Optional[str] = None,
        text_content: Optional[str] = None,
        reply_to: Optional[str] = None,
        cc: Optional[str] = None,
        bcc: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None
    ) -> Dict[str, Any]:
        """
        Send an email using Supabase Edge Function.
        
        Args:
            to_email (str): Recipient email address (required) - can be comma-separated for multiple recipients
            subject (str): Email subject (required)
            html_content (str): HTML content of the email (required)
            from_email (str, optional): Sender email address (defaults to DEFAULT_EMAIL_SENDER from .env)
            text_content (str, optional): Plain text content
            reply_to (str, optional): Reply-to email address
            cc (str, optional): CC email address
            bcc (str, optional): BCC email address
            tags (Dict[str, str], optional): Tags for email tracking
            
        Returns:
            Dict[str, Any]: Response from the Edge Function
        """
        try:
            # Use default sender if not provided
            if not from_email:
                from_email = self.get_default_sender()
            
            # Handle multiple recipients by splitting comma-separated emails
            recipients = [email.strip() for email in to_email.split(',') if email.strip()]
            
            if not recipients:
                return {
                    "success": False,
                    "error": "No valid email addresses found",
                    "status_code": None
                }
            
            # Limit to 50 recipients as per Resend API limit
            if len(recipients) > 50:
                recipients = recipients[:50]
                print(f"⚠️ Warning: Limited to first 50 recipients due to API constraints")
            
            # Prepare the email payload using array format for multiple recipients
            email_payload = {
                "to": recipients if len(recipients) > 1 else recipients[0],  # Array for multiple, string for single
                "from": from_email,
                "subject": subject,
                "html": html_content,
            }
            
            # Add optional fields if provided
            if text_content:
                email_payload["text"] = text_content
            if reply_to:
                email_payload["reply_to"] = reply_to
            if cc:
                email_payload["cc"] = cc
            if bcc:
                email_payload["bcc"] = bcc
            if tags:
                email_payload["tags"] = tags
            
            print(f"📧 Sending email to: {', '.join(recipients)}")
            print(f"📝 Subject: {subject}")
            print(f"👤 From: {from_email}")
            
            # Invoke the Edge Function
            response = self.supabase.functions.invoke(
                self.edge_function_name,
                invoke_options={
                    "body": email_payload
                }
            )
            
            # Handle different response formats from Supabase Python client
            if hasattr(response, 'status_code'):
                status_code = response.status_code
            else:
                status_code = getattr(response, 'status', 200)
            
            if status_code == 200:
                print("✅ Email sent successfully!")
                
                # Extract response data
                if hasattr(response, 'json'):
                    response_data = response.json()
                elif hasattr(response, 'data'):
                    response_data = response.data
                else:
                    response_data = {"message": "Email sent successfully"}
                
                return {
                    "success": True,
                    "data": response_data,
                    "status_code": status_code,
                    "email_id": response_data.get("id") if isinstance(response_data, dict) else None,
                    "recipients": recipients
                }
            else:
                print(f"❌ Failed to send email. Status: {status_code}")
                
                # Extract error message
                if hasattr(response, 'text'):
                    error_msg = response.text
                elif hasattr(response, 'data'):
                    error_msg = str(response.data)
                else:
                    error_msg = f"HTTP {status_code} error"
                
                return {
                    "success": False,
                    "error": error_msg,
                    "status_code": status_code
                }
                
        except Exception as e:
            print(f"❌ Error sending email: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "status_code": None
            }
    
    def send_template_email(
        self,
        to_email: str,
        template_name: str,
        template_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Send an email using a predefined template.
        
        Args:
            to_email (str): Recipient email address
            template_name (str): Name of the email template
            template_data (Dict[str, Any]): Data to populate the template
            
        Returns:
            Dict[str, Any]: Response from the Edge Function
        """
        try:
            # Prepare the template payload
            template_payload = {
                "to": to_email,
                "template": template_name,
                "data": template_data
            }
            
            print(f"Sending template email '{template_name}' to: {to_email}")
            
            # Invoke the Edge Function for templates
            response = self.supabase.functions.invoke(
                "send-template-email",  # Name of your template Edge Function
                invoke_options={
                    "body": template_payload
                }
            )
            
            if response.status_code == 200:
                print("✅ Template email sent successfully!")
                return {
                    "success": True,
                    "data": response.json(),
                    "status_code": response.status_code
                }
            else:
                print(f"❌ Failed to send template email. Status: {response.status_code}")
                return {
                    "success": False,
                    "error": response.text,
                    "status_code": response.status_code
                }
                
        except Exception as e:
            print(f"❌ Error sending template email: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "status_code": None
            }

    def send_ndc_report_email(
        self,
        to_email: str,
        country: str,
        submission_date: str,
        target_years: Union[List[str], str],
        questions_answers: Dict[str, str],
        hoprag_results: Optional[Dict[str, Any]] = None,
        from_email: Optional[str] = None,
        llm_response_file: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Send a complete NDC report email with both HTML and CSV attachments.
        
        Args:
            to_email: Recipient email address
            country: Country name
            submission_date: Submission date
            target_years: Target year(s)
            questions_answers: Q&A dictionary
            hoprag_results: Optional HopRAG analysis results
            from_email: Optional sender email
            llm_response_file: Optional path to LLM response JSON file
            
        Returns:
            Dict[str, Any]: Response from email sending
        """
        try:
            # Load LLM response data if provided
            llm_data = None
            if llm_response_file and Path(llm_response_file).exists():
                try:
                    with open(llm_response_file, 'r', encoding='utf-8') as f:
                        llm_data = json.load(f)
                    print(f"📄 Loaded LLM response data from: {llm_response_file}")
                except Exception as e:
                    print(f"⚠️ Warning: Could not load LLM response file: {e}")
            elif not llm_response_file:
                # Try to find llm_response.json in data folder
                data_dir = project_root / "data"
                llm_file = data_dir / "llm_response.json"
                if llm_file.exists():
                    try:
                        with open(llm_file, 'r', encoding='utf-8') as f:
                            llm_data = json.load(f)
                        print(f"📄 Found and loaded LLM response data from: {llm_file}")
                    except Exception as e:
                        print(f"⚠️ Warning: Could not load default LLM response file: {e}")
            
            # If LLM data exists, integrate it into questions_answers
            if llm_data:
                # Merge LLM data into questions_answers if it has relevant structure
                if isinstance(llm_data, dict):
                    # If llm_data has questions/answers structure, use it
                    if 'questions_answers' in llm_data:
                        questions_answers.update(llm_data['questions_answers'])
                    elif 'response' in llm_data:
                        questions_answers['LLM Analysis'] = str(llm_data['response'])
                    else:
                        # Add the entire LLM data as a formatted response
                        questions_answers['LLM Response'] = json.dumps(llm_data, indent=2)
                print(f"📊 Integrated LLM data into report content")
            
            # Generate HTML content
            html_content = HTMLFormatter.format_to_html(
                country=country,
                submission_date=submission_date,
                target_years=target_years,
                questions_answers=questions_answers
            )
            
            # Generate CSV content
            csv_content = CSVFormatter.format_to_csv(
                country=country,
                submission_date=submission_date,
                target_years=target_years,
                questions_answers=questions_answers
            )
            
            # Prepare attachments
            attachments = [
                {
                    "filename": f"ndc_report_{country.lower().replace(' ', '_')}.csv",
                    "content": csv_content,
                    "content_type": "text/csv"
                }
            ]
            
            # Add HopRAG results if available
            if hoprag_results:
                hoprag_csv = CSVFormatter.format_hoprag_results_to_csv(
                    hoprag_results=hoprag_results,
                    country=country,
                    submission_date=submission_date,
                    target_years=target_years
                )
                attachments.append({
                    "filename": f"hoprag_analysis_{country.lower().replace(' ', '_')}.csv",
                    "content": hoprag_csv,
                    "content_type": "text/csv"
                })
            
            # Add PDF from outputs/factsheets if available
            pdf_path = self.find_latest_pdf_factsheet()
            if pdf_path:
                try:
                    with open(pdf_path, 'rb') as f:
                        pdf_content = f.read()
                    attachments.append({
                        "filename": Path(pdf_path).name,
                        "content": pdf_content,
                        "content_type": "application/pdf"
                    })
                    print(f"📎 Added PDF attachment: {Path(pdf_path).name}")
                except Exception as e:
                    print(f"⚠️ Warning: Could not attach PDF: {e}")
            
            # Email subject
            subject = f"TPI X DS205 NDC Report - {country}"
            
            # Send email with attachments
            return self.send_email_with_attachments(
                to_email=to_email,
                subject=subject,
                html_content=html_content,
                attachments=attachments,
                from_email=from_email,
                text_content=f"Please find attached the NDC report for {country}.",
                tags={"report_type": "ndc", "country": country}
            )
            
        except Exception as e:
            print(f"❌ Error sending NDC report email: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "status_code": None
            }

    def find_latest_pdf_factsheet(self) -> Optional[str]:
        """Find the most recently generated PDF factsheet in the outputs folder."""
        try:
            outputs_dir = project_root / "outputs" / "factsheets"
            
            if not outputs_dir.exists():
                print(f"⚠️ Outputs directory not found: {outputs_dir}")
                return None
            
            # Find all PDF files
            pdf_files = list(outputs_dir.glob("*.pdf"))
            
            if not pdf_files:
                print(f"⚠️ No PDF files found in: {outputs_dir}")
                return None
            
            # Sort by modification time (most recent first)
            pdf_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
            latest_pdf = pdf_files[0]
            
            print(f"📄 Found latest PDF factsheet: {latest_pdf.name}")
            return str(latest_pdf)
            
        except Exception as e:
            print(f"⚠️ Warning: Error finding PDF factsheet: {e}")
            return None

    def send_email_with_attachments(
        self,
        to_email: str,
        subject: str,
        html_content: str,
        attachments: List[Dict[str, Any]],
        from_email: Optional[str] = None,
        text_content: Optional[str] = None,
        reply_to: Optional[str] = None,
        cc: Optional[str] = None,
        bcc: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None
    ) -> Dict[str, Any]:
        """
        Send an email with attachments using Supabase Edge Function.
        
        Args:
            to_email (str): Recipient email address (can be comma-separated)
            subject (str): Email subject
            html_content (str): HTML content
            attachments (List[Dict]): List of attachments with filename, content, content_type
            from_email (str, optional): Sender email
            text_content (str, optional): Plain text content
            reply_to (str, optional): Reply-to email
            cc (str, optional): CC email
            bcc (str, optional): BCC email
            tags (Dict[str, str], optional): Email tags
            
        Returns:
            Dict[str, Any]: Response from email sending
        """
        try:
            # Use default sender if not provided
            if not from_email:
                from_email = self.get_default_sender()
            
            # Handle multiple recipients
            recipients = [email.strip() for email in to_email.split(',') if email.strip()]
            
            if not recipients:
                return {
                    "success": False,
                    "error": "No valid email addresses found",
                    "status_code": None
                }
            
            # Limit to 50 recipients as per Resend API limit
            if len(recipients) > 50:
                recipients = recipients[:50]
                print(f"⚠️ Warning: Limited to first 50 recipients due to API constraints")
            
            # Prepare attachments (encode binary content to base64)
            processed_attachments = []
            for attachment in attachments:
                if isinstance(attachment['content'], bytes):
                    # Binary content - encode to base64
                    content_b64 = base64.b64encode(attachment['content']).decode('utf-8')
                else:
                    # Text content - encode to base64
                    content_b64 = base64.b64encode(attachment['content'].encode('utf-8')).decode('utf-8')
                
                processed_attachments.append({
                    "filename": attachment['filename'],
                    "content": content_b64,
                    "content_type": attachment['content_type']
                })
            
            # Prepare email payload using array format for multiple recipients
            email_payload = {
                "to": recipients if len(recipients) > 1 else recipients[0],  # Array for multiple, string for single
                "from": from_email,
                "subject": subject,
                "html": html_content,
                "attachments": processed_attachments
            }
            
            # Add optional fields
            if text_content:
                email_payload["text"] = text_content
            if reply_to:
                email_payload["reply_to"] = reply_to
            if cc:
                email_payload["cc"] = cc
            if bcc:
                email_payload["bcc"] = bcc
            if tags:
                email_payload["tags"] = tags
            
            print(f"📧 Sending email with {len(attachments)} attachment(s) to: {', '.join(recipients)}")
            
            # Invoke the Edge Function
            response = self.supabase.functions.invoke(
                self.edge_function_name,
                invoke_options={
                    "body": email_payload
                }
            )
            
            # Handle response
            if hasattr(response, 'status_code'):
                status_code = response.status_code
            else:
                status_code = getattr(response, 'status', 200)
            
            if status_code == 200:
                print("✅ Email with attachments sent successfully!")
                
                if hasattr(response, 'json'):
                    response_data = response.json()
                elif hasattr(response, 'data'):
                    response_data = response.data
                else:
                    response_data = {"message": "Email sent successfully"}
                
                return {
                    "success": True,
                    "data": response_data,
                    "status_code": status_code,
                    "recipients": recipients,
                    "attachments_count": len(attachments)
                }
            else:
                print(f"❌ Failed to send email with attachments. Status: {status_code}")
                
                if hasattr(response, 'text'):
                    error_msg = response.text
                elif hasattr(response, 'data'):
                    error_msg = str(response.data)
                else:
                    error_msg = f"HTTP {status_code} error"
                
                return {
                    "success": False,
                    "error": error_msg,
                    "status_code": status_code
                }
                
        except Exception as e:
            print(f"❌ Error sending email with attachments: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "status_code": None
            }

    def send_factsheet_email(
        self,
        to_email: str,
        pdf_path: str,
        from_email: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Send the PDF factsheet as an email attachment.
        
        Args:
            to_email: Recipient email address
            pdf_path: Path to the PDF factsheet file
            from_email: Optional sender email
            
        Returns:
            Dict[str, Any]: Response from email sending
        """
        try:
            # Verify PDF file exists
            pdf_file = Path(pdf_path)
            if not pdf_file.exists():
                raise FileNotFoundError(f"PDF file not found: {pdf_path}")
            
            # Read PDF file content
            with open(pdf_file, 'rb') as f:
                pdf_content = f.read()
            
            # Create HTML email content
            html_content = f"""
            <html>
            <head>
                <style>
                    body {{ font-family: Arial, sans-serif; margin: 20px; }}
                    .header {{ background-color: #f8f9fa; padding: 20px; border-radius: 5px; }}
                    .content {{ margin: 20px 0; }}
                    .footer {{ font-size: 12px; color: #6c757d; margin-top: 30px; }}
                </style>
            </head>
            <body>
                <div class="header">
                    <h2>📄 Climate Policy Fact Sheet</h2>
                </div>
                
                <div class="content">
                    <p>Dear Recipient,</p>
                    
                    <p>Please find attached the Climate Policy Fact Sheet generated from our analysis.</p>
                    
                    <p>This report contains:</p>
                    <ul>
                        <li>Climate policy analysis</li>
                        <li>Source citations with relevance scores</li>
                        <li>Detailed metadata</li>
                    </ul>
                    
                    <p>The attached PDF provides a comprehensive overview of the analyzed climate policies and commitments.</p>
                </div>
                
                <div class="footer">
                    <p>Best regards,<br>Climate Policy Analysis Team</p>
                    <p><small>This report was generated automatically from our RAG-based analysis system.</small></p>
                </div>
            </body>
            </html>
            """
            
            # Create text version
            text_content = f"""
Climate Policy Fact Sheet

Dear Recipient,

Please find attached the Climate Policy Fact Sheet generated from our analysis.

This report contains:
- Climate policy analysis
- Source citations with relevance scores  
- Detailed metadata

The attached PDF provides a comprehensive overview of the analyzed climate policies and commitments.

Best regards,
Climate Policy Analysis Team

---
This report was generated automatically from our RAG-based analysis system.
            """
            
            # Prepare attachment
            attachments = [
                {
                    "filename": pdf_file.name,
                    "content": pdf_content,
                    "content_type": "application/pdf"
                }
            ]
            
            # Email subject
            subject = f"Climate Policy Fact Sheet - {pdf_file.stem}"
            
            # Send email with PDF attachment
            return self.send_email_with_attachments(
                to_email=to_email,
                subject=subject,
                html_content=html_content,
                attachments=attachments,
                from_email=from_email,
                text_content=text_content,
                tags={"report_type": "factsheet", "format": "pdf"}
            )
            
        except Exception as e:
            print(f"❌ Error sending factsheet email: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "status_code": None
            }


def parse_arguments():
    """
    Parse command line arguments for flexible email configuration.
    """
    parser = argparse.ArgumentParser(
        description="Send emails via Supabase Edge Functions",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Send NDC report (default behavior)
  python send_email.py

  # Send NDC report with specific parameters
  python send_email.py --country Singapore --submission-date "2022-06-01" --target-years "2030,2050"

  # Send NDC report with custom LLM response file
  python send_email.py --llm-response-file data/custom_llm_response.json

  # Send latest factsheet PDF
  python send_email.py --factsheet --to user@example.com

  # Send specific factsheet PDF
  python send_email.py --factsheet --pdf-path outputs/factsheets/report.pdf --to user@example.com
        """
    )
    
    # Factsheet specific arguments
    parser.add_argument(
        '--factsheet',
        action='store_true',
        help='Send the generated PDF factsheet as an email attachment'
    )
    
    parser.add_argument(
        '--pdf-path',
        help='Specific path to PDF file (defaults to latest in outputs/factsheets/)'
    )
    
    # NDC Report specific arguments
    parser.add_argument(
        '--ndc-report',
        action='store_true',
        help='Send a complete NDC report with HTML and CSV attachments'
    )
    
    parser.add_argument(
        '--country',
        help='Country name for NDC report'
    )
    
    parser.add_argument(
        '--submission-date',
        help='Submission date for NDC report (YYYY-MM-DD format)'
    )
    
    parser.add_argument(
        '--target-years',
        help='Target years for NDC report (comma-separated, e.g., "2030,2050")'
    )
    
    parser.add_argument(
        '--qa-file',
        help='JSON file containing questions and answers'
    )
    
    parser.add_argument(
        '--hoprag-file',
        help='JSON file containing HopRAG analysis results'
    )
    
    # Required/Primary arguments
    parser.add_argument(
        '--to', '--recipient',
        help='Recipient email address (default: DEFAULT_EMAIL_RECIPIENT from .env)'
    )
    
    parser.add_argument(
        '--subject',
        help='Email subject (default: from .env or auto-generated)'
    )
    
    # Email content arguments
    parser.add_argument(
        '--html',
        help='HTML content of the email'
    )
    
    parser.add_argument(
        '--html-file',
        help='Path to HTML file containing email content'
    )
    
    parser.add_argument(
        '--text',
        help='Plain text content of the email'
    )
    
    parser.add_argument(
        '--text-file',
        help='Path to text file containing plain text content'
    )
    
    # Sender information
    parser.add_argument(
        '--from', '--sender',
        help='Sender email address (default: DEFAULT_EMAIL_SENDER from .env)'
    )
    
    parser.add_argument(
        '--reply-to',
        help='Reply-to email address'
    )
    
    parser.add_argument(
        '--cc',
        help='CC email address'
    )
    
    parser.add_argument(
        '--bcc',
        help='BCC email address'
    )
    
    # Template arguments
    parser.add_argument(
        '--template',
        help='Template name for template-based emails'
    )
    
    parser.add_argument(
        '--template-data',
        help='JSON string with template data (e.g., \'{"name": "John", "company": "Acme"}\')'
    )
    
    # Configuration arguments
    parser.add_argument(
        '--env-file',
        default='.env',
        help='Path to .env file (default: .env)'
    )
    
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show what would be sent without actually sending'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose output'
    )
    
    parser.add_argument(
        '--llm-response-file',
        help='JSON file containing LLM response data for email content (defaults to data/llm_response.json)'
    )
    
    return parser.parse_args()


def load_file_content(file_path: str) -> str:
    """Load content from a file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found: {file_path}")
    except Exception as e:
        raise Exception(f"Error reading file {file_path}: {str(e)}")


def create_default_content(args) -> tuple[str, str]:
    """Create default HTML and text content if not provided."""
    
    # Default HTML content with GitHub Actions context
    html_content = f"""
    <html>
    <head>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            .header {{ background-color: #f8f9fa; padding: 20px; border-radius: 5px; }}
            .content {{ margin: 20px 0; }}
            .footer {{ font-size: 12px; color: #6c757d; margin-top: 30px; }}
            .info-table {{ border-collapse: collapse; width: 100%; }}
            .info-table td {{ padding: 8px; border-bottom: 1px solid #dee2e6; }}
            .info-table .label {{ font-weight: bold; background-color: #f8f9fa; width: 30%; }}
        </style>
    </head>
    <body>
        <div class="header">
            <h2>📧 Automated Email Notification</h2>
        </div>
        
        <div class="content">
            <p>This email was sent automatically using Supabase Edge Functions.</p>
            
            <table class="info-table">
                <tr>
                    <td class="label">Repository:</td>
                    <td>{os.getenv('GITHUB_REPOSITORY', 'N/A')}</td>
                </tr>
                <tr>
                    <td class="label">Workflow:</td>
                    <td>{os.getenv('GITHUB_WORKFLOW', 'N/A')}</td>
                </tr>
                <tr>
                    <td class="label">Actor:</td>
                    <td>{os.getenv('GITHUB_ACTOR', 'N/A')}</td>
                </tr>
                <tr>
                    <td class="label">Run ID:</td>
                    <td>{os.getenv('GITHUB_RUN_ID', 'N/A')}</td>
                </tr>
                <tr>
                    <td class="label">Event:</td>
                    <td>{os.getenv('GITHUB_EVENT_NAME', 'manual')}</td>
                </tr>
                <tr>
                    <td class="label">Timestamp:</td>
                    <td>{os.getenv('GITHUB_RUN_STARTED_AT', 'N/A')}</td>
                </tr>
            </table>
        </div>
        
        <div class="footer">
            <p>Best regards,<br>Your Automated System</p>
            <p><small>This email was generated automatically. Please do not reply to this email.</small></p>
        </div>
    </body>
    </html>
    """
    
    # Default text content
    text_content = f"""
Automated Email Notification

This email was sent automatically using Supabase Edge Functions.

Repository: {os.getenv('GITHUB_REPOSITORY', 'N/A')}
Workflow: {os.getenv('GITHUB_WORKFLOW', 'N/A')}
Actor: {os.getenv('GITHUB_ACTOR', 'N/A')}
Run ID: {os.getenv('GITHUB_RUN_ID', 'N/A')}
Event: {os.getenv('GITHUB_EVENT_NAME', 'manual')}
Timestamp: {os.getenv('GITHUB_RUN_STARTED_AT', 'N/A')}

Best regards,
Your Automated System

---
This email was generated automatically. Please do not reply to this email.
    """
    
    return html_content.strip(), text_content.strip()


def main():
    try:
        # Parse command-line arguments
        args = parse_arguments()
        
        # Initialize the email sender
        email_sender = SupabaseEmailSender()
        
        # Handle factsheet emails
        if args.factsheet:
            # Get PDF path
            if args.pdf_path:
                pdf_path = args.pdf_path
                if not Path(pdf_path).is_absolute():
                    pdf_path = project_root / pdf_path
            else:
                pdf_path = email_sender.find_latest_pdf_factsheet()
                
            if not pdf_path:
                print("❌ No PDF factsheet found to send")
                sys.exit(1)
            
            # Get recipient
            recipient = args.to or email_sender.get_default_recipient()
            
            if args.dry_run:
                print("\n=== DRY RUN - FACTSHEET EMAIL WOULD BE SENT ===")
                print(f"To: {recipient}")
                print(f"PDF: {pdf_path}")
                print(f"From: {getattr(args, 'from', None) or email_sender.get_default_sender()}")
                print("=== DRY RUN COMPLETE - NO EMAIL SENT ===")
                return
            
            # Send factsheet email
            result = email_sender.send_factsheet_email(
                to_email=recipient,
                pdf_path=pdf_path,
                from_email=getattr(args, 'from', None)
            )
        
        # Handle NDC report emails (default behavior if no specific type is specified)
        elif args.ndc_report or (not args.factsheet and not args.template):
            # Get required NDC report parameters with defaults
            country = args.country or "Singapore"  # Default country
            submission_date = args.submission_date or "2022-06-01"  # Default submission date
            
            # Parse target years
            if args.target_years:
                target_years = [year.strip() for year in args.target_years.split(',')]
            else:
                target_years = ["2030", "2050"]  # Default target years
            
            # Load Q&A data
            if args.qa_file:
                try:
                    with open(args.qa_file, 'r', encoding='utf-8') as f:
                        questions_answers = json.load(f)
                except Exception as e:
                    raise ValueError(f"Error loading Q&A file: {e}")
            else:
                questions_answers = DEFAULT_QUESTIONS
            
            # Load HopRAG results if provided
            hoprag_results = None
            if args.hoprag_file:
                try:
                    with open(args.hoprag_file, 'r', encoding='utf-8') as f:
                        hoprag_results = json.load(f)
                except Exception as e:
                    print(f"Warning: Could not load HopRAG file: {e}")
            
            # Get recipient
            recipient = args.to or email_sender.get_default_recipient()
            
            if args.dry_run:
                print("\n=== DRY RUN - NDC REPORT EMAIL WOULD BE SENT ===")
                print(f"To: {recipient}")
                print(f"Country: {country}")
                print(f"Submission Date: {submission_date}")
                print(f"Target Years: {', '.join(target_years)}")
                print(f"From: {getattr(args, 'from', None) or email_sender.get_default_sender()}")
                print(f"Q&A File: {args.qa_file or 'Using defaults'}")
                print(f"HopRAG File: {args.hoprag_file or 'None'}")
                print(f"LLM Response File: {args.llm_response_file or 'data/llm_response.json (auto-detect)'}")
                
                # Check for PDF factsheet
                pdf_path = email_sender.find_latest_pdf_factsheet()
                print(f"PDF Factsheet: {pdf_path if pdf_path else 'None found'}")
                
                print("=== DRY RUN COMPLETE - NO EMAIL SENT ===")
                return
            
            # Send NDC report email
            result = email_sender.send_ndc_report_email(
                to_email=recipient,
                country=country,
                submission_date=submission_date,
                target_years=target_years,
                questions_answers=questions_answers,
                hoprag_results=hoprag_results,
                from_email=getattr(args, 'from', None),
                llm_response_file=args.llm_response_file
            )
        else:
            # Get email configuration from environment variables or use defaults
            recipient = args.to or os.getenv('EMAIL_RECIPIENT', email_sender.get_default_recipient())
            subject = args.subject or os.getenv('EMAIL_SUBJECT', 'Test Email from GitHub Actions')
            sender = getattr(args, 'from', None) or os.getenv('EMAIL_SENDER', email_sender.get_default_sender())
            
            # Handle template emails
            if args.template:
                try:
                    template_data = json.loads(args.template_data) if args.template_data else {}
                except json.JSONDecodeError:
                    raise ValueError(f"Invalid JSON in template data: {args.template_data}")
                
                result = email_sender.send_template_email(
                    to_email=recipient,
                    template_name=args.template,
                    template_data=template_data
                )
            else:
                # Handle content from files or arguments
                html_content = None
                text_content = None
                
                if args.html:
                    html_content = args.html
                elif args.html_file:
                    html_content = load_file_content(args.html_file)
                    
                if args.text:
                    text_content = args.text
                elif args.text_file:
                    text_content = load_file_content(args.text_file)
                    
                # If no content is provided, create default content
                if not html_content and not text_content:
                    html_content, text_content = create_default_content(args)
                    
                # Print what would be sent in dry run mode
                if args.dry_run:
                    print("\n=== DRY RUN - EMAIL WOULD BE SENT WITH THESE DETAILS ===")
                    print(f"To: {recipient}")
                    print(f"From: {sender}")
                    print(f"Subject: {subject}")
                    print(f"Reply-To: {args.reply_to}")
                    print(f"CC: {args.cc}")
                    print(f"BCC: {args.bcc}")
                    print("\n--- HTML Content ---")
                    print(html_content[:500] + ("..." if len(html_content) > 500 else ""))
                    print("\n--- Text Content ---")
                    print(text_content[:500] + ("..." if len(text_content) > 500 else ""))
                    print("\n=== DRY RUN COMPLETE - NO EMAIL SENT ===")
                    return
                    
                # Send the email
                result = email_sender.send_email(
                    to_email=recipient,  # Receiver
                    subject=subject,     # Subject
                    html_content=html_content,  # Message content (HTML)
                    from_email=sender,   # Sender
                    text_content=text_content,  # Message content (plain text)
                    reply_to=args.reply_to,
                    cc=args.cc,
                    bcc=args.bcc
                )
        
        # Print result and exit with appropriate code
        if result['success']:
            print("🎉 Email delivery initiated successfully!")
            sys.exit(0)
        else:
            print("💥 Email delivery failed!")
            print(f"Error details: {result.get('error', 'Unknown error')}")
            sys.exit(1)
            
    except Exception as e:
        print(f"💥 Script execution failed: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()

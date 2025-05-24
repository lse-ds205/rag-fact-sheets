#!/usr/bin/env python3
"""
Email sending script using Supabase Edge Functions.
This script invokes a Supabase Edge Function that handles email sending via Resend.
"""

import os
import sys
import json
import argparse
from typing import Dict, Any, Optional
from dotenv import load_dotenv
from supabase import create_client, Client

# Load environment variables from .env file
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
        return os.getenv('DEFAULT_EMAIL_RECIPIENT', 'admin@yourapp.com')
    
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
            to_email (str): Recipient email address (required)
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
            
            # Prepare the email payload that matches your Edge Function
            email_payload = {
                "to": to_email,
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
            
            print(f"📧 Sending email to: {to_email}")
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
                    "email_id": response_data.get("id") if isinstance(response_data, dict) else None
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


def parse_arguments():
    """
    Parse command line arguments for flexible email configuration.
    """
    parser = argparse.ArgumentParser(
        description="Send emails via Supabase Edge Functions",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic email (uses defaults from .env)
  python send_email.py

  # Custom recipient and subject
  python send_email.py --to user@example.com --subject "Hello World"

  # Full custom email
  python send_email.py --to user@example.com --from sender@company.com \\
                       --subject "Important Update" --html "<h1>Hello</h1>" \\
                       --text "Hello" --reply-to support@company.com

  # Template email
  python send_email.py --template welcome --to newuser@example.com \\
                       --template-data '{"name": "John", "company": "Acme"}'

  # Load HTML from file
  python send_email.py --to user@example.com --html-file email.html
        """
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
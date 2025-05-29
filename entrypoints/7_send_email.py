#!/usr/bin/env python3
"""
Email sending script using Supabase Edge Functions.
This script invokes a Supabase Edge Function that handles email sending via
Resend.
"""

import os
import sys
import re
import argparse
from typing import Dict, Any, Optional, List
from dotenv import load_dotenv
from supabase import create_client, Client
import base64
from pathlib import Path

# Set up project root and path
project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

# Load environment variables
load_dotenv()

# Define a strict tag sanitization pattern
TAG_PATTERN = re.compile(r'[^a-zA-Z0-9_-]')


def sanitize_tag(value):
    """Convert a string to a valid tag format (only ASCII letters, numbers, underscores, dashes)"""
    if not value:
        return ""
    # Replace any disallowed characters with underscores
    return TAG_PATTERN.sub('_', str(value))


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
        default_recipients = (
            "B.W.Q.Tan@lse.ac.uk,Z.Liu116@lse.ac.uk,"
            "R.Liu48@lse.ac.uk,M.Silvestri1@lse.ac.uk"
        )
        return os.getenv('DEFAULT_EMAIL_RECIPIENT', default_recipients)

    def find_latest_pdf_factsheet(self) -> Optional[str]:
        """Find the most recently generated PDF factsheet in the outputs folder."""
        try:
            # Try primary path first
            outputs_dir = project_root / "outputs" / "factsheets"
            print(f"Looking for PDFs in: {outputs_dir}")

            # If primary path doesn't exist, check alternatives
            if not outputs_dir.exists():
                print("⚠️ Primary outputs directory not found: {outputs_dir}")

                # Try alternative paths
                alt_paths = [
                    project_root / "factsheets",
                    project_root / "output" / "factsheets",
                    project_root / "data" / "factsheets",
                ]

                for alt_path in alt_paths:
                    if alt_path.exists():
                        print(f"Trying alternative path: {alt_path}")
                        # Check if this path has PDFs
                        test_pdfs = list(alt_path.glob("**/*.pdf"))
                        if test_pdfs:
                            outputs_dir = alt_path
                            print(f"✅ Found alternative path with PDFs: {outputs_dir}")
                            break

                # If still no valid directory found
                if not outputs_dir.exists():
                    print("❌ No valid directory with PDFs found after checking alternatives")
                    return None

            # Try direct pattern first
            pdf_files = list(outputs_dir.glob("*.pdf"))

            # If no PDFs found directly, search recursively
            if not pdf_files:
                print("No PDFs found directly in {outputs_dir}, searching subdirectories...")
                pdf_files = list(outputs_dir.glob("**/*.pdf"))

            if not pdf_files:
                print(f"❌ No PDF files found in or under: {outputs_dir}")
                return None

            # Sort by modification time (most recent first)
            pdf_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
            latest_pdf = pdf_files[0]

            print(
                f"📄 Found latest PDF factsheet: {latest_pdf.name} "
                f"(Modified: {latest_pdf.stat().st_mtime})"
            )
            print(f"  - Full path: {latest_pdf}")

            # Verify file exists and is readable
            if not latest_pdf.exists() or not os.access(str(latest_pdf), os.R_OK):
                print(f"❌ PDF file exists but is not readable: {latest_pdf}")
                return None

            # Verify file is not empty
            if latest_pdf.stat().st_size == 0:
                print(f"❌ PDF file exists but is empty (0 bytes): {latest_pdf}")
                return None

            print(
                f"✅ PDF validation passed: {latest_pdf.name} "
                f"({latest_pdf.stat().st_size} bytes)"
            )
            return str(latest_pdf)

        except Exception as e:
            print(f"❌ Error finding PDF factsheet: {str(e)}")
            import traceback
            traceback.print_exc()
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
        Send an email with PDF attachment using Supabase Edge Function.

        Args:
            to_email (str): Recipient email address (can be comma-separated)
            subject (str): Email subject
            html_content (str): HTML content
            attachments (List[Dict]): List of attachments (only PDFs will be processed)
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
                print("⚠️ Warning: Limited to first 50 recipients due to API constraints")

            # Filter to only include PDF attachments
            pdf_attachments = [
                att for att in attachments 
                if att.get('filename', '').lower().endswith('.pdf')
            ]
            if not pdf_attachments:
                print(
                    "⚠️ Warning: No PDF attachments found, "
                    "checking outputs/factsheets folder..."
                )
                # Check if we can find a PDF in the outputs/factsheets folder
                pdf_path = self.find_latest_pdf_factsheet()
                if pdf_path:
                    try:
                        pdf_file = Path(pdf_path)
                        with open(pdf_file, 'rb') as f:
                            pdf_content = f.read()
                        pdf_attachments = [{
                            "filename": pdf_file.name,
                            "content": pdf_content
                        }]
                        print(f"✅ Found PDF in outputs/factsheets: {pdf_file.name}")
                    except Exception as e:
                        print(f"❌ Error loading PDF from outputs/factsheets: {str(e)}")

            if not pdf_attachments:
                return {
                    "success": False,
                    "error": "No PDF attachments available to send",
                    "status_code": None
                }

            # Process only PDF attachments
            processed_attachments = []
            for idx, attachment in enumerate(pdf_attachments):
                print(
                    f"Processing PDF attachment {idx+1}/{len(pdf_attachments)}: "
                    f"{attachment.get('filename')}"
                )

                try:
                    # Check if we have binary content
                    if isinstance(attachment['content'], bytes):
                        # Convert binary content to base64
                        content_b64 = base64.b64encode(
                            attachment['content']).decode('utf-8')
                        print(
                            f"  - Binary content: {len(attachment['content'])} bytes "
                            f"→ base64 encoded"
                        )
                    else:
                        # Convert string content to bytes then base64
                        content_b64 = base64.b64encode(
                            attachment['content'].encode('utf-8')).decode('utf-8')
                        print(
                            f"  - Text content: {len(attachment['content'])} chars "
                            f"→ base64 encoded"
                        )

                    # Fix: ensure attachment uses correct format for Resend API
                    processed_attachment = {
                        "filename": attachment['filename'],
                        "content": content_b64
                    }

                    processed_attachments.append(processed_attachment)
                    print(f"  ✓ Successfully processed PDF attachment: {attachment['filename']}")

                except Exception as e:
                    print(
                        f"  ✗ Error processing PDF attachment {attachment.get('filename')}: "
                        f"{str(e)}"
                    )
                    import traceback
                    traceback.print_exc()

            # Prepare email payload using array format for multiple recipients
            # Convert recipients to the format expected by the Edge Function
            recipient_value = recipients if len(recipients) > 1 else recipients[0]

            # Fix: Convert tags from dict to array format if needed
            processed_tags = None
            if tags:
                if isinstance(tags, dict):
                    processed_tags = [{"name": k, "value": v} for k, v in tags.items()]
                else:
                    processed_tags = tags

            # Fix: ensure attachments are included in the proper format
            email_payload = {
                "to": recipient_value,
                "from": from_email,
                "subject": subject,
                "html": html_content,
                "attachments": processed_attachments  # Ensure this is included
            }

            # Debugging - show exactly what's being sent
            print(f"\n📋 Email payload structure:")
            print(
                f"  - to: {type(email_payload['to']).__name__} with "
                f"{len(recipient_value) if isinstance(recipient_value, list) else 1} "
                f"recipient(s)"
            )
            print(f"  - subject: {email_payload['subject'][:50]}...")
            print(f"  - html: {len(email_payload['html'])} characters")
            print(f"  - attachments: {len(email_payload['attachments'])} item(s)")

            # Add optional fields if provided
            if text_content:
                email_payload["text"] = text_content
                print(f"  - text: {len(email_payload['text'])} characters")
            if reply_to:
                email_payload["reply_to"] = reply_to
            if cc:
                email_payload["cc"] = cc
            if bcc:
                email_payload["bcc"] = bcc
            if processed_tags:
                # Sanitize tags - Resend API requires only ASCII letters, numbers, underscores, or dashes
                sanitized_tags = []
                for tag in processed_tags:
                    # Get tag name and value
                    tag_name = tag.get("name", "")
                    tag_value = tag.get("value", "")

                    # Only include tags with valid names and values
                    if tag_name and tag_value:
                        # Sanitize name and value - keep only allowed characters
                        import re
                        # Replace any disallowed characters with underscores
                        sanitized_name = re.sub(r'[^a-zA-Z0-9_-]', '_', str(tag_name))
                        sanitized_value = re.sub(r'[^a-zA-Z0-9_-]', '_', str(tag_value))

                        sanitized_tags.append({
                            "name": sanitized_name,
                            "value": sanitized_value
                        })

                # Only add tags if we have valid ones after sanitization
                if sanitized_tags:
                    email_payload["tags"] = sanitized_tags
                    print(f"  - tags: {len(sanitized_tags)} sanitized tag(s)")
                    for tag in sanitized_tags:
                        print(f"      - {tag['name']}: {tag['value']}")
                else:
                    print("  ⚠️ No valid tags after sanitization, omitting tags")

            print(
                f"📧 Sending email with {len(processed_attachments)} PDF attachment(s) "
                f"to: {', '.join(recipients)}"
            )
            print(
                f"  - Attachments: "
                f"{', '.join([a.get('filename', 'unnamed') for a in processed_attachments])}"
            )

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

            # Read PDF file content as binary
            with open(pdf_file, 'rb') as f:
                pdf_content = f.read()

            # Extract country and date from PDF filename
            # Expected format: Afghanistan_climate_policy_factsheet_20250528_193204.pdf
            filename = pdf_file.name

            # Extract country (everything before first underscore)
            country_name = filename.split('_')[0] if '_' in filename else "Unknown Country"

            # Extract date (after "factsheet_")
            date_str = "Unknown Date"
            if "factsheet_" in filename:
                date_part = filename.split("factsheet_")[1].split("_")[0]
                if len(date_part) >= 8 and date_part.isdigit():
                    # Format YYYYMMDD to YYYY-MM-DD
                    year = date_part[:4]
                    month = date_part[4:6]
                    day = date_part[6:8]
                    date_str = f"{year}-{month}-{day}"

            print(f"Extracted from PDF filename - Country: {country_name}, Date: {date_str}")

            # Create HTML email content with updated line about country updates
            html_content = f"""
            <html>
            <head>
                <style>
                    body {{ font-family: Arial, sans-serif; margin: 20px; }}
                    .header {{ background-color: #f8f9fa; padding: 20px; border-radius: 5px; }}
                    .content {{ margin: 20px 0; }}
                    .footer {{ font-size: 12px; color: #6c757d; margin-top: 30px; }}
                    .update-notice {{ background-color: #e7f3fe; border-left: 4px solid #2196F3; padding: 10px; margin: 15px 0; }}
                </style>
            </head>
            <body>
                <div class="header">
                    <h2>📄 Climate Policy Fact Sheet</h2>
                </div>
                
                <div class="update-notice">
                    <p>There has been an update for <strong>{country_name}</strong> on <strong>{date_str}</strong>. 
                    Please refer to the API for further clarification and details.</p>
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

            # Create text version with update notice
            text_content = f"""
Climate Policy Fact Sheet

UPDATE: There has been an update for {country_name} on {date_str}. Please refer to the API for further clarification and details.

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

            # Ensure filename ends with .pdf
            pdf_filename = pdf_file.name
            if not pdf_filename.lower().endswith('.pdf'):
                pdf_filename += '.pdf'

            # Create attachment with binary content - DO NOT encode to base64 here
            # send_email_with_attachments will handle the encoding
            attachments = [
                {
                    "filename": pdf_filename,
                    "content": pdf_content  # Pass binary content directly
                }
            ]

            # Email subject with country name
            subject = f"Climate Policy Fact Sheet - {country_name} ({date_str})"

            print(f"Attachment prepared: {pdf_filename} ({len(pdf_content)} bytes binary)")

            # Send email with PDF attachment - omit tags completely to avoid validation issues
            return self.send_email_with_attachments(
                to_email=to_email,
                subject=subject,
                html_content=html_content,
                attachments=attachments,
                from_email=from_email,
                text_content=text_content,
                tags=None
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
    Parse command line arguments for email configuration.
    """
    parser = argparse.ArgumentParser(
        description="Send Climate Policy Fact Sheet emails via Supabase Edge Functions",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Send the latest PDF factsheet to default recipients
  python send_email.py

  # Send to a specific recipient (in addition to default recipients)
  python send_email.py --to user@example.com

  # Send a specific PDF file
  python send_email.py --pdf-path outputs/factsheets/report.pdf

  # Dry run (show what would be sent without sending)
  python send_email.py --dry-run
        """
    )

    # Main arguments
    parser.add_argument(
        '--to', '-t',
        help='Additional recipient email address (comma-separated for multiple)'
    )

    parser.add_argument(
        '--pdf-path', '-p',
        help='Specific path to PDF file (defaults to latest in outputs/factsheets/)'
    )

    # Optional arguments
    parser.add_argument(
        '--from', '--sender', '-f',
        help='Sender email address (default: from .env file)'
    )

    parser.add_argument(
        '--dry-run', '-d',
        action='store_true',
        help='Show what would be sent without actually sending'
    )

    parser.add_argument(
        '--env-file',
        default='.env',
        help='Path to .env file (default: .env)'
    )

    return parser.parse_args()


def main():
    try:
        # Parse command-line arguments
        args = parse_arguments()

        # Initialize the email sender
        email_sender = SupabaseEmailSender()

        # Get PDF path (either specified or find latest)
        pdf_path = None
        if args.pdf_path:
            pdf_path = args.pdf_path
            if not Path(pdf_path).is_absolute():
                pdf_path = project_root / pdf_path
        else:
            pdf_path = email_sender.find_latest_pdf_factsheet()

        if not pdf_path:
            print("❌ No PDF factsheet found to send")
            print("Please generate PDF factsheets before attempting to send emails.")
            sys.exit(1)

        # Get recipients (default + any additional)
        default_recipients = email_sender.get_default_recipient()
        additional_recipients = args.to if args.to else ""

        # Combine recipients, ensuring no duplicates
        all_recipients = set(
            [r.strip() for r in default_recipients.split(',') if r.strip()]
        )
        if additional_recipients:
            all_recipients.update(
                [r.strip() for r in additional_recipients.split(',') if r.strip()]
            )

        recipient_str = ",".join(all_recipients)

        if args.dry_run:
            print("\n=== DRY RUN - FACTSHEET EMAIL WOULD BE SENT ===")
            print(f"To: {recipient_str}")
            print(f"PDF: {pdf_path}")
            print(
                f"From: {getattr(args, 'from', None) or email_sender.get_default_sender()}"
            )
            print("=== DRY RUN COMPLETE - NO EMAIL SENT ===")
            return

        # Send factsheet email
        result = email_sender.send_factsheet_email(
            to_email=recipient_str,
            pdf_path=pdf_path,
            from_email=getattr(args, 'from', None)
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
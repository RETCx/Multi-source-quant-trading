"""
Notification Module for Quant Trading Pipeline.
Supports: LINE Notify + Email (Gmail SMTP)

Setup:
  LINE:  https://notify-bot.line.me/my/ -> Generate Token -> Add to .env
  EMAIL: Gmail -> เปิด 2-Step Verification -> สร้าง App Password
         https://myaccount.google.com/apppasswords
         Add SMTP_EMAIL and SMTP_PASSWORD to .env
"""
import os
from dotenv import load_dotenv
load_dotenv()

import smtplib
import base64
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build

LINE_NOTIFY_API = "https://notify-api.line.me/api/notify"


# =============================================================================
# LINE NOTIFY
# =============================================================================
def send_line_notify(message: str, token: str = None) -> bool:
    """Send a message via LINE Notify."""
    import requests
    
    token = token or os.environ.get("LINE_NOTIFY_TOKEN")
    
    if not token or token == "YOUR_TOKEN_HERE":
        print("[NOTIFY] LINE_NOTIFY_TOKEN not set. Skipping LINE.")
        return False
    
    headers = {"Authorization": f"Bearer {token}"}
    data = {"message": message}
    
    try:
        resp = requests.post(LINE_NOTIFY_API, headers=headers, data=data, timeout=10)
        if resp.status_code == 200:
            print("[NOTIFY] LINE message sent successfully.")
            return True
        else:
            print(f"[NOTIFY] LINE API error: {resp.status_code} {resp.text}")
            return False
    except Exception as e:
        print(f"[NOTIFY] Failed to send LINE message: {e}")
        return False


# =============================================================================
# EMAIL (Gmail API)
# =============================================================================
SCOPES = ["https://www.googleapis.com/auth/gmail.send"]

def get_gmail_service():
    """Authenticate and return the Gmail API service."""
    creds = None
    token_path = "token.json"
    creds_path = "credentials.json"
    
    if os.path.exists(token_path):
        creds = Credentials.from_authorized_user_file(token_path, SCOPES)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            try:
                creds.refresh(Request())
            except Exception as e:
                print(f"[NOTIFY] Token refresh failed (invalid/expired). Deleting {token_path} to re-authenticate.")
                os.remove(token_path)
                creds = None
                
        if not creds:
            if not os.path.exists(creds_path):
                print(f"[NOTIFY] '{creds_path}' not found. Please download it from Google Cloud Console.")
                return None
            print("[NOTIFY] Opening browser for Google Login...")
            flow = InstalledAppFlow.from_client_secrets_file(creds_path, SCOPES)
            creds = flow.run_local_server(port=0)
            
        with open(token_path, "w") as token:
            token.write(creds.to_json())
            
    return build("gmail", "v1", credentials=creds)

def send_email(subject: str, body: str,
               to_email: str = None,
               from_email: str = None) -> bool:
    """
    Send an email via Gmail API.
    
    Args:
        subject: Email subject line
        body: Email body (plain text)
        to_email: Recipient email. If None, reads SMTP_TO_EMAIL from .env
        from_email: Sender email (not strictly needed for Gmail API as it uses authenticated user, but useful for display)
    """
    from_email = from_email or os.environ.get("SMTP_EMAIL", "me")
    to_email_str = to_email or os.environ.get("SMTP_TO_EMAIL", from_email)
    
    if not to_email_str:
        print("[NOTIFY] No recipient email specified. Skipping email.")
        return False
        
    # Split by comma to support multiple emails
    to_email_list = [email.strip() for email in to_email_str.split(',') if email.strip()]
    
    try:
        service = get_gmail_service()
        if not service:
            return False
            
        msg = MIMEMultipart("alternative")
        msg["Subject"] = subject
        from email.utils import formataddr
        msg["From"] = formataddr(("Quant Trading Bot", from_email))
        msg["To"] = ", ".join(to_email_list)
        
        # Plain text version
        msg.attach(MIMEText(body, "plain", "utf-8"))
        
        # HTML version (nicer formatting)
        html_body = _format_email_html(subject, body)
        msg.attach(MIMEText(html_body, "html", "utf-8"))
        
        # Encode as base64
        raw_message = base64.urlsafe_b64encode(msg.as_bytes()).decode("utf-8")
        body_dict = {'raw': raw_message}
        
        message = service.users().messages().send(userId="me", body=body_dict).execute()
        
        print(f"[NOTIFY] Gmail API Email sent to: {', '.join(to_email_list)} (Message Id: {message['id']})")
        return True
        
    except Exception as e:
        print(f"[NOTIFY] Gmail API Email failed: {e}")
        return False


def _format_email_html(subject: str, body: str) -> str:
    """Convert plain text body to a styled HTML email."""
    # Extract action from subject to color code
    action_color = "#4CAF50" # default green (BUY)
    if "[SELL" in subject or "SHORT" in subject:
        action_color = "#F44336" # Red
    elif "[HOLD" in subject:
        action_color = "#9E9E9E" # Gray

    lines = body.split('\n')
    formatted_lines = []
    for line in lines:
        if not line.strip():
            continue
        if "ACTION" in line:
            formatted_lines.append(f'<div style="margin-top: 25px; font-size: 20px; font-weight: 800; color: {action_color}; text-align: center; letter-spacing: 1px;">{line}</div>')
        elif "---" in line or "===" in line:
            formatted_lines.append(f'<hr style="border: 0; height: 1px; background: #EAEAEA; margin: 20px 0;">')
        elif ":" in line:
            parts = line.split(":", 1)
            formatted_lines.append(f'<div style="margin: 10px 0; display: flex; justify-content: space-between; font-size: 15px;"><span style="color: #666; font-weight: 600;">{parts[0]}:</span><span style="color: #111; font-weight: 500;">{parts[1].strip()}</span></div>')
        else:
            formatted_lines.append(f'<div style="color: #111; font-weight: 700; font-size: 18px; margin-bottom: 10px; text-align: center;">{line}</div>')
            
    content_html = "".join(formatted_lines)
    
    return f"""
    <!DOCTYPE html>
    <html>
    <head>
    <meta charset="utf-8">
    </head>
    <body style="font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif; background-color: #F4F7F6; margin: 0; padding: 40px 15px;">
      <div style="max-width: 480px; margin: 0 auto; background: #FFFFFF; border-radius: 12px; box-shadow: 0 8px 24px rgba(0,0,0,0.06); overflow: hidden;">
        <div style="background-color: {action_color}; color: #FFFFFF; padding: 24px 30px; text-align: center;">
          <h2 style="margin: 0; font-size: 22px; font-weight: 700; letter-spacing: 0.5px;">{subject}</h2>
        </div>
        <div style="padding: 32px 40px;">
          {content_html}
        </div>
        <div style="text-align: center; padding: 20px; font-size: 12px; color: #999; background: #FAFAFA; border-top: 1px solid #F0F0F0;">
          Automated Quant AI System <br> {datetime.now().strftime('%Y-%m-%d %H:%M')}
        </div>
      </div>
    </body>
    </html>
    """


# =============================================================================
# UNIFIED: ส่งทุกช่องทางพร้อมกัน
# =============================================================================
def notify_all(subject: str, body: str):
    """
    Send notification to ALL configured channels (LINE + Email).
    Channels that are not configured will be silently skipped.
    """
    print("[NOTIFY] Sending notifications...")
    send_line_notify(f"\n{subject}\n{body}")
    send_email(subject=subject, body=body)


# =============================================================================
# MESSAGE FORMATTING
# =============================================================================
def format_signal_message(stock: str, predictions: dict, action: str,
                          best_horizon: int, best_confidence: float) -> str:
    """Format prediction results into a clean notification message."""
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    
    lines = [
        f"STOCK: {stock} | {now}",
        f"{'-'*30}",
    ]
    
    for target_name, (conf, direction) in predictions.items():
        lines.append(f"{target_name}: {conf:.1%} {direction}")
    
    lines.append(f"{'-'*30}")
    lines.append(f"ACTION: {action}")
    lines.append(f"Horizon: {best_horizon}D | Conf: {best_confidence:.1%}")
    
    return "\n".join(lines)

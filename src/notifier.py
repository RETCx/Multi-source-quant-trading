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
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime

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
# EMAIL (Gmail SMTP)
# =============================================================================
def send_email(subject: str, body: str,
               to_email: str = None,
               from_email: str = None,
               password: str = None,
               smtp_server: str = "smtp.gmail.com",
               smtp_port: int = 587) -> bool:
    """
    Send an email via SMTP (default: Gmail).
    
    Args:
        subject: Email subject line
        body: Email body (plain text)
        to_email: Recipient email. If None, sends to self (from .env)
        from_email: Sender email. If None, reads SMTP_EMAIL from .env
        password: App password. If None, reads SMTP_PASSWORD from .env
    
    Returns:
        True if sent successfully.
    
    Gmail Setup:
        1. เปิด 2-Step Verification ใน Google Account
        2. ไปที่ https://myaccount.google.com/apppasswords
        3. สร้าง App Password (เลือก "Mail")
        4. Copy password 16 ตัวมาใส่ใน .env เป็น SMTP_PASSWORD
    """
    from_email = from_email or os.environ.get("SMTP_EMAIL")
    password = password or os.environ.get("SMTP_PASSWORD")
    to_email = to_email or os.environ.get("SMTP_TO_EMAIL", from_email)
    
    if not from_email or not password:
        print("[NOTIFY] SMTP_EMAIL or SMTP_PASSWORD not set. Skipping email.")
        return False
    
    try:
        msg = MIMEMultipart("alternative")
        msg["Subject"] = subject
        msg["From"] = from_email
        msg["To"] = to_email
        
        # Plain text version
        msg.attach(MIMEText(body, "plain", "utf-8"))
        
        # HTML version (nicer formatting)
        html_body = _format_email_html(subject, body)
        msg.attach(MIMEText(html_body, "html", "utf-8"))
        
        with smtplib.SMTP(smtp_server, smtp_port) as server:
            server.starttls()
            server.login(from_email, password)
            server.sendmail(from_email, to_email, msg.as_string())
        
        print(f"[NOTIFY] Email sent to {to_email}")
        return True
        
    except Exception as e:
        print(f"[NOTIFY] Email failed: {e}")
        return False


def _format_email_html(subject: str, body: str) -> str:
    """Convert plain text body to a styled HTML email."""
    # Escape and convert newlines
    import html
    escaped = html.escape(body)
    content_html = escaped.replace("\n", "<br>")
    
    return f"""
    <html>
    <body style="font-family: 'Segoe UI', Arial, sans-serif; background: #1a1a2e; padding: 20px;">
      <div style="max-width: 500px; margin: auto; background: #16213e; border-radius: 12px; 
                  padding: 24px; color: #e0e0e0; border: 1px solid #0f3460;">
        <h2 style="color: #00d2ff; margin-top: 0;">📊 {html.escape(subject)}</h2>
        <pre style="font-family: 'Consolas', monospace; font-size: 13px; 
                    line-height: 1.6; white-space: pre-wrap; color: #c8d6e5;">
{escaped}
        </pre>
        <hr style="border-color: #0f3460;">
        <p style="font-size: 11px; color: #666;">
          Sent by Quant Trading Bot | {datetime.now().strftime('%Y-%m-%d %H:%M')}
        </p>
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
    
    icon = {"BUY": "🟢", "SELL": "🔴", "HOLD": "⚪"}.get(action, "❓")
    
    lines = [
        f"{'='*30}",
        f"📊 {stock} | {now}",
        f"{'='*30}",
    ]
    
    for target_name, (conf, direction) in predictions.items():
        arrow = "▲" if direction == "UP" else "▼"
        lines.append(f"  {target_name}: {conf:.1%} {arrow} {direction}")
    
    lines.append(f"{'─'*30}")
    lines.append(f"{icon} ACTION: {action}")
    lines.append(f"   Horizon: {best_horizon}D | Conf: {best_confidence:.1%}")
    lines.append(f"{'='*30}")
    
    return "\n".join(lines)

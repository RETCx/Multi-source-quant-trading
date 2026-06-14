import os
from dotenv import load_dotenv

load_dotenv()

from src.notifier import send_email

print("Testing Email Notification...")
success = send_email(
    subject="[TEST] Quant Trading Bot Notification",
    body="If you receive this email, your SMTP configuration in .env is working perfectly!\n\nHappy Trading!"
)

if success:
    print("\nSuccess! Please check your inbox (and spam folder just in case).")
else:
    print("\nFailed to send email. Please double-check your SMTP_EMAIL and SMTP_PASSWORD in the .env file.")

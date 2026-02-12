#!/usr/bin/env python3
"""
test_email.py

Quick test to verify email alert works.
"""

from email_alert import send_incident_email
from datetime import datetime

# Create fake test incidents
test_incidents = [
    {
        "incident_type": "collision",
        "severity": 0.85,
        "location": "patterson",
        "vehicles": [101, 102],
        "timestamp": datetime.utcnow().isoformat()
    },
    {
        "incident_type": "near_miss",
        "severity": 0.62,
        "location": "patterson",
        "vehicles": [103, 104, 105],
        "timestamp": datetime.utcnow().isoformat()
    }
]

print("🚨 Testing email alert...")
print(f"Sending test email with {len(test_incidents)} fake incidents...")

try:
    send_incident_email(test_incidents)
    print("✅ Email sent successfully!")
    print(f"📧 Check {test_incidents[0].get('location')} for the alert email")
except Exception as e:
    print(f"❌ Email failed: {e}")
    import traceback
    traceback.print_exc()
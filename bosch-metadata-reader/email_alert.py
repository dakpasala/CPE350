import smtplib
import configparser
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime
from typing import List, Dict


# =========================
# Config
# =========================

RECIPIENT = "dak.pasala@gmail.com"
SUBJECT = "ALERT: INCIDENT DETECTED"


def _get_creds():
    config = configparser.ConfigParser()
    config.read("connection.ini")
    return {
        "sender": config["DEFAULT"]["email"],          # gmail address
        "password": config["DEFAULT"]["app_password"], # the app password
    }


# =========================
# Format incidents
# =========================

def format_incidents(incidents: List[Dict]) -> str:
    """
    Formats the incidents list into a clean HTML table for the email body.
    """
    now = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")

    html = f"""
    <h2 style="color: #d32f2f;">🚨 INCIDENT ALERT</h2>
    <p><strong>Detected at:</strong> {now}</p>
    <p><strong>Total incidents:</strong> {len(incidents)}</p>
    <hr>

    <table border="1" cellpadding="8" cellspacing="0" style="border-collapse: collapse; width: 100%; font-family: Arial, sans-serif;">
        <thead>
            <tr style="background-color: #d32f2f; color: white;">
                <th>#</th>
                <th>Type</th>
                <th>Severity</th>
                <th>Location</th>
                <th>Vehicles Involved</th>
                <th>Timestamp</th>
            </tr>
        </thead>
        <tbody>
    """

    for i, inc in enumerate(incidents, 1):
        severity = inc.get("severity", 0)

        # Color code severity
        if severity >= 0.8:
            sev_color = "#d32f2f"   # red
        elif severity >= 0.5:
            sev_color = "#f57c00"   # orange
        else:
            sev_color = "#388e3c"   # green

        vehicles = ", ".join(str(v) for v in inc.get("vehicles", []))

        html += f"""
            <tr>
                <td style="text-align: center;">{i}</td>
                <td><strong>{inc.get('incident_type', 'unknown')}</strong></td>
                <td style="color: {sev_color}; font-weight: bold;">{severity:.2f}</td>
                <td>{inc.get('location', 'unknown')}</td>
                <td>{vehicles}</td>
                <td>{inc.get('timestamp', 'N/A')}</td>
            </tr>
        """

    html += """
        </tbody>
    </table>
    """

    return html


# =========================
# Send email
# =========================

def send_incident_email(incidents: List[Dict]):
    """
    Sends an incident alert email. Call this when len(incidents) > 0.
    """
    if not incidents:
        return

    creds = _get_creds()

    msg = MIMEMultipart("alternative")
    msg["Subject"] = SUBJECT
    msg["From"]    = creds["sender"]
    msg["To"]      = RECIPIENT

    html_body = format_incidents(incidents)
    msg.attach(MIMEText(html_body, "html"))

    try:
        with smtplib.SMTP("smtp.gmail.com", 587) as server:
            server.ehlo()
            server.starttls()
            server.login(creds["sender"], creds["password"])
            server.sendmail(creds["sender"], RECIPIENT, msg.as_string())

        print(f"[EMAIL] Alert sent successfully | incidents={len(incidents)}")

    except Exception as e:
        print(f"[EMAIL] Failed to send alert: {e}")
import requests

API_URL = "http://127.0.0.1:8000/combined-stats"

def send_to_api(doc):
    try:
        r = requests.post(API_URL, json=doc, timeout=1)
        r.raise_for_status()
    except Exception as e:
        print("❌ API send failed:", e)
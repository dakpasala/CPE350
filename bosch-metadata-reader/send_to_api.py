import requests
import datetime
import numpy as np
import pandas as pd

API_URL = "http://127.0.0.1:8000/raw-vehicles"

def _json_safe(obj):
    if isinstance(obj, (datetime.datetime, datetime.date)):
        return obj.isoformat()
    if isinstance(obj, np.datetime64):
        return pd.to_datetime(obj).isoformat()
    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, np.generic):
        return obj.item()
    return obj

def send_to_api(roadObjectData, *_, **__):
    try:
        payload = [_json_safe(roadObjectData)]
        print("📤 Sending to API:")
        print(payload[0])

        r = requests.post(API_URL, json=payload, timeout=1)
        r.raise_for_status()
    except Exception as e:
        print(f"⚠️ Data push failed: {e}")

#!/usr/bin/env python3
"""
incident_viewer.py

Incident video viewer - Browse and watch accident videos from MongoDB.
Fetches incidents with associated videos and displays them.
"""

import os
import requests
from datetime import datetime
from dotenv import load_dotenv

import dash
from dash import Dash, dcc, html, Input, Output, State
from dash.exceptions import PreventUpdate

load_dotenv()

# =========================
# Configuration
# ========================= 

API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")
HOST = os.getenv("VIEWER_HOST", "127.0.0.1")
PORT = int(os.getenv("VIEWER_PORT", "8051"))

# Caltrans Colors
CALTRANS_BLUE = "#003366"
CALTRANS_GREEN = "#007B5F"

TIME_RANGES = {
    "Last Hour": 60,
    "Last 6 Hours": 360,
    "Last 12 Hours": 720,
    "Last Day": 1440,
    "Last Week": 10080,
    "Last Month": 43200,
}


# =========================
# API Calls
# =========================

def get_recent_videos(limit=10, location=None):
    try:
        params = {"limit": limit}
        if location:
            params["location"] = location
        response = requests.get(f"{API_BASE_URL}/videos", params=params, timeout=10)
        if response.status_code == 200:
            return response.json().get("videos", [])
        return []
    except Exception as e:
        print(f"⚠️ Error fetching videos: {e}")
        return []


def get_videos_by_timerange(minutes, location=None):
    try:
        params = {"minutes": minutes}
        if location:
            params["location"] = location
        response = requests.get(f"{API_BASE_URL}/videos/timerange", params=params, timeout=10)
        if response.status_code == 200:
            return response.json().get("videos", [])
        return []
    except Exception as e:
        print(f"⚠️ Error fetching videos: {e}")
        return []


def get_incidents_by_timerange(minutes, location=None):
    try:
        params = {"minutes": minutes}
        if location:
            params["location"] = location
        response = requests.get(f"{API_BASE_URL}/incidents/timerange", params=params, timeout=10)
        if response.status_code == 200:
            return response.json().get("incidents", [])
        return []
    except Exception as e:
        print(f"⚠️ Error fetching incidents: {e}")
        return []


def get_video_for_incident(incident_id):
    try:
        response = requests.get(f"{API_BASE_URL}/videos/incident/{incident_id}", timeout=10)
        if response.status_code == 200:
            data = response.json()
            if data.get("status") == "success":
                return data.get("video")
        return None
    except Exception as e:
        print(f"⚠️ Error fetching video for incident: {e}")
        return None


# =========================
# Helper Functions
# =========================

def format_timestamp(ts_str):
    try:
        if isinstance(ts_str, str):
            ts = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
        else:
            ts = ts_str
        return ts.strftime("%Y-%m-%d %H:%M:%S")
    except:
        return str(ts_str)


def severity_color(severity):
    if severity >= 0.7:
        return "#f5576c"
    elif severity >= 0.4:
        return "#ff9800"
    return "#4CAF50"


def build_video_card(video):
    video_id = video.get("_id")
    filename = video.get("filename", "Unknown")
    camera = video.get("camera", "Unknown")
    timestamp = format_timestamp(video.get("timestamp"))
    size_mb = video.get("size_bytes", 0) / 1024 / 1024
    incident_count = len(video.get("incident_ids", []))

    return html.Div([
        html.Div([
            html.Div("📹", style={"fontSize": "24px", "marginRight": "12px"}),
            html.Div([
                html.Div(filename, className="card-title", style={"fontWeight": "700", "fontSize": "15px", "color": "#333", "marginBottom": "2px"}),
                html.Div(f"Camera: {camera}", className="card-meta", style={"fontSize": "13px", "color": "#666"}),
            ], style={"flex": "1"}),
            html.Div(
                f"🚨 {incident_count}" if incident_count > 0 else "✓ Clean",
                style={
                    "fontSize": "13px", "fontWeight": "700",
                    "color": "#f5576c" if incident_count > 0 else "#4CAF50",
                    "padding": "4px 10px", "borderRadius": "20px",
                    "background": "rgba(245,87,108,0.1)" if incident_count > 0 else "rgba(76,175,80,0.1)",
                }
            ),
        ], style={"display": "flex", "alignItems": "center", "marginBottom": "12px"}),
        html.Div([
            html.Span(timestamp, className="card-meta", style={"fontSize": "13px", "color": "#666", "marginRight": "16px"}),
            html.Span(f"{size_mb:.2f} MB", className="card-meta", style={"fontSize": "13px", "color": "#666"}),
        ], style={"marginBottom": "14px"}),
        html.Button(
            "▶ Watch Video",
            id={"type": "watch-btn", "index": video_id},
            n_clicks=0,
            style={
                "width": "100%", "padding": "10px", "fontSize": "14px", "fontWeight": "600",
                "borderRadius": "8px", "border": "none",
                "background": CALTRANS_BLUE,  # changed from purple gradient
                "color": "white", "cursor": "pointer", "boxShadow": "0 4px 12px rgba(0,51,102,0.28)",
            }
        ),
    ], className="video-card", style={
        "background": "white", "border": "2px solid #eee", "borderRadius": "12px",
        "padding": "16px", "boxShadow": "0 4px 16px rgba(0,0,0,0.08)",
        "transition": "transform 0.2s ease, box-shadow 0.2s ease",
    })


def build_incident_card(incident):
    incident_id = str(incident.get("_id"))
    incident_type = incident.get("incident_type", "Unknown")
    severity = incident.get("severity", 0)
    timestamp = format_timestamp(incident.get("timestamp"))
    vehicles = incident.get("vehicles", [])
    location = incident.get("location", "Unknown")
    sev_color = severity_color(severity)
    border = "#f5576c" if severity >= 0.7 else ("#ff9800" if severity >= 0.4 else "#eee")

    return html.Div([
        html.Div([
            html.Div("🚨", style={"fontSize": "24px", "marginRight": "12px"}),
            html.Div([
                html.Div(incident_type.upper(), className="card-title", style={"fontWeight": "700", "fontSize": "15px", "color": "#333", "marginBottom": "2px"}),
                html.Div(location, className="card-meta", style={"fontSize": "13px", "color": "#666"}),
            ], style={"flex": "1"}),
            html.Div(f"Sev {severity:.2f}", style={
                "fontSize": "13px", "fontWeight": "700", "color": sev_color,
                "padding": "4px 10px", "borderRadius": "20px", "background": "rgba(245,87,108,0.1)",
            }),
        ], style={"display": "flex", "alignItems": "center", "marginBottom": "12px"}),
        html.Div([
            html.Span(timestamp, className="card-meta", style={"fontSize": "13px", "color": "#666", "marginRight": "16px"}),
            html.Span(f"{len(vehicles)} vehicles", className="card-meta", style={"fontSize": "13px", "color": "#666"}),
        ], style={"marginBottom": "14px"}),
        html.Button(
            "🎥 Watch Video",
            id={"type": "incident-video-btn", "index": incident_id},
            n_clicks=0,
            style={
                "width": "100%", "padding": "10px 20px", "fontSize": "14px", "fontWeight": "600",
                "borderRadius": "8px", "border": "none",
                "background": CALTRANS_GREEN,  # changed from pink gradient
                "color": "white", "cursor": "pointer", "boxShadow": "0 4px 12px rgba(0,123,95,0.28)",
            }
        ),
    ], className="incident-card", style={
        "background": "white", "border": f"2px solid {border}", "borderRadius": "12px",
        "padding": "16px", "boxShadow": "0 4px 16px rgba(0,0,0,0.08)", "transition": "transform 0.2s ease",
    })


# =========================
# Shared theme helpers
# =========================

TOGGLE_CSS = """
body { margin: 0; padding: 0; overflow-x: hidden; }

/* Main container themed via CSS — no inline style override */
.main-container {
    padding: 30px;
    background: #f5f7fa;
    min-height: 100vh;
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
    color: #333;
    transition: background 0.2s ease, color 0.2s ease;
}
[data-theme="dark"] .main-container {
    background: #1a1a1a !important;
    color: #e0e0e0 !important;
}

.theme-toggle-btn {
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 8px 18px;
    border-radius: 4px;
    border: 2px solid rgba(255,255,255,0.2);
    background: rgba(255,255,255,0.1);
    cursor: pointer;
    font-size: 14px;
    font-weight: 600;
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    color: white;
    box-shadow: none;
    transition: all 0.25s ease;
    user-select: none;
    white-space: nowrap;
}
.theme-toggle-btn:hover {
    background: rgba(255,255,255,0.2);
}
.theme-toggle-btn .toggle-track {
    width: 38px;
    height: 22px;
    border-radius: 11px;
    background: #ccc;
    position: relative;
    transition: background 0.25s ease;
    flex-shrink: 0;
}
.theme-toggle-btn .toggle-track.active {
    background: #007B5F;
}
.theme-toggle-btn .toggle-thumb {
    width: 18px;
    height: 18px;
    border-radius: 50%;
    background: white;
    position: absolute;
    top: 2px;
    left: 2px;
    transition: transform 0.25s ease;
    box-shadow: 0 1px 4px rgba(0,0,0,0.2);
}
.theme-toggle-btn .toggle-track.active .toggle-thumb {
    transform: translateX(16px);
}
.theme-toggle-btn.dark-mode {
    background: #2d2d2d;
    border-color: rgba(255,255,255,0.15);
    color: #e0e0e0;
    box-shadow: 0 2px 8px rgba(0,0,0,0.4);
}

/* Tab styles */
.tab-btn {
    padding: 12px 24px;
    border: none;
    border-bottom: 3px solid transparent;
    background: transparent;
    font-size: 15px;
    font-weight: 600;
    cursor: pointer;
    transition: all 0.2s ease;
    color: #666;
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
}
.tab-btn.active {
    border-bottom-color: #003366;  /* changed from #667eea */
    color: #003366;                 /* changed from #667eea */
}
.tab-btn:hover {
    color: #003366;                 /* changed from #667eea */
}

/* Dark mode overrides via data-theme attribute */
[data-theme="dark"] #tab-bar {
    background: #2d2d2d !important;
    border-bottom: 2px solid #404040 !important;
}
[data-theme="dark"] #tab-content {
    background: #2d2d2d !important;
    color: #e0e0e0;
}
[data-theme="dark"] .tab-btn {
    color: #aaa;
}
[data-theme="dark"] .tab-btn.active {
    color: #7fb3ff; /* brighter blue for dark mode readability */
}
[data-theme="dark"] .tab-section-title {
    color: #e0e0e0 !important;
}
[data-theme="dark"] .tab-section-subtitle {
    color: #aaa !important;
}
[data-theme="dark"] .tab-label {
    color: #aaa !important;
}
[data-theme="dark"] .search-text-input {
    background: #1a1a1a !important;
    color: #e0e0e0 !important;
    border-color: #404040 !important;
}
[data-theme="dark"] .empty-state {
    color: #aaa !important;
    border-color: #404040 !important;
}
[data-theme="dark"] .video-card, [data-theme="dark"] .incident-card {
    background: #2d2d2d !important;
    border-color: #404040 !important;
}
[data-theme="dark"] .card-title { color: #e0e0e0 !important; }
[data-theme="dark"] .card-meta { color: #aaa !important; }

/* Dark mode for Dash dropdowns */
[data-theme="dark"] .Select-control,
[data-theme="dark"] .dash-dropdown .Select-control {
    background: #1a1a1a !important;
    border-color: #404040 !important;
    color: #e0e0e0 !important;
}
[data-theme="dark"] .Select-value-label,
[data-theme="dark"] .Select-placeholder,
[data-theme="dark"] .dash-dropdown .Select-value-label {
    color: #e0e0e0 !important;
}
[data-theme="dark"] .Select-menu-outer,
[data-theme="dark"] .dash-dropdown .Select-menu-outer {
    background: #2d2d2d !important;
    border-color: #404040 !important;
}
[data-theme="dark"] .VirtualizedSelectOption {
    background: #2d2d2d !important;
    color: #e0e0e0 !important;
}
[data-theme="dark"] .VirtualizedSelectOption:hover,
[data-theme="dark"] .VirtualizedSelectFocusedOption {
    background: #404040 !important;
}

/* Dark mode for video modal */
[data-theme="dark"] .video-modal-overlay .video-modal-body {
    background: #2d2d2d !important;
}
[data-theme="dark"] #video-player-content {
    background: #2d2d2d !important;
}
[data-theme="dark"] .video-modal-overlay > div {
    background: #2d2d2d !important;
}
[data-theme="dark"] #video-player-content p {
    color: #aaa !important;
}

/* Video modal */
.video-modal-overlay {
    display: none;
    position: fixed;
    top: 0; left: 0;
    width: 100vw; height: 100vh;
    background: rgba(0,0,0,0.75);
    z-index: 9999;
    align-items: center;
    justify-content: center;
    backdrop-filter: blur(4px);
}
.video-modal-overlay.open {
    display: flex;
}

/* Search inputs */
.search-input {
    width: 100%;
    padding: 10px 14px;
    border-radius: 8px;
    border: 2px solid #e0e0e0;
    font-size: 14px;
    font-family: inherit;
    transition: border-color 0.2s;
    box-sizing: border-box;
}
.search-input:focus {
    outline: none;
    border-color: #667eea;
}

.search-btn {
    padding: 10px 24px;
    border: none;
    border-radius: 8px;
    font-size: 14px;
    font-weight: 600;
    cursor: pointer;
    transition: all 0.2s ease;
    font-family: inherit;
}
.search-btn:hover {
    transform: translateY(-1px);
    box-shadow: 0 4px 12px rgba(0,0,0,0.15);
}
"""

# FIX 1: No forced dark-mode in the inline script — theme is applied after React mounts.
TOGGLE_JS = """
window.DASHBOARD_THEME_KEY = 'dashboard_theme';
(function() {
    var match = document.cookie.match('(?:^|; )dashboard_theme=([^;]*)');
    var saved = match ? match[1] : null;
    if (saved) document.documentElement.setAttribute('data-theme', saved);
})();
"""

INDEX_STRING = '''
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
        {%css%}
        <style>''' + TOGGLE_CSS + '''</style>
        <script>''' + TOGGLE_JS + '''</script>
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
'''


def make_theme_toggle(toggle_id="theme-toggle"):
    return html.Div(
        id=f"{toggle_id}-wrapper",
        children=[
            dcc.Checklist(
                id=toggle_id,
                options=[{"label": "", "value": "dark"}],
                value=[],
                style={"display": "none"},
            ),
            html.Button(
                id=f"{toggle_id}-btn",
                children=[
                    html.Div([
                        html.Div(className="toggle-thumb"),
                    ], id=f"{toggle_id}-track", className="toggle-track"),
                    html.Span("Theme: Light", id=f"{toggle_id}-label"),
                ],
                className="theme-toggle-btn",
                n_clicks=0,
            ),
        ],
        style={"position": "fixed", "top": "20px", "right": "20px", "zIndex": 10000}
    )


# =========================
# Dash App
# =========================

app = Dash(__name__)
app.title = "Incident Video Viewer"
app.index_string = INDEX_STRING


def make_layout():
    return html.Div(id="main-container", className="main-container", children=[
        make_theme_toggle("theme-toggle"),

        # ---- Header ----
        html.Div([
            html.Div([
                html.H2("TRAFFIC INCIDENT VIEWER", style={
                    "color": "white",
                    "margin": "0",
                    "fontSize": "30px",
                    "fontWeight": "800",
                    "letterSpacing": "1.2px"
                }),
                html.Div(style={
                    "width": "44px",
                    "height": "4px",
                    "background": CALTRANS_GREEN,
                    "margin": "10px 0 8px 0"
                }),
                html.P("DIVISION OF TRAFFIC OPERATIONS • INCIDENT REVIEW", style={
                    "color": "rgba(255,255,255,0.9)",
                    "margin": "0",
                    "fontSize": "12px",
                    "letterSpacing": "1.4px",
                    "fontWeight": "500"
                }),
            ], style={"flex": "1"}),

            html.Div([
                html.A(
                    html.Button("Home", style={
                        "fontSize": "14px",
                        "padding": "10px 18px",
                        "background": "white",
                        "color": CALTRANS_BLUE,
                        "border": f"1px solid {CALTRANS_BLUE}",
                        "borderRadius": "4px",
                        "cursor": "pointer",
                        "fontWeight": "700",
                    }),
                    href="http://127.0.0.1:8050",
                ),
                html.A(
                    html.Button("Live Feed", style={
                        "fontSize": "14px",
                        "padding": "10px 18px",
                        "background": CALTRANS_BLUE,
                        "color": "white",
                        "border": f"1px solid {CALTRANS_BLUE}",
                        "borderRadius": "4px",
                        "cursor": "pointer",
                        "fontWeight": "700",
                    }),
                    href="http://127.0.0.1:8053",
                ),
                html.A(
                    html.Button("Traffic Heatmap", style={
                        "fontSize": "14px",
                        "padding": "10px 18px",
                        "background": CALTRANS_GREEN,
                        "color": "white",
                        "border": f"1px solid {CALTRANS_GREEN}",
                        "borderRadius": "4px",
                        "cursor": "pointer",
                        "fontWeight": "700",
                    }),
                    href="http://127.0.0.1:8052",
                ),
            ], style={"display": "flex", "gap": "8px", "marginRight": "18px", "alignItems": "center"}),
        ], style={
            "display": "flex",
            "alignItems": "center",
            "justifyContent": "space-between",
            "padding": "28px 30px",
            "background": CALTRANS_BLUE,
            "borderRadius": "4px",
            "marginBottom": "24px",
            "boxShadow": "0 4px 15px rgba(0,0,0,0.1)",
            "borderBottom": f"6px solid {CALTRANS_GREEN}"
        }),

        # ---- Tab bar ----
        html.Div([
            html.Div([
                html.Button("🚨 Incidents by Time", id="tab-incidents-btn", className="tab-btn active", n_clicks=0),
                html.Button("📹 All Videos", id="tab-videos-btn", className="tab-btn", n_clicks=0),
                html.Button("🎬 Latest Video", id="tab-latest-btn", className="tab-btn", n_clicks=0),
            ], style={"display": "flex", "gap": "4px"}),
        ], id="tab-bar", style={
            "background": "white",
            "borderRadius": "12px 12px 0 0",
            "padding": "0 20px",
            "borderBottom": "2px solid #eee",
            "boxShadow": "0 2px 8px rgba(0,0,0,0.05)",
        }),

        # ---- Tab content panels ----
        html.Div(id="tab-content", style={
            "background": "white",
            "borderRadius": "0 0 12px 12px",
            "padding": "24px",
            "boxShadow": "0 4px 20px rgba(0,0,0,0.08)",
            "minHeight": "400px",
            "transition": "background 0.25s ease",
        }),

        # ---- Video Modal ----
        html.Div(
            id="video-modal",
            className="video-modal-overlay",
            children=[
                html.Div([
                    html.Div([
                        html.H3(id="video-modal-title", style={"margin": "0", "color": "white", "fontSize": "20px"}),
                        html.Button("✕", id="close-video-modal", n_clicks=0, style={
                            "background": "rgba(255,255,255,0.2)",
                            "border": "none",
                            "color": "white",
                            "fontSize": "18px",
                            "cursor": "pointer",
                            "borderRadius": "6px",
                            "padding": "4px 10px",
                        }),
                    ], style={
                        "display": "flex",
                        "justifyContent": "space-between",
                        "alignItems": "center",
                        "padding": "20px 24px",
                        "background": CALTRANS_BLUE,  # changed from purple gradient
                    }),
                    html.Div(id="video-player-content", style={"padding": "24px"}),
                ], style={
                    "background": "white",
                    "borderRadius": "16px",
                    "width": "700px",
                    "maxWidth": "90vw",
                    "boxShadow": "0 20px 60px rgba(0,0,0,0.4)",
                    "overflow": "hidden",
                }),
            ],
        ),

        # Stores
        dcc.Store(id="theme-store", data="light"),
        dcc.Store(id="active-tab", data="incidents"),
        dcc.Store(id="modal-open", data=False),

        html.Div(
            f"Incident Video Viewer • http://{HOST}:{PORT}",
            style={"textAlign": "center", "color": "#999", "fontSize": "13px", "marginTop": "20px"}
        ),
    ])


app.layout = make_layout


# =========================
# Theme callbacks
# =========================

app.clientside_callback(
    """
    function(n_clicks, current_value) {
        if (n_clicks === undefined || n_clicks === null) return window.dash_clientside.no_update;
        var isDark = current_value && current_value.includes('dark');
        return isDark ? [] : ['dark'];
    }
    """,
    Output("theme-toggle", "value"),
    Input("theme-toggle-btn", "n_clicks"),
    State("theme-toggle", "value"),
    prevent_initial_call=True,
)

# FIX 1: Read cookie after React mounts
app.clientside_callback(
    """
    function(id) {
        var match = document.cookie.match('(?:^|; )dashboard_theme=([^;]*)');
        var saved = match ? match[1] : null;
        if (saved === 'dark') return ['dark'];
        return [];
    }
    """,
    Output("theme-toggle", "value", allow_duplicate=True),
    Input("theme-toggle", "id"),
    prevent_initial_call='initial_duplicate',
)

app.clientside_callback(
    """
    function(value) {
        var isDark = value && value.includes('dark');
        var theme = isDark ? 'dark' : 'light';
        document.cookie = 'dashboard_theme=' + theme + ';path=/;max-age=31536000;SameSite=Lax';
        document.documentElement.setAttribute('data-theme', theme);
        
        var btn = document.getElementById('theme-toggle-btn');
        var track = document.getElementById('theme-toggle-track');
        var label = document.getElementById('theme-toggle-label');
        if (btn && track && label) {
            if (isDark) {
                track.classList.add('active'); label.textContent = 'Theme: Dark'; btn.classList.add('dark-mode');
            } else {
                track.classList.remove('active'); label.textContent = 'Theme: Light'; btn.classList.remove('dark-mode');
            }
        }
        
        var tabBar = document.getElementById('tab-bar');
        var tabContent = document.getElementById('tab-content');
        if (tabBar) {
            tabBar.style.background = isDark ? '#2d2d2d' : 'white';
            tabBar.style.borderBottom = isDark ? '2px solid #404040' : '2px solid #eee';
        }
        if (tabContent) {
            tabContent.style.background = isDark ? '#2d2d2d' : 'white';
        }
        
        return theme;
    }
    """,
    Output("theme-store", "data"),
    Input("theme-toggle", "value"),
)


app.clientside_callback(
    """
    function(theme) {
        var isDark = theme === 'dark';
        // tab-bar still needs inline style update since it has dynamic inline styles
        return [
            {
                background: isDark ? '#2d2d2d' : 'white',
                borderRadius: '12px 12px 0 0',
                padding: '0 20px',
                borderBottom: isDark ? '2px solid #404040' : '2px solid #eee',
                boxShadow: '0 2px 8px rgba(0,0,0,0.05)',
            }
        ];
    }
    """,
    Output("tab-bar", "style"),
    Input("theme-store", "data"),
)


# =========================
# Tab switching
# =========================

app.clientside_callback(
    """
    function(inc_clicks, vid_clicks, lat_clicks, current_tab) {
        var ctx = window.dash_clientside.callback_context;
        if (!ctx.triggered || ctx.triggered.length === 0) {
            return [
                current_tab,
                current_tab === 'incidents' ? 'tab-btn active' : 'tab-btn',
                current_tab === 'videos' ? 'tab-btn active' : 'tab-btn',
                current_tab === 'latest' ? 'tab-btn active' : 'tab-btn',
            ];
        }
        var trigger = ctx.triggered[0].prop_id;
        var tab = current_tab;
        if (trigger.includes('tab-incidents-btn')) tab = 'incidents';
        else if (trigger.includes('tab-videos-btn')) tab = 'videos';
        else if (trigger.includes('tab-latest-btn')) tab = 'latest';
        
        return [
            tab,
            tab === 'incidents' ? 'tab-btn active' : 'tab-btn',
            tab === 'videos' ? 'tab-btn active' : 'tab-btn',
            tab === 'latest' ? 'tab-btn active' : 'tab-btn',
        ];
    }
    """,
    Output("active-tab", "data"),
    Output("tab-incidents-btn", "className"),
    Output("tab-videos-btn", "className"),
    Output("tab-latest-btn", "className"),
    Input("tab-incidents-btn", "n_clicks"),
    Input("tab-videos-btn", "n_clicks"),
    Input("tab-latest-btn", "n_clicks"),
    State("active-tab", "data"),
    prevent_initial_call=False,
)


@app.callback(
    Output("tab-content", "children"),
    Input("active-tab", "data"),
)
def render_tab_content(tab):
    INPUT_STYLE = {
        "width": "100%", "padding": "10px 14px", "borderRadius": "8px",
        "border": "2px solid #e0e0e0", "fontSize": "14px", "fontFamily": "inherit",
        "background": "white", "color": "#333", "boxSizing": "border-box",
    }

    if tab == "incidents" or tab is None:
        return html.Div([
            html.H3("Incidents by Time Range", className="tab-section-title", style={"margin": "0 0 6px 0", "fontSize": "22px", "fontWeight": "700", "color": "#333"}),
            html.P("Browse incidents from a specific time period and watch their videos", className="tab-section-subtitle", style={"color": "#666", "marginBottom": "24px", "marginTop": "4px"}),
            html.Div([
                html.Div([
                    html.Label("Time Range", className="tab-label", style={"fontSize": "13px", "fontWeight": "600", "marginBottom": "6px", "display": "block", "color": "#555"}),
                    dcc.Dropdown(
                        id="incident-timerange-select",
                        options=[{"label": k, "value": v} for k, v in TIME_RANGES.items()],
                        value=1440, clearable=False, style={"fontSize": "14px"},
                    ),
                ], style={"flex": "1", "minWidth": "160px"}),
                html.Div([
                    html.Label("Location (optional)", className="tab-label", style={"fontSize": "13px", "fontWeight": "600", "marginBottom": "6px", "display": "block", "color": "#555"}),
                    dcc.Input(id="incident-location-input", placeholder="e.g., patterson", type="text",
                        debounce=False, className="search-text-input", style=INPUT_STYLE),
                ], style={"flex": "1", "minWidth": "160px"}),
                html.Div([
                    html.Label("\u00a0", style={"fontSize": "13px", "marginBottom": "6px", "display": "block"}),
                    html.Button("Search Incidents", id="search-incidents-btn", n_clicks=0, style={
                        "width": "100%", "padding": "10px 20px", "fontSize": "14px", "fontWeight": "700",
                        "borderRadius": "8px", "border": "none",
                        "background": CALTRANS_GREEN,  # changed from pink gradient
                        "color": "white", "cursor": "pointer", "boxShadow": "0 4px 12px rgba(0,123,95,0.28)",
                    }),
                ], style={"minWidth": "160px"}),
            ], style={"display": "flex", "gap": "16px", "marginBottom": "28px", "flexWrap": "wrap", "alignItems": "flex-end"}),
            html.Div(id="incidents-container"),
        ])

    elif tab == "videos":
        return html.Div([
            html.H3("Browse All Videos", className="tab-section-title", style={"margin": "0 0 6px 0", "fontSize": "22px", "fontWeight": "700", "color": "#333"}),
            html.P("Search all recorded footage by time range and location", className="tab-section-subtitle", style={"color": "#666", "marginBottom": "24px", "marginTop": "4px"}),
            html.Div([
                html.Div([
                    html.Label("Time Range", className="tab-label", style={"fontSize": "13px", "fontWeight": "600", "marginBottom": "6px", "display": "block", "color": "#555"}),
                    dcc.Dropdown(id="timerange-select",
                        options=[{"label": k, "value": v} for k, v in TIME_RANGES.items()],
                        value=1440, clearable=False, style={"fontSize": "14px"},
                    ),
                ], style={"flex": "1", "minWidth": "160px"}),
                html.Div([
                    html.Label("Location (optional)", className="tab-label", style={"fontSize": "13px", "fontWeight": "600", "marginBottom": "6px", "display": "block", "color": "#555"}),
                    dcc.Input(id="location-input", placeholder="e.g., patterson", type="text",
                        debounce=False, className="search-text-input", style=INPUT_STYLE),
                ], style={"flex": "1", "minWidth": "160px"}),
                html.Div([
                    html.Label("\u00a0", style={"fontSize": "13px", "marginBottom": "6px", "display": "block"}),
                    html.Button("Search Videos", id="search-videos-btn", n_clicks=0, style={
                        "width": "100%", "padding": "10px 20px", "fontSize": "14px", "fontWeight": "700",
                        "borderRadius": "8px", "border": "none",
                        "background": CALTRANS_BLUE,  # changed from purple gradient
                        "color": "white", "cursor": "pointer", "boxShadow": "0 4px 12px rgba(0,51,102,0.28)",
                    }),
                ], style={"minWidth": "160px"}),
            ], style={"display": "flex", "gap": "16px", "marginBottom": "28px", "flexWrap": "wrap", "alignItems": "flex-end"}),
            html.Div(id="videos-container"),
        ])

    else:  # latest
        return html.Div([
            html.H3("Latest Captured Video", className="tab-section-title", style={"margin": "0 0 6px 0", "fontSize": "22px", "fontWeight": "700", "color": "#333"}),
            html.P("Most recent recording from the camera system", className="tab-section-subtitle", style={"color": "#666", "marginBottom": "24px", "marginTop": "4px"}),
            html.Button("↻ Refresh", id="refresh-latest-btn", n_clicks=0, style={
                "padding": "10px 24px", "fontSize": "14px", "fontWeight": "700",
                "borderRadius": "8px", "border": "none",
                "background": CALTRANS_BLUE,  # changed from purple gradient
                "color": "white", "cursor": "pointer", "marginBottom": "24px",
                "boxShadow": "0 4px 12px rgba(0,51,102,0.28)",
            }),
            html.Div(id="latest-video-container"),
        ])


# =========================
# Search callbacks
# =========================

@app.callback(
    Output("incidents-container", "children"),
    Input("search-incidents-btn", "n_clicks"),
    State("incident-timerange-select", "value"),
    State("incident-location-input", "value"),
    prevent_initial_call=True,
)
def search_incidents(n_clicks, minutes, location):
    location = location.strip() if location else None
    incidents = get_incidents_by_timerange(minutes, location)

    if not incidents:
        return html.Div("No incidents found for this time range.", className="empty-state", style={
            "padding": "40px", "textAlign": "center", "color": "#999",
            "fontSize": "16px", "borderRadius": "12px", "border": "2px dashed #ddd",
        })

    cards = [build_incident_card(inc) for inc in incidents]
    return html.Div([
        html.Div(
            f"Found {len(incidents)} incidents in the last {minutes} minutes",
            style={"fontWeight": "700", "fontSize": "16px", "marginBottom": "20px", "color": CALTRANS_GREEN}
        ),
        html.Div(cards, style={"display": "grid", "gridTemplateColumns": "repeat(auto-fill, minmax(280px, 1fr))", "gap": "16px"}),
    ])


@app.callback(
    Output("videos-container", "children"),
    Input("search-videos-btn", "n_clicks"),
    State("timerange-select", "value"),
    State("location-input", "value"),
    prevent_initial_call=True,
)
def search_videos(n_clicks, minutes, location):
    location = location.strip() if location else None
    videos = get_videos_by_timerange(minutes, location)

    if not videos:
        return html.Div("No videos found for this time range.", className="empty-state", style={
            "padding": "40px", "textAlign": "center", "color": "#999",
            "fontSize": "16px", "borderRadius": "12px", "border": "2px dashed #ddd",
        })

    cards = [build_video_card(v) for v in videos]
    return html.Div([
        html.Div(
            f"Found {len(videos)} videos",
            style={"fontWeight": "700", "fontSize": "16px", "marginBottom": "20px", "color": CALTRANS_BLUE}
        ),
        html.Div(cards, style={"display": "grid", "gridTemplateColumns": "repeat(auto-fill, minmax(280px, 1fr))", "gap": "16px"}),
    ])


@app.callback(
    Output("latest-video-container", "children"),
    Input("refresh-latest-btn", "n_clicks"),
    prevent_initial_call=True,
)
def load_latest_video(n_clicks):
    videos = get_recent_videos(limit=1)
    if not videos:
        return html.Div("No videos found.", style={"padding": "40px", "textAlign": "center", "color": "#999", "fontSize": "16px"})

    video = videos[0]
    return html.Div([
        html.Div(style={"maxWidth": "640px"}, children=[
            build_video_card(video),
        ])
    ])


# =========================
# Video modal callbacks
# =========================

@app.callback(
    Output("video-modal", "className"),
    Output("video-player-content", "children"),
    Output("video-modal-title", "children"),
    Input({"type": "watch-btn", "index": dash.ALL}, "n_clicks"),
    Input({"type": "incident-video-btn", "index": dash.ALL}, "n_clicks"),
    Input("close-video-modal", "n_clicks"),
    State("video-modal", "className"),
    prevent_initial_call=True,
)
def toggle_video_modal(watch_clicks, incident_clicks, close_click, current_class):
    ctx = dash.callback_context

    if not ctx.triggered:
        return "video-modal-overlay", "", "Video Player"

    trigger = ctx.triggered[0]
    prop_id = trigger["prop_id"]

    if "close-video-modal" in prop_id:
        return "video-modal-overlay", "", "Video Player"

    if "watch-btn" in prop_id:
        import json
        button_id = json.loads(prop_id.split(".")[0])
        video_id = button_id["index"]
        player = html.Video(
            src=f"{API_BASE_URL}/videos/{video_id}",
            controls=True,
            autoPlay=True,
            style={"width": "100%", "borderRadius": "8px"}
        )
        return "video-modal-overlay open", player, "📹 Video Playback"

    if "incident-video-btn" in prop_id:
        import json
        button_id = json.loads(prop_id.split(".")[0])
        incident_id = button_id["index"]
        video_info = get_video_for_incident(incident_id)

        if video_info:
            video_id = video_info.get("_id")
            filename = video_info.get("filename", "")
            player = html.Div([
                html.P(
                    f"Camera: {video_info.get('camera')} • {format_timestamp(video_info.get('timestamp'))}",
                    style={"color": "#666", "fontSize": "14px", "marginBottom": "12px"}
                ),
                html.Video(
                    src=f"{API_BASE_URL}/videos/{video_id}",
                    controls=True,
                    autoPlay=True,
                    style={"width": "100%", "borderRadius": "8px"}
                ),
            ])
            return "video-modal-overlay open", player, f"🚨 {filename}"
        else:
            return "video-modal-overlay open", html.Div(
                "No video found for this incident.",
                style={"color": "#f5576c", "padding": "20px", "textAlign": "center"}
            ), "No Video Found"

    return "video-modal-overlay", "", "Video Player"


# =========================
# Main
# =========================

def main():
    print("\n" + "=" * 60)
    print("Incident Video Viewer")
    print("=" * 60)
    print(f"📍 Open: http://{HOST}:{PORT}")
    print(f"🔗 API: {API_BASE_URL}")
    print("=" * 60 + "\n")

    app.run(debug=False, host=HOST, port=PORT)


if __name__ == "__main__":
    main()
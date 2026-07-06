"""
dashboard/__init__.py
──────────────────────
Public API for the dashboard package.

Provides the FastAPI + Leaflet.js GIS dashboard for
real-time disaster alert visualisation.

Usage:
    # Start the dashboard server
    python -m uvicorn dashboard.app:dashboard_app --host 0.0.0.0 --port 8080

    # Or directly:
    python dashboard/app.py
"""

from dashboard.app import dashboard_app

__all__ = [
    "dashboard_app",
]

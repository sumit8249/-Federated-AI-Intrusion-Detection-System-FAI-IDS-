"""
Root Application Entrypoint for Vercel and ASGI Servers
Spec Reference: Section 3 (API: FastAPI / Flask REST API)
"""

from api.index import app

# Export ASGI app for Vercel
__all__ = ["app"]

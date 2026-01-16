"""
Shim entrypoint so `python api_server.py` works from repo root.
"""
from core.api_server import app

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info",
        access_log=False,
    )

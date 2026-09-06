#!/usr/bin/env python3
"""FastAPI ASGI Server Launcher for DualCNN Inference Service.

This entrypoint starts the high-performance asynchronous REST API server exposing
the Dual-Head Convolutional Neural Network for production deployment, interactive
testing, and batch evaluation.

Endpoints Exposed:
    - GET  /              : Welcome message, API metadata, and interactive doc links.
    - GET  /health        : Health check, system readiness, GPU device info, model status.
    - POST /predict       : Single image multipart upload classification (cube/sphere probs).
    - POST /predict/batch : Multi-image batch classification with summary counts.

Environment Variables:
    - HOST    : Interface IP address to bind server (default: '0.0.0.0').
    - PORT    : Network port to listen on (default: 8000).
    - RELOAD  : Enable hot-reloading for local development (default: 'false').
    - WORKERS : Number of worker processes in production mode (default: 1).

Usage:
    # Run directly via Python:
    $ python app.py

    # Run with custom host and port:
    $ PORT=8080 HOST=127.0.0.1 python app.py

    # Production run via Uvicorn CLI:
    $ uvicorn src.api.app:app --host 0.0.0.0 --port 8000 --workers 4
"""

from __future__ import annotations

import os
import sys
from typing import NoReturn

# Import the FastAPI application instance
from src.api.app import app


def run_server() -> None:
    """Configures environment parameters and initiates the Uvicorn ASGI server.

    Reads binding host, port, and auto-reload settings from process environment
    variables, verifies Uvicorn installation, and launches the server.

    Raises:
        SystemExit: If the uvicorn package is not installed in the environment.
    """
    try:
        import uvicorn
    except ImportError:
        sys.stderr.write(
            "Error: Uvicorn is required to run the ASGI server.\n"
            "Please install it using: pip install uvicorn[standard]\n"
        )
        sys.exit(1)

    # Resolve server networking configuration from environment
    host: str = os.getenv("HOST", "0.0.0.0")
    port: int = int(os.getenv("PORT", "8000"))
    reload_enabled: bool = os.getenv("RELOAD", "false").lower() in ("true", "1", "yes")

    print(
        f"\n"
        f"DualCNN FastAPI Inference Service Starting\n"
        f" • Host Interface: {host}\n"
        f" • Listening Port: {port}\n"
        f" • Auto-Reload:    {reload_enabled}\n"
        f" • Swagger Docs:   http://{host}:{port}/docs\n"
        f" • Redoc Docs:     http://{host}:{port}/redoc\n"
        f" • Health Check:   http://{host}:{port}/health\n"
    )

    # Launch Uvicorn event loop
    uvicorn.run(
        "src.api.app:app",
        host=host,
        port=port,
        reload=reload_enabled,
    )


if __name__ == "__main__":
    run_server()


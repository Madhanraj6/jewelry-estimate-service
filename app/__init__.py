"""Jewelry estimate service application package.

Configuration is loaded here so that every entry point - the FastAPI app, the evaluation
harness and the maintenance scripts - sees the same environment before any submodule
reads it.
"""

from .config import load_env

load_env()
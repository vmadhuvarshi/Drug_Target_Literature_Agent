"""Centralized configuration for Ollama and model settings."""

import os

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # python-dotenv is optional

import ollama

MODEL_NAME = os.environ.get("OLLAMA_MODEL", "gemma4:e4b")
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://192.168.86.64:11434")

def get_ollama_client(timeout: int = 120) -> ollama.Client:
    """Return a configured Ollama client."""
    return ollama.Client(host=OLLAMA_HOST, timeout=timeout)

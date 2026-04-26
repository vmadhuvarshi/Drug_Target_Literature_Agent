"""Centralized configuration for Ollama and model settings."""

import ollama

MODEL_NAME = "deepseek-r1:8b"
OLLAMA_HOST = "http://192.168.86.64:11434/"

def get_ollama_client(timeout: int = 120) -> ollama.Client:
    """Return a configured Ollama client."""
    return ollama.Client(host=OLLAMA_HOST, timeout=timeout)

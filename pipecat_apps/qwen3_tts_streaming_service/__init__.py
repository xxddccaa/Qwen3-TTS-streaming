"""Standalone streaming TTS service built on top of Qwen3-TTS-streaming."""

from .app import app, create_app

__all__ = ["app", "create_app"]

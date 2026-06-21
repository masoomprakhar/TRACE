"""Shared application dependencies: a lazily-constructed pipeline singleton
and accessors for the repository and settings."""

from __future__ import annotations

import os
from typing import Optional

from trace_cv.core.config import Settings, load_settings
from trace_cv.pipeline import TracePipeline

_pipeline: Optional[TracePipeline] = None
_settings: Optional[Settings] = None


def get_settings() -> Settings:
    global _settings
    if _settings is None:
        _settings = load_settings(os.environ.get("TRACE_CONFIG"))
    return _settings


def get_pipeline() -> TracePipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = TracePipeline(get_settings())
    return _pipeline


def get_repo():
    return get_pipeline().repo

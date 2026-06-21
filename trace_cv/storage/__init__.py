"""Persistence: SQLAlchemy ORM + a repository for violations, analytics and
plate search."""

from trace_cv.storage.db import Repository
from trace_cv.storage.models import ViolationRow

__all__ = ["Repository", "ViolationRow"]

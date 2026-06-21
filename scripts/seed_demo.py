#!/usr/bin/env python3
"""Seed the TRACE database + evidence store with illustrative records so the
dashboard is populated without any ML models. Thin wrapper around
trace_cv.demo.seed_demo.

Usage:  python scripts/seed_demo.py [N]
"""

import sys

from trace_cv.core.config import load_settings
from trace_cv.demo import seed_demo

if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 40
    count = seed_demo(load_settings(), n=n)
    print(f"seeded {count} violation records into the database.")

"""
Pytest configuration: make the repository root importable.

With this file at the repo root, ``pytest`` adds the root to ``sys.path`` so
tests can ``import server`` / ``import data`` without setting PYTHONPATH.
"""

import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

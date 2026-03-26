"""Shim so that `pip install -e .` works on older pip that lacks PEP 660 support."""
from setuptools import setup

setup()

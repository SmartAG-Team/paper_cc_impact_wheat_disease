"""Stable command-line entrypoint for the role-based publication figure build."""
from .figure_pipeline import build

if __name__ == '__main__':
    build()

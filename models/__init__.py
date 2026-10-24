"""Shared scientific models and compatibility with their historical imports."""

from .compat import install_legacy_imports

install_legacy_imports()

__all__ = ["install_legacy_imports"]

"""Resolve historical module names to the same canonical module objects.

Installing the finder imports no scientific packages. Canonical modules are
loaded only when one of the explicitly listed historical names is requested.
"""

import importlib
import importlib.abc
import importlib.util
import sys


_PREFIXES = (
    ("analysis.gcb_preparation", "paper.scripts"),
    ("analysis.disease_improvement", "models.monitoring"),
    ("analysis.final_models", "models.learned"),
    ("analysis.final_climate", "models.climate"),
    ("analysis.methods", "models.methods"),
    ("analysis.final_study", "paper.analyses.source_support"),
    ("analysis.food_security_extension", "paper.analyses.food_security"),
    ("analysis.hafeez_metadata", "paper.analyses.hafeez_metadata"),
    ("analysis", "paper.analyses"),
    ("process_model", "models.phenology"),
    ("calibration", "models.calibration"),
    ("model", "models"),
)


class _AliasLoader(importlib.abc.Loader):
    def __init__(self, canonical_name, canonical_spec):
        self.canonical_name = canonical_name
        self.canonical_spec = canonical_spec
        self._metadata = None

    def create_module(self, spec):
        module = importlib.import_module(self.canonical_name)
        self._metadata = (module.__spec__, module.__loader__, module.__package__)
        return module

    def exec_module(self, module):
        # The import machinery temporarily assigns the alias spec to the reused
        # module. Preserve its canonical identity for reload and resource access.
        module.__spec__, module.__loader__, module.__package__ = self._metadata

    def get_code(self, fullname):
        getter = getattr(self.canonical_spec.loader, "get_code", None)
        if getter is None:
            raise ImportError(f"No executable code for {self.canonical_name}")
        return getter(self.canonical_name)


class _LegacyAliasFinder(importlib.abc.MetaPathFinder):
    _wheat_stb_legacy_alias_finder = True

    def find_spec(self, fullname, path=None, target=None):
        for previous, current in _PREFIXES:
            if fullname == previous or fullname.startswith(previous + "."):
                canonical = current + fullname[len(previous):]
                canonical_spec = importlib.util.find_spec(canonical)
                if canonical_spec is None:
                    raise ModuleNotFoundError(
                        f"No canonical module {canonical!r} for {fullname!r}",
                        name=fullname,
                    )
                alias_spec = importlib.util.spec_from_loader(
                    fullname,
                    _AliasLoader(canonical, canonical_spec),
                    origin=canonical_spec.origin,
                    is_package=canonical_spec.submodule_search_locations is not None,
                )
                alias_spec.has_location = canonical_spec.has_location
                return alias_spec
        return None


def install_legacy_imports():
    """Install the narrow alias finder once, before loading saved models."""
    for finder in sys.meta_path:
        if getattr(finder, "_wheat_stb_legacy_alias_finder", False):
            return finder
    finder = _LegacyAliasFinder()
    sys.meta_path.insert(0, finder)
    return finder

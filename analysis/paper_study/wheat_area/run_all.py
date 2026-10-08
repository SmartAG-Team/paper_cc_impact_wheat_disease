"""Reproduce wheat area and calendar artifacts using the archived provider bytes."""
from pathlib import Path
import importlib.metadata
import json
import platform

import reproduce_sources
import build_registry
import sample_point_calendars
import verify_registry
import plot_wheat_area


def run():
    reproduce_sources.run()
    build_registry.run()
    sample_point_calendars.run()
    verify_registry.run()
    plot_wheat_area.run()
    packages=['numpy','pandas','pyarrow','rasterio','geopandas','shapely','pyproj','xarray','netCDF4','matplotlib']
    versions={p:importlib.metadata.version(p) for p in packages}
    (Path(__file__).resolve().parent/'software_versions.json').write_text(json.dumps(dict(python=platform.python_version(),packages=versions),indent=2)+'\n')


if __name__=='__main__':run()

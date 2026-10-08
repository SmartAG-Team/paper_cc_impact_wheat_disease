"""Bundle source/evidence and install the reviewed manuscript with an archive."""
from pathlib import Path
import hashlib,json,shutil,zipfile
from .build import ROOT,DEST,STUDY


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    if not (DEST/'Manuscript.pdf').exists():raise FileNotFoundError('Reviewed staging manuscript is required.')
    paths=set()
    for folder in ['model','calibration','configurations/wheat_stb','examples/wheat_stb','tests','process_model']:
        for path in (ROOT/folder).rglob('*'):
            if path.is_file() and '__pycache__' not in path.parts and path.suffix in ('.py','.json','.csv','.md','.txt'):
                paths.add(path)
    for path in STUDY.rglob('*'):
        if not path.is_file() or '__pycache__' in path.parts:continue
        if any(part in ('annual_weather_cache','annual_outputs','aborted_before_juvenile_handover') for part in path.parts):continue
        if path.suffix in ('.py','.json','.csv','.parquet','.log','.md','.txt','.bib','.csl'):
            paths.add(path)
    # Include the active support code needed by field replay and document build.
    for relative in ['analysis/paper_study/structural_evaluation','analysis/paper_study/phenology_assumptions_20261006/partial_endpoint_inclusive',
        'analysis/paper_study/infection_priority_20261006/anthesis_clock']:
        for path in (ROOT/relative).glob('*.py'):paths.add(path)
    for relative in ['analysis/paper_study/calibrate_seasonal.py','analysis/paper_study/run_structural_canopy.py',
        'analysis/paper_study/run_regional.py','analysis/paper_study/infection_priority_20261006/build_priority_package.py',
        'analysis/paper_study/manuscript/build_documents.py','analysis/paper_study/manuscript/author_date_citations.py',
        'analysis/paper_study/manuscript/render_csl.js','analysis/paper_study/manuscript/csl_vendor/apa.csl',
        'analysis/paper_study/manuscript/csl_vendor/citeproc.js','WHEAT_STB_ENGINE.md','pyproject.toml','pytest.ini',
        'data/paper_study/field_weather_completed/daily_weather.parquet',
        'analysis/paper_study/seasonal_calibration_v1/basf_source_assessments_snapshot.csv',
        'analysis/paper_study/seasonal_calibration_v1/field_season_membership.csv',
        'analysis/paper_study/corteva_external_seasonal_v1/external_input_membership.csv',
        'data/paper_study/observations/corteva_external_assessments.csv',
        'data/paper_study/observations/corteva_location_disjoint_external_units.csv',
        'data/paper_study/wheat_area/trial_point_calendar_scenarios.csv',
        'data/paper_study/regional_parameter_uncertainty/spatial_draws.csv',
        'data/paper_study/regional_parameter_uncertainty/sampling_receipt.json']:
        paths.add(ROOT/relative)
    (DEST/'REPRODUCIBILITY.txt').write_text('Wheat STB model and evidence\n\n'
        'model/: parameterized simulation code only.\ncalibration/: fitting, observation preparation and validation utilities.\n'
        'configurations/: external parameter records.\nexamples/wheat_stb/: explicit frozen-field configuration and synthetic weather.\n'
        'analysis/paper_study/overwinter_leaf_model_20261006/: frozen study evidence and verification.\n\n'
        'Run the standalone example with python -m model.wheat_stb --config examples/wheat_stb/configuration.json '
        '--weather examples/wheat_stb/weather.csv --output <new-directory>. The wheel installs only model code.\n'
        'The field forcing and source assessments are bundled. Full climate reruns additionally require the registered sampled-weather '
        'cache or checksum-identified NEX input files, historical alignment coefficients, and area/calendar registries in the workspace. '
        'All draw-season outputs and period summaries are supplied; raw whole-domain climate and sampled weather caches are retained externally.\n\n'
        'Disease fits use original-field signs and timing; evaluation is retrospective. Infection events are model states, '
        'yield transfers are conditional, and neither actual yield loss nor accurate disease absence is established.\n')
    package=DEST/'Software_and_Evidence.zip'
    manifest=[]
    with zipfile.ZipFile(package,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as archive:
        for path in sorted(paths):
            if not path.is_file():raise FileNotFoundError(path)
            relative=str(path.relative_to(ROOT));archive.write(path,relative)
            manifest.append(dict(path=relative,bytes=path.stat().st_size,sha256=sha(path)))
        archive.writestr('PACKAGE_MANIFEST.json',json.dumps(manifest,indent=2)+'\n')
        archive.write(DEST/'REPRODUCIBILITY.txt','REPRODUCIBILITY.txt')
    with zipfile.ZipFile(package) as archive:assert archive.testzip() is None
    (DEST/'software_package_receipt.json').write_text(json.dumps(dict(status='complete',members=len(manifest)+2,
        bundled_sources_and_evidence=manifest,package_sha256=sha(package),raw_climate_cache_bundled=False),indent=2)+'\n')
    active=ROOT/'publication/european_wheat_stb'
    backup=ROOT/'publication/european_wheat_stb_before_overwinter_oop_20261006'
    if backup.exists():raise FileExistsError('The original manuscript archive already exists.')
    before={str(path.relative_to(active)):sha(path) for path in active.iterdir() if path.is_file() and path.suffix in ('.docx','.pdf','.txt','.json','.xlsx')}
    active.rename(backup);DEST.rename(active)
    assert all(sha(backup/name)==digest for name,digest in before.items())
    (active/'previous_manuscript_archive.json').write_text(json.dumps(dict(archive_path=str(backup.relative_to(ROOT)),
        original_files_sha256=before,archive_unchanged=True),indent=2)+'\n')
    print(json.dumps(dict(active=str(active),archive=str(backup),package_members=len(manifest)+2,
        package_bytes=package.stat().st_size if package.exists() else (active/package.name).stat().st_size)))


if __name__=='__main__':main()

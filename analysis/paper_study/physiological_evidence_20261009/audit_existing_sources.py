"""Read the supplied archives in place; distinguish summaries from observations."""
import csv
import hashlib
import io
import json
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parent
ARCHIVE = Path('/Users/gangzhao/Documents/workspace/paper_cc_impact_wheat_disease/data/STB_consolidated_collection_20261009.zip')
FOLLOWUP = Path('/Users/gangzhao/Documents/workspace/paper_cc_impact_wheat_disease/data/STB_paper_followup_20261009')


def main():
    results = []
    with zipfile.ZipFile(ARCHIVE) as archive:
        for tail, evidence_type in [
            ('bancal2007_treatment_aggregates.csv', 'published_treatment_aggregates'),
            ('foulkes2006_table2_paired_predicted_means.csv', 'mixed_model_predicted_means'),
            ('taylor_cunniffe2023_S1_Data.csv', 'fitted_response_grid'),
        ]:
            matches = [n for n in archive.namelist() if n.endswith('/' + tail)]
            if len(matches) != 1:
                raise ValueError(f'Expected one exact archive member for {tail}')
            data = archive.read(matches[0])
            reader = csv.DictReader(io.StringIO(data.decode('utf-8-sig')))
            rows = list(reader)
            result = dict(source=str(ARCHIVE), member=matches[0],
                          sha256=hashlib.sha256(data).hexdigest(), rows=len(rows),
                          columns=reader.fieldnames, evidence_type=evidence_type,
                          admissible_independent_plot_harvests=0)
            if tail.startswith('taylor'):
                xs = [float(r['disease_severity']) for r in rows]
                result.update(unique_severity_values=len(set(xs)), min_severity=min(xs),
                              max_severity=max(xs),
                              increments=sorted(set(round(b-a, 12) for a, b in zip(xs, xs[1:]))))
            if tail.startswith('bancal'):
                result['unique_years'] = sorted(set(r['year'] for r in rows))
                result['unique_treatments'] = len(set(r['treatment'] for r in rows))
            results.append(result)
    p = FOLLOWUP / 'analysis_ready/additional_ordinal_stb_yield_948.csv'
    data = p.read_bytes()
    reader = csv.DictReader(io.StringIO(data.decode('utf-8-sig')))
    rows = list(reader)
    results.append(dict(source=str(p), sha256=hashlib.sha256(data).hexdigest(),
                        rows=len(rows), columns=reader.fieldnames,
                        evidence_type='existing_plot_pairs_not_new_acquisition',
                        admissible_physiological_trajectories=0,
                        classification_basis='Supplied START_HERE.txt and prior yield_transfer_audit; no new yield analysis'))
    output = dict(raw_observations_newly_acquired=0,
                  prior_sources_unchanged=True, checks=results)
    (ROOT / 'existing_rejection_audit.json').write_text(json.dumps(output, indent=2)+'\n')
    print(json.dumps({'source_row_counts': [x['rows'] for x in results],
                      'raw_observations_newly_acquired': 0}))


if __name__ == '__main__':
    main()

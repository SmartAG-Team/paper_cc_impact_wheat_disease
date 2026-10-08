"""Read downloaded public trial tables; no model fitting or manuscript writes."""
from pathlib import Path
from bs4 import BeautifulSoup
import csv
import json
import re
from datetime import datetime

ROOT = Path(__file__).resolve().parent

def grid(table):
    cells = {}
    for ri, row in enumerate(table.find_all('tr')):
        ci = 0
        for cell in row.find_all(['td', 'th'], recursive=False):
            while (ri, ci) in cells:
                ci += 1
            value = cell.get_text(' ', strip=True).replace('\xa0', ' ')
            nr = int(cell.get('rowspan', 1))
            nc = int(cell.get('colspan', 1))
            for dr in range(nr):
                for dc in range(nc):
                    cells[ri + dr, ci + dc] = value
            ci += nc
    if not cells:
        return []
    nr = max(r for r, c in cells) + 1
    nc = max(c for r, c in cells) + 1
    return [[cells.get((r, c), '') for c in range(nc)] for r in range(nr)]

def number(value):
    try:
        return float(value.replace(',', '.'))
    except (ValueError, AttributeError):
        return None

def date_stage(text):
    d = re.search(r'\b\d{1,2}/\d{1,2}/\d{4}\b', text)
    stage = re.search(r'ST\.\s*(\d+)', text)
    return (datetime.strptime(d.group(), '%m/%d/%Y').date().isoformat() if d else None,
            int(stage.group(1)) if stage else None)

def save_csv(path, rows):
    if not rows:
        return
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with path.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

def main():
    measurements = []
    yields = []
    metadata = []
    native_units = {}
    if (ROOT/'nfts_native_units_provenance.json').exists():
        native_units = {x['registry_id']:x for x in json.loads((ROOT/'nfts_native_units_provenance.json').read_text())}
    for path in sorted(ROOT.glob('nfts_*.html')):
        if 'plan_' in path.name or 'native_' in path.name:
            continue
        soup = BeautifulSoup(path.read_text(), 'html.parser')
        text = soup.get_text(' ', strip=True)
        tables = soup.find_all('table')
        if 'Trial documentation' not in text:
            continue
        title = re.search(r'(L?\d[^ ]+?\d{3})\.\s+([^\n]+?) Field Trial results', text)
        trial_name = title.group(1) if title else path.stem
        link = soup.find('a', id='hyperlinkExport')
        urlmatch = re.search(r'KardexID=(\d+)', str(soup))
        registry_id = int(urlmatch.group(1)) if urlmatch else None
        native_path = ROOT/f'nfts_native_{registry_id}_sv.html'
        native_tables = {}
        if native_path.exists():
            nsoup = BeautifulSoup(native_path.read_text(), 'html.parser')
            native_tables = {t.get('id'):grid(t) for t in nsoup.find_all('table') if (t.get('id') or '').startswith('tabResultatLedNieveau')}
        record = f'nfts_{registry_id}' if registry_id else path.stem
        if link:
            source_url = f'https://nfts.dlbr.dk/Forms/Dokumentation.aspx?KardexID={registry_id}&applLangID=en'
        else:
            source_url = None
        reps = re.search(r'No\. of replicates:\s*(\d+)', text)
        nf = re.search(r'Trial design:\s*([^\.]+)', text)
        latlon = re.search(r'Gps:\s*([\d.-]+),?\s+([\d.-]+)', text)
        crop = re.search(r'Crop:\s*([^\.]+)', text)
        cultivar = re.search(r'Variety:\s*([^\.]+)', text)
        cultivars = {}
        treatment_grid = None
        validation = []
        statistical_notes = []
        for ti, table in enumerate(tables):
            tx = table.get_text(' ', strip=True)
            if tx.startswith('Factor Segmt.') or tx.startswith('Segmt. Tr.'):
                treatment_grid = grid(table)
                save_csv(ROOT/(record + '_treatment_rows.csv'), [dict(source_row=i, **{f'col{j}':v for j,v in enumerate(r)}) for i,r in enumerate(treatment_grid)])
                if treatment_grid[0][0] == 'Factor':
                    for r in treatment_grid[1:]:
                        if r[0] == '1' and r[1].isdigit():
                            cultivars[r[1]] = r[6]
            if tx.startswith('Classification Remarks'):
                validation += [r for r in grid(table)[1:]]
            if tx.startswith('Assessm. time Date Measurement of') or tx.startswith('Måleparameter Assessm. time'):
                statistical_notes += grid(table)
        meta = {
            'record':record,'registry_id':registry_id,'trial_name':trial_name,
            'source_html':path.name,'source_url':source_url,
            'replicates':int(reps.group(1)) if reps else None,
            'design':nf.group(1) if nf else None,
            'latitude':float(latlon.group(1)) if latlon else None,
            'longitude':float(latlon.group(2)) if latlon else None,
            'crop':crop.group(1) if crop else None,
            'single_variety':cultivar.group(1) if cultivar else None,
            'cultivar_factor_labels':cultivars,'validation':validation,
            'statistical_notes':statistical_notes,
            'source_mean_status':'Published treatment means; arithmetic or model-adjusted status must follow registry notes, not raw plot observations.',
            'native_yield_label':'Yield hkg grain',
            'absolute_yield_conversion':'Not applied: exact per-hectare and moisture normalization require registry calculation documentation.',
        }
        nu = native_units.get(registry_id, {})
        verified = 'Skörd dt/ha kärna 15%' in nu.get('yield_labels', [])
        if verified:
            meta['native_swedish_yield_label'] = 'Skörd dt/ha kärna 15%'
            meta['native_yield_unit'] = 'dt/ha (= hkg/ha), grain at 15% moisture'
            meta['native_unit_source'] = nu['file']
            meta['absolute_yield_conversion'] = 'Native yield multiplied by 0.1 gives t/ha at the same 15% moisture.'
        metadata.append(meta)
        for ti, table in enumerate(tables):
            if not (table.get('id') or '').startswith('tabResultatLedNieveau'):
                continue
            g = grid(table)
            if len(g) < 4:
                continue
            parameters = g[2]
            save_csv(ROOT/(record+f'_table{ti:02d}_grid.csv'), [dict(source_row=i, **{f'col{j}':v for j,v in enumerate(r)}) for i,r in enumerate(g)])
            stat = 'Lower conf.' in parameters
            keycols = next((i for i,h in enumerate(parameters) if h), 0)
            for ri, r in enumerate(g[3:], 3):
                if keycols == 2:
                    factor, entry = r[:2]
                elif keycols == 1:
                    factor, entry = '', r[0]
                else:
                    continue
                if not entry.isdigit():
                    continue
                for ci in range(keycols, len(parameters)):
                    label = parameters[ci]
                    if stat and label != 'Yield hkg grain':
                        continue
                    allowed = (stat and label == 'Yield hkg grain') or (not stat and any(x in label for x in ['Septoria', 'Wheat leaf blotch', 'Yellow rust', 'Brown rust', 'Powdery mildew', 'Leaf spot, yellow', 'Leaf spot disease DRECSP', 'Bladfläcksjuka', 'Brunrost']))
                    if not allowed or 'resistant' in label.lower():
                        continue
                    value = number(r[ci])
                    date, stage = date_stage(g[1][ci])
                    item = dict(record=record,registry_id=registry_id,trial_name=trial_name,
                        cultivar_entry=entry,cultivar=cultivars.get(entry,meta['single_variety']),
                        treatment_factor=factor,treatment_code=(factor+entry),
                        date=date,growth_stage=stage,measurement=label,value=value,
                        native_value=r[ci],source_html=path.name,source_table=ti,source_row=ri,
                        source_column=ci,source_table_id=table.get('id'),
                        replicates=meta['replicates'],mean_status=meta['source_mean_status'])
                    ng = native_tables.get(table.get('id'), [])
                    if len(ng)>2 and ci<len(ng[2]):
                        item['native_swedish_measurement_label'] = ng[2][ci]
                        item['native_language_source_html'] = native_path.name
                    if stat:
                        item['lower_confidence_limit'] = number(r[ci+1])
                        item['upper_confidence_limit'] = number(r[ci+2])
                        item['native_unit_verified'] = verified
                        item['yield_t_ha_15percent_moisture'] = value * 0.1 if verified and value is not None else None
                        item['lower_confidence_limit_t_ha'] = item['lower_confidence_limit'] * 0.1 if verified and item['lower_confidence_limit'] is not None else None
                        item['upper_confidence_limit_t_ha'] = item['upper_confidence_limit'] * 0.1 if verified and item['upper_confidence_limit'] is not None else None
                        yields.append(item)
                    else:
                        measurements.append(item)
    save_csv(ROOT/'nfts_disease_observations_long.csv', measurements)
    save_csv(ROOT/'nfts_yield_treatment_means.csv', yields)
    (ROOT/'nfts_trial_metadata.json').write_text(json.dumps(metadata, indent=2, ensure_ascii=False))
    print('Trials',len(metadata),'yield means',len(yields),'disease cells',len(measurements),'nonmissing disease cells',sum(x['value'] is not None for x in measurements))
    for m in metadata:
        yy=[x for x in yields if x['record']==m['record']]
        dd=[x for x in measurements if x['record']==m['record']]
        print(m['record'],m['trial_name'],'yield',len(yy),'disease',len(dd),'nonmissing',sum(x['value'] is not None for x in dd),'validation',m['validation'])

if __name__ == '__main__':
    main()

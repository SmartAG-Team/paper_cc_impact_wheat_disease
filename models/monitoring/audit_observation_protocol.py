"""Read-only record-level audit of native BASF observation-method metadata."""
import json
from pathlib import Path
import shutil

import pandas as pd

from .innovation_common import ROOT,PARENT
from .inputs import sha

SOURCE=ROOT.parent/'data/zenodo-6521175/basf-wheat-diseases.txt'


def run():
    out=ROOT/'outputs/disease_improvement/observation_protocol_audit_20261022'
    if out.exists():raise FileExistsError('Fresh audit output required')
    out.mkdir(parents=True)
    paths=[SOURCE,PARENT/'training_observation_ledger.csv',Path(__file__)]
    registration=dict(source_sha256={str(p):sha(p) for p in paths},
        purpose='Audit metadata without reading disease values, prediction errors or changing eligibility',
        primary_documentation='https://www.ipmdecisions.net/media/evlpe55l/d4-2-data-set-for-dss-evaluation-septoria-tritici-and-grapevine-downby-mildew-003.pdf',
        interpretation='Provider D4.2 identifies Corteva INFECT as percentage disease severity. BASF wheat records report percentage infection; its P%INF method is distinguished from frequency in the report grapevine section, which does not establish the complete wheat sampling protocol.',
        limitations=['Native sample-unit descriptors do not recover numbers of leaves, plants or replicates',
                     'Clarifier wording alone cannot establish a wrong target',
                     'No anatomical cross-source remapping or error-dependent exclusions'])
    (out/'registration.json').write_text(json.dumps(registration,indent=2)+'\n')
    shutil.copyfile(Path(__file__),out/'audit_source_snapshot.py')
    keys=['global_target_index','field_id','endpoint_series','date']
    obs=pd.read_csv(PARENT/'training_observation_ledger.csv',usecols=keys+['source'])
    columns=['TrialId','Date','PlantPart','Organism','Parameter','Treatment','NumSprays',
             'Clarifier','Method','SampleUnit']
    raw=pd.read_csv(SOURCE,sep='\t',usecols=columns)
    raw['source_row_id']=raw.index+2
    raw['field_id']='basf-wheat-diseases|'+raw.TrialId.astype(str)+'|'+pd.to_datetime(raw.Date).dt.year.astype(str)
    raw['endpoint_series']=raw.TrialId.astype(str)+'|'+raw.PlantPart.fillna('unspecified')
    raw['date']=raw.Date
    raw=raw.loc[raw.Organism.eq('SEPTTR')&raw.Parameter.eq('INFECT')&raw.Treatment.eq('Untreated')&raw.NumSprays.eq(0)]
    retained=obs.loc[obs.source.eq('BASF'),keys].merge(raw,on=['field_id','endpoint_series','date'],
                                                   how='left',validate='one_to_one')
    assert len(retained)==584 and retained.source_row_id.notna().all()
    retained[keys+['source_row_id','PlantPart','Clarifier','Method','SampleUnit']].to_csv(out/'retained_basf_protocol.csv',index=False)
    excluded_crop=raw.loc[raw.Clarifier.eq('CROP INJURY')]
    assert not set(excluded_crop.source_row_id)&set(retained.source_row_id)
    result=dict(passed=True,retained_BASF_observations=len(retained),native_method_counts=retained.Method.fillna('<missing>').value_counts().to_dict(),
        sample_unit_counts=retained.SampleUnit.fillna('<missing>').value_counts().to_dict(),
        clarifier_counts=retained.Clarifier.fillna('<missing>').value_counts().to_dict(),
        crop_injury_clarifier_in_retained=0,crop_injury_clarifier_in_all_source_septoria_untreated=int(len(excluded_crop)),
        target_values_read=False,prediction_errors_read=False,observations_deleted=0,
        source_unchanged=all(sha(Path(p))==h for p,h in registration['source_sha256'].items()),
        artifacts_sha256={p.name:sha(p) for p in out.iterdir() if p.is_file()})
    (out/'verification.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='artifacts_sha256'},indent=2))


if __name__=='__main__':run()

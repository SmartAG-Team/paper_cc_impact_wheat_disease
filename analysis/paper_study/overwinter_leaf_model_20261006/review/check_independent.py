from pathlib import Path
import importlib.util,sys,json,math,os
from dataclasses import replace
import numpy as np
HERE=Path(__file__).resolve().parent

def load(name,rel):
 spec=importlib.util.spec_from_file_location(name,HERE/os.environ.get('REVIEW_SNAPSHOT','source_snapshot')/rel)
 mod=importlib.util.module_from_spec(spec);sys.modules[name]=mod;spec.loader.exec_module(mod);return mod
leaf=load('review_leaf','model/seasonal_septoria/leaf_phenology.py')
ow=load('review_ow','model/seasonal_septoria/overwinter.py')
stage_path=HERE.parent/'phenology/calibrated_stage_thresholds.json'
stages={int(k):v for k,v in json.loads(stage_path.read_text())['all_stage_thresholds'].items()}
out={}
# A rates object for a different instance can rewind the receiving development state.
a=leaf.LeafCanopyModel(stages,rank_spacing_units=120.)
b=leaf.LeafCanopyModel(stages,rank_spacing_units=120.)
a.step(900.,18.);b.step(500.,18.)
foreign=b.calc_rates(600.,18.)
try:
 a.integrate(foreign)
 out['foreign_leaf_rates']={'accepted':True,'before_accumulation':900.,'after_accumulation':a.snapshot()['previous_accumulation']}
except ValueError as e:out['foreign_leaf_rates']={'accepted':False,'error':str(e)}
# Restore to a higher-development checkpoint at the same integration day.
c=leaf.LeafCanopyModel(stages,rank_spacing_units=120.);c.step(500.,18.)
old=c.calc_rates(600.,18.)
high=leaf.LeafCanopyModel(stages,rank_spacing_units=120.);high.step(900.,18.)
c.restore(high.snapshot())
try:
 c.integrate(old)
 out['pre_restore_leaf_rates']={'accepted':True,'restored_accumulation':900.,'after_accumulation':c.snapshot()['previous_accumulation']}
except ValueError as e:out['pre_restore_leaf_rates']={'accepted':False,'error':str(e)}
# The single shared residue decay imposes a closed-form total-source invariant.
res=[]
for maturation,decay,step,days in [(30.,90.,.25,1),(30.,90.,.25,100),(.001,1.,.25,1),(30.,90.,.125,100)]:
 params=ow.OverwinterParameters(primary_scale=0.,secondary_scale=0.,initial_ready_fraction=0.,residue_maturation_reference_days=maturation,residue_decay_reference_days=decay)
 model=ow.OverwinterModel(params,leaf_count=1,time_step=step)
 for _ in range(days):model.integrate(model.calc_rates(18.,1.,0.,np.ones(1,bool),np.zeros(1),np.ones(1)))
 actual=float(model.residue.sum());expected=math.exp(-days/decay)
 res.append(dict(maturation_reference_days=maturation,residue_decay_reference_days=decay,time_step=step,days=days,actual_total=actual,exact_expected_total=expected,relative_error=actual/expected-1))
out['residue_total_decay_invariant']=res
# Show clipped rank overlap and persistent juvenile capacity.
ranks=[]
for spacing in [80.,120.,160.]:
 accum=np.linspace(0.,1800.,181)[None,:];t=np.full_like(accum,12.);mask=np.ones_like(accum,bool)
 host=leaf.leaf_host(accum,t,stages,rank_spacing_units=spacing,forcing_mask=mask)
 ranks.append(dict(spacing=spacing,visible_thresholds=host.visible_thresholds.tolist(),ranks_clipped_at_C10=[f'F{i+1}' for i,v in enumerate(host.visible_thresholds) if v==stages[10]],juvenile_area_at_grain_filling=float(host.area[0,160,7]),juvenile_renewal_at_grain_filling=float(host.renewal[0,160,7]),final_area_sum=float(host.area[0,-1].sum())))
out['rank_overlap_and_juvenile']=ranks
# Cold persistence is explicitly a0C-base frozen-stage hypothesis.
p=ow.OverwinterParameters(primary_scale=0.,secondary_scale=0.)
m=ow.OverwinterModel(p,leaf_count=1)
s=m.snapshot();s['tissue_state'][0][0]=.5;s['tissue_state'][0][1]=.5;m.restore(s)
start=m.state.copy()
for _ in range(30):m.integrate(m.calc_rates(-1.,0.,0.,np.ones(1,bool),np.zeros(1),np.ones(1)))
out['subzero_infection_state']={'unchanged_after_30_days_at_minus1C':bool(np.array_equal(start,m.state)),'hypothesis':'zero-base thermal clock; retention is implemented, subzero development is not'}
(HERE/os.environ.get('REVIEW_OUTPUT','independent_checks.json')).write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out,indent=2))

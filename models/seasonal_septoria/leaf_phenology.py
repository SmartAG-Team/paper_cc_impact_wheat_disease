"""Stage-anchored final-leaf ranks under explicit developmental-spacing scenarios.

BBCH32/33 are node events, not leaf counts. Flag-tip visibility (37) and flag
unfolding (39) have independently stage-calibrated thresholds. Effective rank
spacing is in the donor T-P-V index, not a measured thermal phyllochron. Counts
describe the modeled F1--F7 ranks and never estimate true final leaf number.
"""

from dataclasses import dataclass
import hashlib
import json
from types import MappingProxyType
from uuid import uuid4
import numpy as np


@dataclass
class LeafHost:
    active: np.ndarray
    renewal: np.ndarray
    area: np.ndarray
    visible: np.ndarray
    fully_unfolded: np.ndarray
    modeled_visible_final_leaf_count: np.ndarray
    modeled_unfolded_final_leaf_count: np.ndarray
    top3_visible: np.ndarray
    top3_fully_unfolded: np.ndarray
    stage_day_index: dict
    visible_final_ranks_at_stage: dict
    unfolded_final_ranks_at_stage: dict
    visible_thresholds: np.ndarray
    unfolded_thresholds: np.ndarray


def leaf_host(accumulation, temperature, calibrated_stage_thresholds, *,
              rank_spacing_units, forcing_mask, juvenile_policy='handover_31_39'):
    """Daily F1--F7 plus juvenile capacity with prior-day availability gates.

    ``visible``/``fully_unfolded`` refer to each valid day's end; ``active``
    requires visibility on the preceding valid day and positive unfolding
    capacity. First stage crossings use zero-based forcing-day indices (-1 for
    unreached). Stage rank status refers to the day's end at that crossing.

    Lower-rank visibility is C37 minus rank times the mandatory effective
    spacing, bounded below by C10. Lower ranks clipped at C10 share that date;
    they are unresolved juvenile-stage approximations. Each final rank retains
    the C39-C37 unfolding duration. Slot7 is a normalized juvenile reservoir,
    with the retained 300 positive-degree-day renewal assumption. Its primary
    capacity withdraws linearly between C31 and C39; persistent capacity is an
    explicit structural sensitivity. Natural senescence and actual leaf-area
    magnitude are not measured or estimated by this policy.
    """
    a,t=np.asarray(accumulation,float),np.asarray(temperature,float)
    raw=np.asarray(forcing_mask)
    thresholds={int(key):float(value) for key,value in calibrated_stage_thresholds.items()}
    required={10,31,37,39,51,85}
    ordered=np.array([thresholds[key] for key in sorted(thresholds)])
    if (a.ndim!=2 or not a.size or t.shape!=a.shape or raw.shape!=a.shape
            or not np.isfinite(a).all() or not np.isfinite(t).all()
            or np.any(a<0) or np.any(np.diff(a,axis=1)<-1e-10)
            or np.any((raw!=0)&(raw!=1)) or not required.issubset(thresholds)
            or not np.isfinite(ordered).all() or np.any(ordered<=0) or np.any(np.diff(ordered)<=0)
            or not np.isfinite(rank_spacing_units) or rank_spacing_units<=0
            or juvenile_policy not in ('handover_31_39','persistent')):
        raise ValueError('Ordered stage thresholds, valid forcing and positive effective rank spacing required.')
    mask=raw.astype(bool)
    starts=mask&~np.column_stack([np.zeros(len(mask),bool),mask[:,:-1]])
    if np.any(starts.sum(axis=1)>1):
        raise ValueError('A forcing mask must contain at most one contiguous crop window per field.')
    visibility=np.maximum(thresholds[10],thresholds[37]-np.arange(7)*rank_spacing_units)
    unfolding=visibility+(thresholds[39]-thresholds[37])
    visible=(a[:,:,None]>=visibility)&mask[:,:,None]
    unfolded=(a[:,:,None]>=unfolding)&mask[:,:,None]
    prior=np.column_stack([np.zeros(len(a)),a[:,:-1]])
    prior_valid=np.column_stack([np.zeros(len(mask),bool),mask[:,:-1]])
    active=(prior[:,:,None]>=visibility)&mask[:,:,None]&prior_valid[:,:,None]
    area=np.clip((a[:,:,None]-visibility)/(unfolding-visibility),0.,1.)*active
    active&=area>0.
    juvenile_capacity=(np.clip((thresholds[39]-a)/(thresholds[39]-thresholds[31]),0.,1.)
        if juvenile_policy=='handover_31_39' else np.ones_like(a))
    juvenile=(prior>=thresholds[10])&mask&prior_valid&(juvenile_capacity>0.)
    active=np.concatenate([active,juvenile[:,:,None]],axis=2)
    area=np.concatenate([area,(juvenile*juvenile_capacity)[:,:,None]],axis=2)
    juvenile_visible=(a>=thresholds[10])&mask&(juvenile_capacity>0.)
    visible=np.concatenate([visible,juvenile_visible[:,:,None]],axis=2)
    unfolded=np.concatenate([unfolded,juvenile_visible[:,:,None]],axis=2)
    previous=np.concatenate([np.zeros_like(area[:,:1]),area[:,:-1]],axis=1)
    renewal=np.divide(np.maximum(area-previous,0.),area,out=np.zeros_like(area),where=area>0.)
    renewal[:,:,7]=(-np.expm1(-np.maximum(t,0.)/300.))*juvenile
    renewal[~active]=0.
    stage_days,stage_visible,stage_unfolded={},{},{}
    for stage,threshold in sorted(thresholds.items()):
        reached=(a>=threshold)&mask
        found=reached.any(axis=1)
        indices=np.where(found,reached.argmax(axis=1),-1)
        stage_days[stage]=indices
        status_visible=np.zeros((len(a),7),bool)
        status_unfolded=np.zeros_like(status_visible)
        fields=np.flatnonzero(found)
        status_visible[fields]=visible[fields,indices[fields],:7]
        status_unfolded[fields]=unfolded[fields,indices[fields],:7]
        stage_visible[stage]=status_visible
        stage_unfolded[stage]=status_unfolded
    return LeafHost(active,renewal,area,visible,unfolded,visible[:,:,:7].sum(axis=2),
        unfolded[:,:,:7].sum(axis=2),visible[:,:,:3],unfolded[:,:,:3],
        stage_days,stage_visible,stage_unfolded,visibility,unfolding)


def _immutable_array(value):
    array=np.asarray(value)
    return np.frombuffer(array.tobytes(),dtype=array.dtype).reshape(array.shape)


@dataclass(frozen=True)
class DailyLeafState:
    active: np.ndarray
    renewal: np.ndarray
    area: np.ndarray
    visible: np.ndarray
    fully_unfolded: np.ndarray
    modeled_visible_final_leaf_count: int
    modeled_unfolded_final_leaf_count: int
    top3_visible: np.ndarray
    top3_fully_unfolded: np.ndarray
    predicted_stage_code: int
    stage_progress: object
    newly_crossed_stage: object


@dataclass(frozen=True)
class DailyLeafRates(DailyLeafState):
    config_identity: str
    owner_identity: str
    previous_day_count: int
    next_accumulation: float
    forcing_valid: bool


class LeafCanopyModel:
    """One-field daily component with separate rate calculation and integration.

    ``calc_rates`` is pure; ``integrate`` commits only rates calculated against
    the current state. Layer arrays have length8, F1--F7 followed by juvenile.
    Stage progress is bounded development towards the next declared threshold,
    never an interpolated numerical BBCH or an observed node/leaf count.
    """

    def __init__(self,stage_thresholds,*,rank_spacing_units,juvenile_policy='handover_31_39'):
        template=leaf_host(np.zeros((1,1)),np.zeros((1,1)),stage_thresholds,
            rank_spacing_units=rank_spacing_units,forcing_mask=np.zeros((1,1),bool),juvenile_policy=juvenile_policy)
        self.thresholds=MappingProxyType({int(k):float(v) for k,v in sorted(stage_thresholds.items(),key=lambda x:int(x[0]))})
        self.rank_spacing_units=float(rank_spacing_units)
        self.juvenile_policy=juvenile_policy
        self.visibility=template.visible_thresholds.copy()
        self.unfolding=template.unfolded_thresholds.copy()
        payload=dict(schema_version=1,thresholds=dict(self.thresholds),rank_spacing_units=self.rank_spacing_units,juvenile_policy=juvenile_policy)
        self.config_identity=hashlib.sha256(json.dumps(payload,sort_keys=True,separators=(',',':')).encode()).hexdigest()
        self._owner_identity=uuid4().hex
        self._previous_accumulation=0.
        self._previous_area=np.zeros(8)
        self._previous_valid=False
        self._started=False
        self._ended=False
        self._day_count=0
        self._reached={stage:False for stage in self.thresholds}

    def calc_rates(self,current_accumulation,current_temperature,forcing_valid=True):
        if (np.ndim(current_accumulation)!=0 or np.ndim(current_temperature)!=0 or np.ndim(forcing_valid)!=0
                or not np.isfinite([current_accumulation,current_temperature]).all()
                or current_accumulation<self._previous_accumulation-1e-10
                or forcing_valid not in (False,True,0,1)):
            raise ValueError('One finite nondecreasing scalar development state, temperature and validity flag are required.')
        a,t,valid=float(current_accumulation),float(current_temperature),bool(forcing_valid)
        if valid and self._ended:raise ValueError('A finished forcing window cannot resume.')
        visible=(a>=self.visibility)&valid
        unfolded=(a>=self.unfolding)&valid
        active=(self._previous_accumulation>=self.visibility)&valid&self._previous_valid
        area=np.clip((a-self.visibility)/(self.unfolding-self.visibility),0.,1.)*active
        active&=area>0.
        juvenile_capacity=(float(np.clip((self.thresholds[39]-a)/(self.thresholds[39]-self.thresholds[31]),0.,1.))
            if self.juvenile_policy=='handover_31_39' else 1.)
        juvenile=bool(self._previous_accumulation>=self.thresholds[10] and valid and self._previous_valid and juvenile_capacity>0.)
        active=np.append(active,juvenile)
        area=np.append(area,float(juvenile)*juvenile_capacity)
        juvenile_visible=bool(a>=self.thresholds[10] and valid and juvenile_capacity>0.)
        visible=np.append(visible,juvenile_visible)
        unfolded=np.append(unfolded,juvenile_visible)
        renewal=np.divide(np.maximum(area-self._previous_area,0.),area,out=np.zeros(8),where=area>0.)
        renewal[7]=float(-np.expm1(-max(t,0.)/300.))*juvenile
        renewal[~active]=0.
        progress={};crossed={};previous=0.;stage_code=0 if valid else -1
        for stage,threshold in self.thresholds.items():
            progress[stage]=float(np.clip((a-previous)/(threshold-previous),0.,1.)) if valid else 0.
            reached=valid and a>=threshold
            crossed[stage]=bool(reached and not self._reached[stage])
            if reached:stage_code=stage
            previous=threshold
        return DailyLeafRates(*[_immutable_array(x) for x in (active,renewal,area,visible,unfolded)],
            int(visible[:7].sum()),int(unfolded[:7].sum()),_immutable_array(visible[:3]),_immutable_array(unfolded[:3]),
            stage_code,MappingProxyType(progress),MappingProxyType(crossed),self.config_identity,self._owner_identity,self._day_count,a,valid)

    def integrate(self,rates):
        if (not isinstance(rates,DailyLeafRates) or rates.config_identity!=self.config_identity
                or rates.owner_identity!=self._owner_identity
                or rates.previous_day_count!=self._day_count):
            raise ValueError('Leaf rates must match the current configuration and integration state.')
        self._previous_accumulation=rates.next_accumulation
        self._previous_area=np.asarray(rates.area).copy()
        self._previous_valid=rates.forcing_valid
        self._ended=self._ended or (self._started and not rates.forcing_valid)
        self._started=self._started or rates.forcing_valid
        for stage,crossed in rates.newly_crossed_stage.items():self._reached[stage]|=crossed
        self._day_count+=1
        return DailyLeafState(rates.active,rates.renewal,rates.area,rates.visible,rates.fully_unfolded,
            rates.modeled_visible_final_leaf_count,rates.modeled_unfolded_final_leaf_count,rates.top3_visible,
            rates.top3_fully_unfolded,rates.predicted_stage_code,rates.stage_progress,rates.newly_crossed_stage)

    def step(self,current_accumulation,current_temperature,forcing_valid=True):
        return self.integrate(self.calc_rates(current_accumulation,current_temperature,forcing_valid))

    def snapshot(self):
        return dict(schema_version=1,config_identity=self.config_identity,previous_accumulation=self._previous_accumulation,
            previous_area=self._previous_area.tolist(),previous_valid=self._previous_valid,started=self._started,
            ended=self._ended,day_count=self._day_count,reached={str(k):v for k,v in self._reached.items()})

    def restore(self,snapshot):
        if snapshot.get('schema_version')!=1 or snapshot.get('config_identity')!=self.config_identity:
            raise ValueError('Leaf checkpoint configuration identity differs.')
        area=np.asarray(snapshot['previous_area'],float)
        a=float(snapshot['previous_accumulation']);count=snapshot['day_count']
        reached={int(k):bool(v) for k,v in snapshot['reached'].items()}
        if (area.shape!=(8,) or not np.isfinite(area).all() or np.any((area<0)|(area>1))
                or not np.isfinite(a) or a<0 or count!=int(count) or count<0 or set(reached)!=set(self.thresholds)):
            raise ValueError('Invalid leaf checkpoint state.')
        self._previous_area=area.copy();self._previous_accumulation=a;self._day_count=int(count)
        self._previous_valid=bool(snapshot['previous_valid']);self._started=bool(snapshot['started'])
        self._ended=bool(snapshot['ended']);self._reached=reached
        self._owner_identity=uuid4().hex

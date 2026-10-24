"""Independent support, arithmetic and held-GCM audit of the sowing experiment."""
from pathlib import Path
import json

import numpy as np
import pandas as pd

from models.climate.run import digest
from paper.analyses.food_security.provenance import write_json


def audit(output):
    output=Path(output)
    receipt=json.loads((output/"receipt.json").read_text())
    for name,expected in receipt["output_hashes"].items():
        assert digest(output/name)==expected,name
    frame=pd.read_parquet(output/"adaptation_pairs.parquet")
    assert len(frame)==57600
    identity=["model","reference_year","spatial_draw_id"]
    assert not frame.duplicated([*identity,"period","shift_days"]).any()
    assert set(frame.shift_days)=={-14,-7,0,7,14}
    assert frame.loc[frame.common_three_gcm_panel].cell_id.nunique()==57
    strict=frame.groupby(["reference_year","spatial_draw_id"]).shared_interval_nonempty.all()
    expected=strict.reindex(pd.MultiIndex.from_frame(frame[["reference_year","spatial_draw_id"]])).to_numpy()
    np.testing.assert_array_equal(frame.common_three_gcm_panel,expected)
    assert int(strict.sum())==1656
    main=frame.loc[frame.common_three_gcm_panel]
    assert main.whole_proxy_days.notna().all() and main.shared_proxy_days.notna().all()
    baseline=frame.loc[frame.shift_days.eq(0)].set_index([*identity,"period"])
    q=[c for c in frame if c.startswith("delta_")]
    for column in q:
        source=column[len("delta_"):]
        base=baseline[source].reindex(pd.MultiIndex.from_frame(frame[[*identity,"period"]])).to_numpy()
        np.testing.assert_allclose(frame[column],frame[source]-base,rtol=0,atol=1e-12,equal_nan=True)
    # Every GCM uses identical draw/year weights and all-option support.
    weights=main.loc[main.period.eq("future")&main.shift_days.eq(0)].groupby("model").area_mean_weight.sum()
    np.testing.assert_allclose(weights,weights.iloc[0],rtol=0,atol=1e-12)
    for filename,groups in (("fixed_shift_summary.csv",["period","shift_days","model"]),
                            ("fixed_shift_regions.csv",["environment_region","period","shift_days","model"])):
        summary=pd.read_csv(output/filename)
        for key,part in main.groupby(groups):
            key=key if isinstance(key,tuple) else (key,)
            mask=np.ones(len(summary),bool)
            for column,value in zip(groups,key):mask &= summary[column].eq(value)
            shown=summary.loc[mask]
            assert len(shown)==1
            for column in ("whole_proxy_days","shared_proxy_days",*q):
                expected=float(np.average(part[column],weights=part.area_mean_weight))
                np.testing.assert_allclose(shown[column].item(),expected,rtol=0,atol=1e-11)
    regional=pd.read_csv(output/"fixed_shift_regions.csv")
    policy=pd.read_csv(output/"leave_gcm_out_region_policy.csv")
    for choice in policy.itertuples():
        part=regional.loc[regional.environment_region.eq(choice.environment_region)&regional.period.eq("future")]
        training=part.loc[~part.model.isin(["ensemble",choice.held_model])]
        if choice.eligible_region:
            objective=training.groupby("shift_days").whole_proxy_days.mean()
            selected=min(objective.index,key=lambda k:(objective.loc[k],abs(k),k))
            assert selected==choice.selected_shift_days
        else:assert choice.selected_shift_days==0
        held=part.loc[part.model.eq(choice.held_model)&part.shift_days.eq(choice.selected_shift_days)].iloc[0]
        np.testing.assert_allclose(choice.held_delta_whole_proxy_days,held.delta_whole_proxy_days,atol=1e-12)
        np.testing.assert_allclose(choice.held_delta_shared_proxy_days,held.delta_shared_proxy_days,atol=1e-12)
    checks=pd.read_csv(output/"baseline_replay.csv")
    assert len(checks)==180 and checks.matches_retained_original.all()
    assert checks.maximum_absolute_difference.max()<1e-10
    support=pd.read_csv(output/"support.csv")
    assert (support.zero_overlap_complete_seasons>=0).all()
    assert int(support.zero_overlap_complete_seasons.sum())==556
    audit_record=dict(status="passed",records=len(frame),baseline_season_comparisons=180,
        maximum_baseline_difference=float(checks.maximum_absolute_difference.max()),
        all_option_three_gcm_common_draw_years=1656,common_unique_cells=57,
        zero_overlap_complete_seasons=556,delta_columns_verified=q,
        validation="independently recomputed common support, weighted endpoint/contrast means and training-only regional choices",
        source_sha256=digest(Path(__file__)),receipt_sha256=digest(output/"receipt.json"))
    write_json(output/"audit.json",audit_record)
    return audit_record


if __name__=="__main__":
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output",type=Path)
    print(json.dumps(audit(parser.parse_args().output),indent=2))

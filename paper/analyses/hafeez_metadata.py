"""Authoritative assay metadata from source worksheet fields."""
import pandas as pd


def isolate_metadata(frame,sheet):
    if 'Isolate' in frame:
        return frame.Isolate.astype(str).str.strip()
    panel={'IPO323_results_AUDPC':'IPO323','IPOO88004_raw':'IPO88004',
           'IPO90012_scores_raw':'IPO90012','ArinaEMS_IPO88004':'IPO88004'}
    if sheet not in panel:
        raise ValueError(f'Unresolved isolate source for worksheet{sheet}.')
    return pd.Series(panel[sheet],index=frame.index)


def leaf_organ_metadata(sheet):
    return 'second_leaf_6cm_section' if sheet=='ArinaEMS_IPO88004' else 'seedling_leaf'


def transgenic_sheet_names(names):
    return [name for name in names if 'transgenics' in name]

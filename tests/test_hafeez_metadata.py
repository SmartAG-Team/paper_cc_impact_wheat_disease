"""A source isolate must survive metadata harmonization."""
import importlib
import pandas as pd
import pytest


def helper():
    try:
        return importlib.import_module('analysis.hafeez_metadata')
    except ModuleNotFoundError:
        pytest.fail('Assay metadata reconciliation is not implemented')


def test_transgenic_isolate_comes_from_authoritative_source_column():
    h=helper()
    f=pd.DataFrame({'Isolate':['IPO88004','IPO92006']})
    assert h.isolate_metadata(f,'septoria_transgenics_test2').tolist()==['IPO88004','IPO92006']


def test_panel_typographic_sheet_name_maps_to_declared_isolate():
    h=helper()
    f=pd.DataFrame({'Name':['a','b']})
    assert h.isolate_metadata(f,'IPOO88004_raw').tolist()==['IPO88004','IPO88004']


def test_ems_leaf_organ_matches_second_leaf_section():
    h=helper()
    assert h.leaf_organ_metadata('ArinaEMS_IPO88004')=='second_leaf_6cm_section'


def test_both_source_transgenic_sheet_prefixes_are_discovered():
    h=helper()
    names=['IPO323_results_AUDPC','septoria_transgenics_test1','stb15_transgenics_test2']
    assert h.transgenic_sheet_names(names)==['septoria_transgenics_test1','stb15_transgenics_test2']

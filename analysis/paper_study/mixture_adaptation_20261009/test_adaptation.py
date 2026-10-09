"""Scientific contracts: experimental units, honest controls and inference scope."""
import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent


class AdaptationContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        spec = importlib.util.spec_from_file_location("mixture_core", HERE / "core.py")
        if spec is None or not (HERE / "core.py").exists():
            cls.core = None
        else:
            cls.core = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = cls.core
            spec.loader.exec_module(cls.core)

    def setUp(self):
        self.assertIsNotNone(self.core, "The experimental-unit/control module is not implemented")

    def test_unit_conversion_is_source_specific(self):
        self.assertAlmostEqual(self.core.yield_t_ha(500, "g_m2"), 5)
        self.assertAlmostEqual(self.core.yield_t_ha(50, "dt_ha"), 5)
        with self.assertRaises(ValueError):
            self.core.yield_t_ha(50, "ordinal_severity")

    def test_rows_aggregate_to_one_plot_and_missing_row_stays_missing(self):
        self.assertEqual(self.core.complete_mean([100, 100, 200, 200]), 150)
        self.assertTrue(np.isnan(self.core.complete_mean([100, 100, 200, np.nan])))
        self.assertTrue(np.isnan(self.core.complete_mean([100, 100, 200])))

    def test_plot_total_ratio_differs_from_component_ryt(self):
        r = self.core.yield_contrast(6, [2, 8])
        self.assertEqual(r["baseline"], 5)
        self.assertEqual(r["delta"], 1)
        self.assertEqual(r["relative_percent"], 20)
        self.assertNotEqual(1 + r["relative_percent"] / 100, np.mean([3 / 2, 9 / 8]))

    def test_incomplete_constituents_fail_closed(self):
        with self.assertRaises(ValueError):
            self.core.yield_contrast(6, [4, np.nan])
        with self.assertRaises(ValueError):
            self.core.yield_contrast(6, [0, 0])

    def test_equal_trial_weight_not_plot_weight(self):
        a = pd.DataFrame({"environment_id": ["A"] * 10 + ["B"],
                          "composition_id": ["x"] * 10 + ["y"], "delta": [1.] * 10 + [5.]})
        means = self.core.environment_means(a, "delta")
        self.assertEqual(means.delta.mean(), 3)

    def test_equal_composition_weight_precedes_equal_trial_weight(self):
        a = pd.DataFrame({"environment_id": ["A"] * 4, "composition_id": ["x", "x", "x", "y"], "delta": [1, 1, 1, 5]})
        self.assertEqual(self.core.environment_means(a, "delta").delta.iloc[0], 3)

    def test_shared_controls_form_connected_groups(self):
        a = pd.DataFrame({"constituent_1": ["a", "b", "x"], "constituent_2": ["b", "c", "y"]})
        groups = self.core.dependency_groups(a)
        self.assertEqual(groups[0], groups[1])
        self.assertNotEqual(groups[0], groups[2])

    def test_site_resampling_keeps_years_together(self):
        a = pd.DataFrame({"site_id": ["a", "a", "b", "b"], "delta": [-100., 102., 2., 2.]})
        ci = self.core.site_bootstrap(a, "delta", draws=2000)
        self.assertGreaterEqual(ci[0], 1)
        self.assertLessEqual(ci[1], 2)
        self.assertEqual(ci, self.core.site_bootstrap(a, "delta", draws=2000))

    def test_group_holdout_has_no_site_leakage(self):
        a = pd.DataFrame({"environment_id": ["a_1", "a_2", "b_1", "b_2"], "site_id": ["a", "a", "b", "b"], "year": [1, 2, 1, 2], "delta": [1., 3., 10., 20.]})
        out = self.core.transfer_predictions(a, "delta")
        h = out[out.scheme.eq("leave_site_out")]
        self.assertTrue((h.loc[h.site_id.eq("a"), "prediction"] == 15).all())
        self.assertTrue((h.loc[h.site_id.eq("b"), "prediction"] == 2).all())
        f = out[out.scheme.eq("forward_year")]
        self.assertEqual(set(f.year), {2})
        self.assertTrue((f.prediction == 5.5).all())

    def test_release_mapping_quarantines_ambiguity(self):
        current = pd.DataFrame({"year": [2020], "yield_dtha": [60.]})
        previous = pd.DataFrame({"year": [2020, 2020], "yield_dtha": [60., 60.], "ww": [1, 2], "plot_id": [None, None]})
        matched, audit = self.core.map_release(current, previous)
        self.assertEqual(len(matched), 0)
        self.assertEqual(audit.mapping_status.iloc[0], "ambiguous")

    def test_release_duplicate_aliases_are_not_replicates(self):
        current = pd.DataFrame({"year": [2020, 2020], "yield_dtha": [60., 60.]})
        previous = pd.DataFrame({"year": [2020], "yield_dtha": [60.], "ww": [1], "plot_id": [None]})
        matched, audit = self.core.map_release(current, previous)
        self.assertEqual(len(matched), 1)
        self.assertEqual(matched.current_source_lines.iloc[0], "2|3")

    def test_kernel_weight_uses_kernel_number_not_arithmetic_mean(self):
        self.assertAlmostEqual(self.core.kernel_weight([100, 300], [20, 60]), 40)
        self.assertTrue(np.isnan(self.core.kernel_weight([100, 300], [20, np.nan])))

    def test_control_baseline_never_crosses_source_trial_codes(self):
        controls = pd.DataFrame({"environment_id": ["e", "e"], "control_group": ["1", "2"], "cultivar": ["a", "b"], "yield_t_ha": [4., 8.], "plot_id": ["p1", "p2"]})
        self.assertIsNone(self.core.control_baseline(controls, "e", "1", ["a", "b"], "yield_t_ha"))
        self.assertIsNone(self.core.control_baseline(controls, "wrong", "1", ["a"], "yield_t_ha"))

    def test_controls_average_within_cultivar_before_between_cultivars(self):
        controls = pd.DataFrame({"environment_id": ["e"] * 4, "control_group": ["1"] * 4, "cultivar": ["a", "a", "a", "b"], "yield_t_ha": [2., 2., 2., 8.], "plot_id": ["a1", "a2", "a3", "b"]})
        r = self.core.control_baseline(controls, "e", "1", ["a", "b"], "yield_t_ha")
        self.assertEqual(r["values"], [2., 8.])
        self.assertEqual(r["control_ids"], ["a1|a2|a3", "b"])

    def test_unsupported_claims_remain_false(self):
        a = self.core.applicability()
        for source in ("france", "swiss"):
            for claim in ("validated_climate_adaptation", "stb_mediated_grain_loss", "european_production_benefit", "independent_environment_validation"):
                self.assertIs(a[source][claim], False)
        self.assertIs(a["swiss"]["causal_constituent_contrast"], False)
        self.assertIs(a["france"]["randomized_within_trial_allocation"], True)


if __name__ == "__main__":
    unittest.main()

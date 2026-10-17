"""Integration checks for the Python API and configurable C++ runner.

Run: python tests/test_experiment_interface.py --binary build-paper-release/carbonfund_experiment
Optional reference programs are the existing benchmark and hedging example.
"""
from __future__ import annotations

import argparse
import copy
import csv
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np
from scipy.stats import t
import carbonfund_experiments as cf

BINARY = None
BENCHMARK = None
HEDGER = None


def mix(x):
    mask = 2**64 - 1
    x = (x + 0x9E3779B97F4A7C15) & mask
    x = ((x ^ (x >> 30)) * 0xBF58476D1CE4E5B9) & mask
    x = ((x ^ (x >> 27)) * 0x94D049BB133111EB) & mask
    return x ^ (x >> 31)


class ExperimentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.binary = cf._find_binary(BINARY)

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.index = 0

    def tearDown(self):
        self.temporary.cleanup()

    def run_case(self, config, plots=False):
        self.index += 1
        return cf.run_experiment(config, self.root / str(self.index), self.binary,
                                 plots=plots, progress=False)

    def small_pricing(self, model="CIR"):
        c = cf.demo_config("pricing", "EI")
        c["pricing"].update(paths=101, intervals=8, repetitions=3)
        state = [1.0, 1.2, .8, 1.1]
        c["carbon"] = {"model": model, "initial": state,
                       "mean_reversion": [.3] * 4, "volatility": [.2] * 4}
        c["carbon"]["long_run_log_mean" if model == "GeoOU" else "long_run_mean"] = (
            np.log(state).tolist() if model == "GeoOU" else state)
        return c

    def small_hedging(self):
        c = cf.demo_config("hedging")
        c["population"] = [
            {"type": kind, "age": 60 + i % 3, "notional": 1000 + 100 * i}
            for kind in ("PE", "TI", "EI") for i in range(10)]
        c["hedging"].update(outer_scenarios=12, intervals=5, progress_every=0)
        c["hedging"]["initial"].update(paths=64, intervals=12)
        c["hedging"]["revaluation"].update(paths=16, intervals=12)
        return c

    def test_all_models_counts_variance_and_repeatability(self):
        for model in ("OU", "GeoOU", "CIR"):
            with self.subTest(model=model):
                c = self.small_pricing(model)
                first, second = self.run_case(c), self.run_case(c)
                self.assertEqual(len(first.summary()), 8)
                for a, b in zip(first.summary(), second.summary()):
                    self.assertEqual(a["mean"], b["mean"])
                    self.assertEqual(a["sample_variance"], b["sample_variance"])
                    self.assertEqual(a["actual_paths"], 102)
                    n = 51 if "antithetic" in a["configuration"] else 102
                    self.assertEqual(a["independent_samples"], n)
                    self.assertAlmostEqual(a["variance_of_mean"],
                                           a["sample_variance"] / n, places=14)
                    self.assertAlmostEqual(a["standard_deviation"]**2,
                                           a["sample_variance"], places=14)
                    self.assertAlmostEqual(a["effective_variance"],
                                           102 * a["variance_of_mean"], places=14)
                    critical = t.isf(.025, n - 1)
                    self.assertAlmostEqual(a["ci_high"],
                        a["mean"] + critical * a["standard_error"], places=12)
                base = {r["quantity"]: r["variance_of_mean"] for r in first.summary()
                        if r["configuration"] == "standard"}
                for row in first.summary():
                    self.assertAlmostEqual(row["variance_reduction_percent"],
                        100 * (1 - row["variance_of_mean"] / base[row["quantity"]]), places=10)

    def test_pricing_normalization_and_paired_repetitions(self):
        c = self.small_pricing()
        c["population"] = cf.homogeneous_population("EI", 3, 60, 7)
        result = self.run_case(c)
        for raw, scaled in zip(result.summary("aggregate"), result.summary("per_notional")):
            self.assertAlmostEqual(scaled["mean"], raw["mean"] / 21)
            self.assertAlmostEqual(scaled["sample_variance"], raw["sample_variance"] / 21**2)
        comparisons = result.comparisons()
        self.assertEqual(len(comparisons), 12)
        row = comparisons[0]
        series = result.pricing["repetitions"]
        a = [r["mean"] for r in series
             if r["configuration"] == row["reference"] and r["quantity"] == row["quantity"]]
        b = [r["mean"] for r in series
             if r["configuration"] == row["candidate"] and r["quantity"] == row["quantity"]]
        differences = np.array(b) - a
        self.assertAlmostEqual(row["mean"], float(differences.mean()))
        self.assertAlmostEqual(row["variance"], float(np.var(differences, ddof=1)))
        self.assertEqual(row["simultaneous_family_size"], 12)
        self.assertLessEqual(row["simultaneous_ci_low"], row["mean_ci_low"])
        self.assertEqual(len(result.repetition_statistics()), 8)

    def test_no_false_comparison_precision_with_one_repetition(self):
        c = self.small_pricing()
        c["pricing"]["repetitions"] = 1
        result = self.run_case(c)
        self.assertTrue(all(math.isnan(r["mean_se"]) for r in result.comparisons()))
        self.assertTrue(all(math.isnan(r["observed_variance_of_estimates"])
                            for r in result.repetition_statistics()))

    def test_distinct_repetition_seeds_even_for_close_input_seeds(self):
        c = self.small_pricing()
        c["pricing"].update(paths=10, repetitions=40, carbon_seed=1, financial_seed=2)
        result = self.run_case(c)
        records = [r for r in result.pricing["repetitions"]
                   if r["quantity"] == "price" and r["configuration"] == "standard"]
        seeds = [r[k] for r in records for k in ("carbon_seed", "financial_seed")]
        self.assertEqual(len(seeds), len(set(seeds)))
        self.assertEqual(seeds[:2], [1, 2])

    def test_hedging_accounting_budget_independence_and_scaling(self):
        c = self.small_hedging()
        first = self.run_case(c)
        c["hedging"]["revaluation"]["paths"] = 64
        second = self.run_case(c)
        def records(r):
            with (r.directory / "hedging.csv").open() as f:
                return list(csv.DictReader(f))
        a, b = records(first), records(second)
        for old, new in zip(a, b):
            for key in ("claims_pv", "fund_terminal", "initial_price", "static_hedge", "no_hedge"):
                self.assertEqual(old[key], new[key], key)
            for name, gains in (("no_hedge", 0),
                                ("static_hedge", float(old["static_trading_gains_pv"])),
                                ("dynamic_hedge", float(old["dynamic_trading_gains_pv"]))):
                identity = float(old["claims_pv"]) - float(old["initial_price"]) - gains
                self.assertAlmostEqual(float(old[name]), identity, delta=1e-9)
        c["hedging"]["revaluation"]["paths"] = 16
        for person in c["population"]:
            person["notional"] *= 7
        scaled = self.run_case(c)
        np.testing.assert_allclose(scaled.costs, first.costs * 7, rtol=1e-10, atol=1e-8)
        for raw, scale in zip(first.summary(), scaled.summary()):
            np.testing.assert_allclose(scale["variance"], raw["variance"] * 49,
                                       rtol=1e-12, atol=1e-9)

    def test_hedging_supports_standard_valuation(self):
        c = self.small_hedging()
        c["hedging"]["initial"]["configuration"] = "standard"
        c["hedging"]["revaluation"]["configuration"] = "standard_antithetic"
        result = self.run_case(c)
        self.assertTrue(np.all(np.isfinite(result.costs)))

    def test_validation_catches_bad_dimensions_seeds_and_matrix(self):
        c = self.small_pricing()
        mutations = [
            lambda x: x["market"].update(covariance=np.eye(4).tolist()),
            lambda x: x["market"].update(volatilities=[.2] * 3),
            lambda x: x["market"]["correlations"][0].__setitem__(0, .9),
            lambda x: x["market"].update(correlations=[[1] * 4 for _ in range(4)]),
            lambda x: x["pricing"].update(paths=True),
            lambda x: x["pricing"].update(financial_seed=x["pricing"]["carbon_seed"]),
            lambda x: x["analysis"].update(comparison_family_size=1),
            lambda x: x["analysis"].update(confidence_level=1),
        ]
        for change in mutations:
            invalid = copy.deepcopy(c)
            change(invalid)
            with self.assertRaises(ValueError):
                cf.validate_config(invalid)

    def test_removed_fund_sampling_field_is_rejected(self):
        cases = (("pricing", ("pricing",)),
                 ("hedging", ("hedging", "initial")),
                 ("hedging", ("hedging", "revaluation")))
        for index, (experiment, location) in enumerate(cases):
            with self.subTest(location=location):
                config = cf.demo_config(experiment)
                section = config
                for key in location:
                    section = section[key]
                section["fund_sampling"] = "path"
                with self.assertRaisesRegex(ValueError, "fund_sampling"):
                    cf.validate_config(config)
                path = self.root / f"removed-{index}.json"
                path.write_text(json.dumps(config))
                result = subprocess.run(
                    [str(self.binary), "--config", str(path),
                     "--output", str(self.root / f"removed-{index}")],
                    capture_output=True, text=True)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("fund_sampling", result.stderr)

    def test_backend_checks_inputs_when_python_is_bypassed(self):
        c = self.small_pricing()
        c["market"]["volatilities"] = [.2] * 3
        path = self.root / "bad.json"
        path.write_text(json.dumps(c))
        result = subprocess.run([str(self.binary), "--config", str(path),
                                 "--output", str(self.root / "bad")], capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("four", result.stderr)
        self.assertFalse((self.root / "bad" / "metadata.json").exists())

    def test_plots_exports_and_zero_variance(self):
        c = self.small_pricing()
        c["market"]["expected_returns"] = [c["market"]["risk_free_rate"]] * 4
        result = self.run_case(c, plots=True)
        self.assertTrue(all(r["sample_variance"] == 0 for r in result.summary()))
        for name in ("pricing.png", "pricing.pdf", "summary.csv", "paired_comparisons.csv",
                     "repetition_statistics.csv", "analysis.json"):
            self.assertGreater((result.directory / name).stat().st_size, 0)
        hedge = self.run_case(self.small_hedging(), plots=True)
        self.assertGreater((hedge.directory / "distributions.png").stat().st_size, 0)
        self.assertEqual(len(cf.load_result(hedge.directory).summary()), 3)

    def test_expected_shortfall_fractional_endpoint(self):
        x = np.arange(10, dtype=float)
        self.assertAlmostEqual(cf.expected_shortfall(x, .85), (9 + .5 * 8) / 1.5)

    def test_existing_hedging_example_agrees(self):
        reference = Path(HEDGER) if HEDGER else self.binary.parent / "example_petiei_hedging"
        if not reference.is_file():
            self.skipTest("Build existing example_petiei_hedging for the reference comparison")
        c = self.small_hedging()
        c["hedging"].update(intervals=20)
        c["hedging"]["initial"].update(paths=8192, intervals=50)
        c["hedging"]["revaluation"].update(paths=128, intervals=50)
        result = self.run_case(c)
        prefix = self.root / "reference"
        subprocess.run([str(reference), "12", "128", str(prefix)],
                       check=True, capture_output=True, text=True)
        with prefix.with_suffix(".csv").open() as f:
            expected = list(csv.DictReader(f))
        with (result.directory / "hedging.csv").open() as f:
            actual = list(csv.DictReader(f))
        for a, b in zip(actual, expected):
            for key in a:
                self.assertAlmostEqual(float(a[key]), float(b[key]), delta=2e-9, msg=key)

    def test_existing_paper_benchmark_agrees_all_maturities_and_contracts(self):
        reference = Path(BENCHMARK) if BENCHMARK else self.binary.parent / "benchmark_petiei_variance_reduction"
        if not reference.is_file():
            self.skipTest("Build the existing variance benchmark for the reference comparison")
        prefix = self.root / "paper"
        subprocess.run([str(reference), "--paths", "1000", "--repetitions", "1",
                        "--seed", "20261015", "--batches", "0",
                        "--output-prefix", str(prefix)], check=True, capture_output=True, text=True)
        with Path(str(prefix) + "_summary.csv").open() as f:
            expected = list(csv.DictReader(f))
        for maturity in (5, 10, 20, 30):
            for ci, contract in enumerate(("PE", "TI", "EI")):
                c = cf.paper_config("pricing", contract, maturity=maturity)
                c["analysis"]["normalization"] = "aggregate"
                c["pricing"].update(paths=1000, repetitions=1)
                ti = (5, 10, 20, 30).index(maturity)
                tag = mix(20261015 ^ ((ti + 1) << 48) ^ ((ci + 1) << 40))
                seed = mix(tag)
                c["pricing"].update(carbon_seed=mix(seed ^ 0x434152424F4E),
                                    financial_seed=mix(seed ^ 0x46494E414E4345))
                actual = self.run_case(c).summary()
                for row in actual:
                    target = next(x for x in expected if int(x["T"]) == maturity
                        and x["contract"] == contract and x["configuration"] == row["configuration"]
                        and x["quantity"] == row["quantity"])
                    for key in ("mean", "sample_variance", "standard_error"):
                        self.assertAlmostEqual(row[key], float(target[key]),
                                               delta=2e-11, msg=f"{maturity} {contract} {key}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary")
    parser.add_argument("--reference-benchmark")
    parser.add_argument("--reference-hedger")
    options, rest = parser.parse_known_args()
    BINARY, BENCHMARK, HEDGER = options.binary, options.reference_benchmark, options.reference_hedger
    unittest.main(argv=[sys.argv[0], *rest], verbosity=2)

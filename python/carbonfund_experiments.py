"""Configure CarbonFund_UL experiments and analyse their C++ results.

All simulation loops run in the carbonfund_experiment executable.
Returned Table objects display as HTML tables in Jupyter without pandas.
"""
from __future__ import annotations

import argparse
import copy
import csv
import html
import itertools
import json
import math
from pathlib import Path
import subprocess
import sys
from typing import Any

import numpy as np
from scipy.stats import t as student_t

MODES = ("standard", "standard_antithetic", "conditioned", "conditioned_antithetic")
STRATEGIES = ("no_hedge", "static_hedge", "dynamic_hedge")
LABELS = ("No hedge", "Static hedge", "Dynamic hedge")
COLORS = ("#4477AA", "#EE7733", "#228833")


def _native(value: Any) -> Any:
    """Convert NumPy inputs to the strict JSON representation used by C++."""
    if isinstance(value, np.ndarray):
        return _native(value.tolist())
    if isinstance(value, np.generic):
        return _native(value.item())
    if isinstance(value, dict):
        return {key: _native(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_native(item) for item in value]
    return value


class Table(list):
    """A list of dictionaries with a notebook HTML representation."""

    def _repr_html_(self) -> str:
        if not self:
            return "<p>No observations available.</p>"
        def cell(value: Any) -> str:
            if isinstance(value, float):
                return "" if not math.isfinite(value) else f"{value:.10g}"
            return html.escape(str(value))
        headings = "".join(f"<th>{html.escape(str(k))}</th>" for k in self[0])
        rows = "".join("<tr>" + "".join(f"<td>{cell(v)}</td>" for v in row.values())
                       + "</tr>" for row in self)
        return ('<div style="overflow:auto"><table><thead><tr>' + headings
                + "</tr></thead><tbody>" + rows + "</tbody></table></div>")


def homogeneous_population(contract: str, count: int, age: float,
                           notional: float = 1.0) -> list[dict]:
    if contract not in ("PE", "TI", "EI"):
        raise ValueError("contract must be PE, TI or EI")
    if isinstance(count, bool) or not isinstance(count, int) or count < 1:
        raise ValueError("policyholder count must be a positive integer")
    return [{"type": contract, "age": age, "notional": notional} for _ in range(count)]


def demo_config(experiment: str = "pricing", contract: str = "PE") -> dict:
    """Fully specified small example; every returned field is editable."""
    if experiment not in ("pricing", "hedging"):
        raise ValueError("experiment must be pricing or hedging")
    valuation = {
        "paths": 128, "intervals": 50, "configuration": "conditioned_antithetic",
        "carbon_seed": 2026101604, "financial_seed": 2026101606,
    }
    initial = {**valuation, "paths": 8192, "carbon_seed": 2026101605,
               "financial_seed": 2026101607}
    valuation["minimum_intervals"] = 1
    return {
        "schema_version": 1, "experiment": experiment,
        "market": {
            "risk_free_rate": 0.03, "expected_returns": [0.08, 0.10, 0.07, 0.09],
            "volatilities": [0.20, 0.25, 0.18, 0.22],
            "correlations": np.eye(4).tolist(),
        },
        "carbon": {
            "model": "CIR", "initial": [5.0, 4.0, 3.0, 1.0],
            "long_run_mean": [2.5, 2.0, 1.5, 0.5],
            "mean_reversion": [0.1] * 4, "volatility": [0.2] * 4,
        },
        "fund": {"risk_aversion": 2.0, "carbon_penalty": 0.1},
        "contract": {
            "fund_value": 1.2, "maturity": 5.0, "minimum_guarantee": 1.0,
            "maximum_guarantee": 1.5, "minimum_guarantee_growth_rate": 0.01,
            "maximum_guarantee_growth_rate": 0.04,
            "endowment_death_benefit_fraction": 0.8,
        },
        "mortality": {"xi": 0.0041959, "m": 79.6921211, "b": 11.5818911},
        "population": homogeneous_population(contract, 1, 60.0),
        "pricing": {
            "contract": contract, "paths": 10000, "intervals": 50,
            "configurations": list(MODES), "repetitions": 3,
            "carbon_seed": 20261015, "financial_seed": 20261016,
        },
        "hedging": {
            "outer_scenarios": 100, "intervals": 20, "progress_every": 10,
            "threads": 0,
            "carbon_seed": 2026101601, "financial_seed": 2026101602,
            "mortality_seed": 2026101603, "initial": initial, "revaluation": valuation,
        },
        "analysis": {
            "normalization": "aggregate", "confidence_level": 0.95,
            "comparison_family_size": 0,
            "quantiles": [0.025, 0.5, 0.9, 0.95, 0.975, 0.99],
            "expected_shortfall_levels": [0.95, 0.99],
            "bins": 70, "zoom": None, "plot_range": None,
            "figure_formats": ["png", "pdf"], "dpi": 180, "title": "",
        },
        "notes": "Illustrative inputs. This is not a reproduction of the paper's hedging figures.",
    }


def paper_config(experiment: str = "pricing", contract: str = "PE",
                 policyholders: int = 1, maturity: float = 20.0,
                 age: float = 60.0, notional: float = 1.0) -> dict:
    """Paper-calibration starting point with explicit, editable numerical budgets.

    Guarantees use the convention of the existing paper variance benchmark.
    A 5T hedging grid is an explicit starting choice, not a recovered paper input.
    """
    c = demo_config(experiment, contract)
    c["market"] = {
        "risk_free_rate": 0.05, "expected_returns": [0.25, 0.15, 0.10, 0.08],
        "volatilities": [0.30, 0.25, 0.20, 0.16],
        "correlations": [[1, .44, .39, .32], [.44, 1, .30, .33],
                         [.39, .30, 1, .31], [.32, .33, .31, 1]],
    }
    c["carbon"] = {
        "model": "CIR", "initial": [5000, 4000, 3000, 1000],
        "long_run_mean": [2500, 2000, 1500, 500],
        "mean_reversion": [.05] * 4, "volatility": [3.0] * 4,
    }
    c["fund"] = {"risk_aversion": 1.0, "carbon_penalty": .0025}
    c["contract"] = {
        "fund_value": 1.0, "maturity": maturity, "minimum_guarantee": 1.0,
        "maximum_guarantee": 1.0, "minimum_guarantee_growth_rate": .05,
        "maximum_guarantee_growth_rate": .5, "endowment_death_benefit_fraction": 1.0,
    }
    c["population"] = homogeneous_population(contract, policyholders, age, notional)
    intervals = max(1, math.ceil(5 * maturity))
    c["pricing"].update(paths=1_000_000, intervals=intervals)
    c["hedging"].update(outer_scenarios=10_000, intervals=intervals, progress_every=100)
    c["hedging"]["initial"].update(paths=8192, intervals=intervals)
    c["hedging"]["revaluation"].update(paths=512, intervals=intervals)
    c["analysis"]["normalization"] = "per_policy"
    c["notes"] = (
        "Financial, CIR and mortality calibration from the paper; corrected xi and age+t-m. "
        "Guarantees match the existing paper variance benchmark. Hedging/pricing budgets "
        "are explicit starting choices. Costs use corrected time-zero accounting and "
        "mortality-adjusted static deltas. Verify conventions before comparing published numbers."
    )
    return c


def _keys(value: Any, allowed: set[str], required: set[str], name: str) -> dict:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a dictionary")
    missing, extra = required - value.keys(), value.keys() - allowed
    if missing or extra:
        raise ValueError(f"{name}: missing={sorted(missing)}, unknown={sorted(extra)}")
    return value


def _numeric(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    return float(value)


def _integer(value: Any, name: str, minimum: int = 0, maximum: int = 2**64 - 1) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not minimum <= value <= maximum:
        raise ValueError(f"{name} must be an integer between {minimum} and {maximum}")
    return value


def validate_config(config: dict) -> None:
    """Catch configuration errors before launching C++; C++ validates again."""
    config = _native(config)
    root_keys = {"schema_version", "experiment", "market", "carbon", "fund", "contract",
                 "mortality", "population", "pricing", "hedging", "analysis", "notes"}
    c = _keys(config, root_keys, root_keys - {"notes", "pricing", "hedging"}, "configuration")
    if c["schema_version"] != 1 or isinstance(c["schema_version"], bool):
        raise ValueError("schema_version must be 1")
    if c["experiment"] not in ("pricing", "hedging"):
        raise ValueError("experiment must be pricing or hedging")
    def exact(obj: Any, fields: set[str], name: str) -> dict:
        return _keys(obj, fields, fields, name)
    def vec(v: Any, name: str) -> np.ndarray:
        if not isinstance(v, list) or len(v) != 4:
            raise ValueError(f"{name} must contain exactly four numbers")
        return np.array([_numeric(x, name) for x in v])
    market = exact(c["market"], {"risk_free_rate", "expected_returns", "volatilities",
                                 "correlations"}, "market")
    _numeric(market["risk_free_rate"], "risk_free_rate")
    vec(market["expected_returns"], "expected_returns")
    vol = vec(market["volatilities"], "volatilities")
    rows = market["correlations"]
    if not isinstance(rows, list) or len(rows) != 4:
        raise ValueError("correlations must have four rows")
    corr = np.array([vec(row, "correlation row") for row in rows])
    if (np.any(vol <= 0) or not np.allclose(corr, corr.T, rtol=0, atol=1e-12)
            or not np.allclose(np.diag(corr), 1, rtol=0, atol=1e-12)
            or np.any(np.abs(corr) > 1 + 1e-12)):
        raise ValueError("Invalid volatilities or correlation matrix")
    try:
        np.linalg.cholesky(corr)
    except np.linalg.LinAlgError as error:
        raise ValueError("Correlation matrix must be positive definite") from error
    fund = exact(c["fund"], {"risk_aversion", "carbon_penalty"}, "fund")
    if _numeric(fund["risk_aversion"], "risk_aversion") <= 0:
        raise ValueError("risk_aversion must be positive")
    if _numeric(fund["carbon_penalty"], "carbon_penalty") < 0:
        raise ValueError("carbon_penalty must be nonnegative")
    carbon = c["carbon"]
    model = carbon.get("model") if isinstance(carbon, dict) else None
    if model not in ("OU", "GeoOU", "CIR"):
        raise ValueError("carbon.model must be OU, GeoOU or CIR")
    mean_key = "long_run_log_mean" if model == "GeoOU" else "long_run_mean"
    exact(carbon, {"model", "initial", mean_key, "mean_reversion", "volatility"}, "carbon")
    state = vec(carbon["initial"], "carbon.initial")
    mean = vec(carbon[mean_key], mean_key)
    kappa = vec(carbon["mean_reversion"], "mean_reversion")
    sigma = vec(carbon["volatility"], "carbon.volatility")
    if np.any(kappa < 0) or np.any(sigma < 0):
        raise ValueError("Carbon mean reversion and volatility must be nonnegative")
    if model == "GeoOU" and np.any(state <= 0):
        raise ValueError("GeoOU initial carbon must be positive")
    if model == "CIR" and (np.any(state < 0) or np.any(mean < 0)
                           or np.any(kappa * mean < sigma**2 / 4)):
        raise ValueError("CIR needs nonnegative states/means and kappa*mean >= volatility^2/4")
    terms = exact(c["contract"], {"fund_value", "maturity", "minimum_guarantee",
        "maximum_guarantee", "minimum_guarantee_growth_rate",
        "maximum_guarantee_growth_rate", "endowment_death_benefit_fraction"}, "contract")
    for key, value in terms.items():
        _numeric(value, key)
    if terms["fund_value"] <= 0 or terms["maturity"] <= 0:
        raise ValueError("Fund value and maturity must be positive")
    if not 0 <= terms["minimum_guarantee"] <= terms["maximum_guarantee"]:
        raise ValueError("Need 0 <= minimum_guarantee <= maximum_guarantee")
    if terms["endowment_death_benefit_fraction"] < 0:
        raise ValueError("EI death-benefit fraction must be nonnegative")
    mortality = exact(c["mortality"], {"xi", "m", "b"}, "mortality")
    if _numeric(mortality["xi"], "xi") < 0 or _numeric(mortality["b"], "b") <= 0:
        raise ValueError("Need xi >= 0 and b > 0")
    _numeric(mortality["m"], "m")
    if not isinstance(c["population"], list):
        raise ValueError("population must be a list")
    for p in c["population"]:
        exact(p, {"type", "age", "notional"}, "policyholder")
        if p["type"] not in ("PE", "TI", "EI"):
            raise ValueError("Policyholder type must be PE, TI or EI")
        if _numeric(p["age"], "age") < 0 or _numeric(p["notional"], "notional") < 0:
            raise ValueError("Ages and notionals must be nonnegative")
    max_count = min(sys.maxsize * 2, 2**64 - 2)
    def valuation(v: dict, name: str, *, revaluation: bool = False) -> list[int]:
        fields = {"paths", "intervals", "configuration", "carbon_seed", "financial_seed"}
        _keys(v, fields | ({"minimum_intervals"} if revaluation else set()), fields, name)
        if v["configuration"] not in MODES:
            raise ValueError(f"Unknown {name} configuration")
        minimum = 3 if "antithetic" in v["configuration"] else 2
        _integer(v["paths"], name + ".paths", minimum, max_count)
        _integer(v["intervals"], name + ".intervals", 1, max_count)
        if revaluation:
            _integer(v.get("minimum_intervals", 1), name + ".minimum_intervals",
                     1, v["intervals"])
        return [_integer(v[k], k) for k in ("carbon_seed", "financial_seed")]
    if c["experiment"] == "pricing":
        p = exact(c.get("pricing"), {"contract", "paths", "intervals", "configurations",
            "repetitions", "carbon_seed", "financial_seed"}, "pricing")
        if p["contract"] not in ("PE", "TI", "EI"):
            raise ValueError("pricing.contract must be PE, TI or EI")
        if any(x["type"] != p["contract"] for x in c["population"]):
            raise ValueError("Pricing requires a single contract type")
        modes = p["configurations"]
        if (not isinstance(modes, list) or not modes
                or any(x not in MODES for x in modes) or len(set(modes)) != len(modes)):
            raise ValueError("Select distinct estimator configurations")
        for name in modes:
            v = {k: p[k] for k in ("paths", "intervals",
                                    "carbon_seed", "financial_seed")}
            seeds = valuation({**v, "configuration": name}, "pricing")
            if len(set(seeds)) != 2:
                raise ValueError("Carbon and financial seeds must differ")
        _integer(p["repetitions"], "repetitions", 1, max_count)
    else:
        fields = {"outer_scenarios", "intervals", "progress_every", "threads",
                  "carbon_seed", "financial_seed", "mortality_seed", "initial", "revaluation"}
        h = _keys(c.get("hedging"), fields, fields - {"threads"}, "hedging")
        _integer(h["outer_scenarios"], "outer_scenarios", 2, max_count)
        _integer(h["intervals"], "hedging.intervals", 1, max_count)
        _integer(h["progress_every"], "progress_every", 0, max_count)
        _integer(h.get("threads", 1), "hedging.threads", 0, 2**31 - 1)
        seeds = [_integer(h[k], k) for k in ("carbon_seed", "financial_seed", "mortality_seed")]
        seeds += valuation(h["initial"], "initial")
        seeds += valuation(h["revaluation"], "revaluation", revaluation=True)
        if len(set(seeds)) != 7:
            raise ValueError("Use distinct seeds for all seven hedging streams")
    a = exact(c["analysis"], {"normalization", "confidence_level", "comparison_family_size", "quantiles",
        "expected_shortfall_levels", "bins", "zoom", "plot_range", "figure_formats",
        "dpi", "title"}, "analysis")
    if a["normalization"] not in ("aggregate", "per_policy", "per_notional"):
        raise ValueError("Unknown normalization")
    if not 0 < _numeric(a["confidence_level"], "confidence_level") < 1:
        raise ValueError("confidence_level must lie between 0 and 1")
    _integer(a["comparison_family_size"], "comparison_family_size", 0, max_count)
    contrasts = (len(c["pricing"]["configurations"]) *
                 (len(c["pricing"]["configurations"]) - 1)
                 if c["experiment"] == "pricing" else 3)
    if 0 < a["comparison_family_size"] < contrasts:
        raise ValueError("comparison_family_size cannot be smaller than this run's comparisons")
    total_notional = sum(p["notional"] for p in c["population"])
    if not math.isfinite(total_notional):
        raise ValueError("Total notional overflows")
    if (a["normalization"] == "per_policy" and not c["population"]
            or a["normalization"] == "per_notional" and total_notional <= 0):
        raise ValueError("This normalization needs positive count/notional")
    for name in ("quantiles", "expected_shortfall_levels"):
        values = a[name]
        if (not isinstance(values, list) or len(set(values)) != len(values)
                or any(not 0 < _numeric(x, name) < 1 for x in values)):
            raise ValueError(f"{name} must contain distinct levels strictly between 0 and 1")
    _integer(a["bins"], "bins", 1, max_count)
    _integer(a["dpi"], "dpi", 1, max_count)
    for name in ("zoom", "plot_range"):
        if a[name] is not None:
            v = a[name]
            if (not isinstance(v, list) or len(v) != 2
                    or _numeric(v[0], name) >= _numeric(v[1], name)):
                raise ValueError(f"{name} must be null or [minimum, maximum]")
    if (not isinstance(a["figure_formats"], list)
            or any(x not in ("png", "pdf", "svg") for x in a["figure_formats"])):
        raise ValueError("Figure formats must be png, pdf or svg")
    if not isinstance(a["title"], str):
        raise ValueError("title must be a string")
    # Also enforce the strict JSON number/type representation used by the backend.
    json.dumps(c, allow_nan=False)


def save_config(config: dict, path: str | Path) -> Path:
    config = _native(config)
    validate_config(config)
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(config, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return p


def _write_rows(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        for row in rows:
            writer.writerow({k: "" if isinstance(v, float) and not math.isfinite(v) else v
                             for k, v in row.items()})


def _mean_statistics(x: np.ndarray, confidence: float) -> dict:
    n = len(x)
    mean = float(np.mean(x))
    variance = float(np.var(x, ddof=1)) if n > 1 else math.nan
    std = math.sqrt(variance)
    se = std / math.sqrt(n)
    z = float(student_t.isf((1 - confidence) / 2, n - 1)) if n > 1 else math.nan
    return {"n": n, "mean": mean, "variance": variance, "std": std,
            "mean_se": se, "mean_ci_low": mean - z * se, "mean_ci_high": mean + z * se}


def expected_shortfall(x: np.ndarray, level: float) -> float:
    ordered = np.sort(x)
    cutoff = level * len(ordered)
    index = int(math.floor(cutoff))
    total = float(np.sum(ordered[index + 1:]))
    total += (index + 1 - cutoff) * float(ordered[index])
    return total / (len(ordered) - cutoff)


def _level_name(prefix: str, level: float) -> str:
    return prefix + format(100 * level, ".10g")


class ExperimentResult:
    def __init__(self, directory: str | Path):
        self.directory = Path(directory).resolve()
        self.metadata = json.loads((self.directory / "metadata.json").read_text())
        self.config = copy.deepcopy(self.metadata["configuration"])
        self.kind = self.config["experiment"]
        self.pricing = None
        self.costs = None
        if self.kind == "pricing":
            self.pricing = json.loads((self.directory / "pricing.json").read_text())
        else:
            with (self.directory / "hedging.csv").open(newline="") as f:
                rows = list(csv.DictReader(f))
            self.costs = np.array([[float(row[k]) for k in STRATEGIES] for row in rows])
            if not np.all(np.isfinite(self.costs)):
                raise ValueError("Hedging CSV contains nonfinite costs")

    def _denominator(self, normalization: str | None) -> float:
        name = normalization or self.config["analysis"]["normalization"]
        if name == "aggregate":
            return 1.0
        if name == "per_policy":
            denominator = float(self.metadata["policy_count"])
        elif name == "per_notional":
            denominator = float(self.metadata["total_notional"])
        else:
            raise ValueError("Unknown normalization")
        if not math.isfinite(denominator) or denominator <= 0:
            raise ValueError("This normalization needs positive count/notional")
        return denominator

    def summary(self, normalization: str | None = None) -> Table:
        """Price/delta estimator statistics or realized hedging-cost statistics."""
        denominator = self._denominator(normalization)
        a = self.config["analysis"]
        if self.kind == "pricing":
            rows = Table()
            base = {r["quantity"]: r["standard_error"]**2
                    for r in self.pricing["summary"] if r["configuration"] == "standard"}
            for r in self.pricing["summary"]:
                row = dict(r)
                row["mean"] /= denominator
                row["sample_variance"] /= denominator**2
                row["standard_error"] /= denominator
                row["standard_deviation"] = math.sqrt(row["sample_variance"])
                row["variance_of_mean"] = row["standard_error"]**2
                row["effective_variance"] = row["actual_paths"] * row["variance_of_mean"]
                z = float(student_t.isf(
                    (1 - a["confidence_level"]) / 2, row["independent_samples"] - 1))
                row["ci_low"] = row["mean"] - z * row["standard_error"]
                row["ci_high"] = row["mean"] + z * row["standard_error"]
                reference = base.get(r["quantity"], 0.0)
                row["variance_reduction_percent"] = (
                    0.0 if r["configuration"] == "standard" else
                    100 * (1 - r["standard_error"]**2 / reference) if reference > 0 else math.nan)
                rows.append(row)
            return rows
        data = self.costs / denominator
        rows = Table()
        for i, name in enumerate(STRATEGIES):
            x = data[:, i]
            row = {"strategy": name, **_mean_statistics(x, a["confidence_level"]),
                   "min": float(x.min()), "max": float(x.max()),
                   "probability_positive": float(np.mean(x > 0))}
            for q in a["quantiles"]:
                row[_level_name("q", q)] = float(np.quantile(x, q, method="linear"))
            for q in a["expected_shortfall_levels"]:
                row[_level_name("es", q)] = expected_shortfall(x, q)
            rows.append(row)
        return rows

    def comparisons(self, normalization: str | None = None) -> Table:
        """Paired strategy cost differences or paired differences across pricing repetitions.

        Pricing comparisons need >=2 independent repetitions to estimate uncertainty.
        Intervals use Student t approximations. Simultaneous intervals use
        Bonferroni over this run's contrasts or the larger declared family.
        """
        denominator = self._denominator(normalization)
        confidence = self.config["analysis"]["confidence_level"]
        rows = Table()
        if self.kind == "hedging":
            data = self.costs / denominator
            for i, j in itertools.combinations(range(3), 2):
                reference, candidate = data[:, i], data[:, j]
                va, vb = np.var(reference, ddof=1), np.var(candidate, ddof=1)
                ratio = float(vb / va) if va > 0 else math.nan
                rows.append({
                    "reference": STRATEGIES[i], "candidate": STRATEGIES[j],
                    **_mean_statistics(candidate - reference, confidence),
                    "variance_ratio": ratio, "variance_reduction_percent": 100 * (1 - ratio),
                    "correlation": float(np.corrcoef(reference, candidate)[0, 1])
                    if va > 0 and vb > 0 else math.nan,
                })
        else:
            for quantity in ("price", "delta"):
                series = {}
                for name in self.config["pricing"]["configurations"]:
                    series[name] = np.array([
                        r["mean"] / denominator for r in self.pricing["repetitions"]
                        if r["quantity"] == quantity and r["configuration"] == name])
                for first, second in itertools.combinations(series, 2):
                    rows.append({"quantity": quantity, "reference": first, "candidate": second,
                        **_mean_statistics(series[second] - series[first], confidence)})
        declared = self.config["analysis"]["comparison_family_size"]
        family = declared or len(rows)
        for row in rows:
            critical = float(student_t.isf((1 - confidence) / (2 * family), row["n"] - 1)) \
                if row["n"] > 1 else math.nan
            row["simultaneous_family_size"] = family
            row["simultaneous_ci_low"] = row["mean"] - critical * row["mean_se"]
            row["simultaneous_ci_high"] = row["mean"] + critical * row["mean_se"]
        return rows

    def repetition_statistics(self, normalization: str | None = None) -> Table:
        """Observed variance of repeated price/delta estimates versus their reported variances."""
        if self.kind != "pricing":
            raise ValueError("repetition_statistics is for pricing experiments")
        denominator = self._denominator(normalization)
        rows = Table()
        for name in self.config["pricing"]["configurations"]:
            for quantity in ("price", "delta"):
                r = [x for x in self.pricing["repetitions"]
                     if x["configuration"] == name and x["quantity"] == quantity]
                means = np.array([x["mean"] / denominator for x in r])
                observed = float(np.var(means, ddof=1)) if len(means) > 1 else math.nan
                predicted = float(np.mean([x["standard_error"]**2 for x in r])) / denominator**2
                rows.append({"configuration": name, "quantity": quantity, "repetitions": len(r),
                    "mean_of_estimates": float(means.mean()),
                    "observed_variance_of_estimates": observed,
                    "mean_reported_variance_of_mean": predicted,
                    "observed_to_reported_ratio": observed / predicted if predicted > 0 else math.nan})
        return rows

    def plot(self, normalization: str | None = None):
        """Return an editable Matplotlib figure, suitable for a notebook cell."""
        import matplotlib.pyplot as plt
        a = self.config["analysis"]
        if self.kind == "pricing":
            rows = self.summary(normalization)
            fig, axes = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True)
            for i, quantity in enumerate(("price", "delta")):
                r = [x for x in rows if x["quantity"] == quantity]
                labels = [x["configuration"].replace("_", "\n") for x in r]
                x = np.arange(len(r))
                axes[0, i].errorbar(x, [v["mean"] for v in r],
                    yerr=[(v["ci_high"] - v["ci_low"]) / 2 for v in r],
                    fmt="o", capsize=4)
                axes[0, i].set(title=quantity.capitalize() + " with confidence intervals",
                               xticks=x, xticklabels=labels)
                variances = [v["effective_variance"] for v in r]
                axes[1, i].bar(x, variances)
                if any(v > 0 for v in variances):
                    axes[1, i].set_yscale("log")
                axes[1, i].set(title="Equal-trajectory variance: " + quantity,
                    ylabel="actual_paths × SE²", xticks=x, xticklabels=labels)
            for ax in axes.flat:
                ax.grid(alpha=.2)
        else:
            series = (self.costs / self._denominator(normalization)).T
            edges = np.histogram_bin_edges(np.concatenate(series), bins=a["bins"],
                                          range=a["plot_range"])
            zoom = a["zoom"]
            if zoom is None:
                low, high = np.quantile(np.concatenate(series[1:]), [.01, .99])
                if low == high:
                    half = max(1.0, abs(float(low))) * .05
                    low, high = low - half, high + half
            else:
                low, high = zoom
            zoom_edges = np.linspace(low, high, a["bins"] + 1)
            fig, axes = plt.subplots(1, 3, figsize=(17, 5.2))
            fig.subplots_adjust(left=.06, right=.98, bottom=.19, top=.84, wspace=.30)
            inset = axes[0].inset_axes([.61, .50, .36, .45])
            for x, label, color in zip(series, LABELS, COLORS):
                for ax, e in ((axes[0], edges), (inset, zoom_edges)):
                    hist, _ = np.histogram(x, bins=e)
                    ax.stairs(hist / (len(x) * np.diff(e)), e, label=label, color=color)
                ordered = np.sort(x)
                axes[1].step(np.r_[ordered[0], ordered],
                    np.r_[0.0, np.arange(1, len(x) + 1) / len(x)],
                    where="post", label=label, color=color)
            axes[0].set(title="Density and central zoom", xlabel="Hedging cost", ylabel="Density")
            axes[0].legend(fontsize=8)
            inset.set(xlim=(low, high), ylim=(0, None), title="Zoom")
            inset.tick_params(labelsize=7)
            axes[1].set(title="Empirical CDF", xlabel="Hedging cost",
                        ylabel="Probability", ylim=(0, 1.02))
            axes[1].legend(fontsize=8)
            axes[2].boxplot(list(series), showfliers=False)
            axes[2].set_xticks([1, 2, 3], LABELS, rotation=20)
            axes[2].set(title="Box plots (outliers hidden)", ylabel="Hedging cost")
            if a["plot_range"] is not None:
                axes[0].set_xlim(a["plot_range"])
                axes[1].set_xlim(a["plot_range"])
            for ax in (*axes, inset):
                ax.grid(alpha=.2)
        name = normalization or a["normalization"]
        fig.suptitle(a["title"] or f"{self.kind.capitalize()} experiment ({name})")
        return fig

    def save_analysis(self, plots: bool = True) -> None:
        _write_rows(self.directory / "summary.csv", self.summary())
        _write_rows(self.directory / "paired_comparisons.csv", self.comparisons())
        if self.kind == "pricing":
            _write_rows(self.directory / "repetition_statistics.csv", self.repetition_statistics())
        a = self.config["analysis"]
        (self.directory / "analysis.json").write_text(json.dumps({
            **a, "confidence_intervals": "Student t approximations; paired Bonferroni intervals also reported",
            "pricing_summary": "first repetition; median CPU time across repetitions",
            "variance_units": "path contributions or antithetic pair averages",
        }, indent=2) + "\n")
        if plots and a["figure_formats"]:
            import matplotlib.pyplot as plt
            fig = self.plot()
            name = "pricing" if self.kind == "pricing" else "distributions"
            for extension in a["figure_formats"]:
                fig.savefig(self.directory / f"{name}.{extension}", dpi=a["dpi"])
            plt.close(fig)


def load_result(directory: str | Path) -> ExperimentResult:
    return ExperimentResult(directory)


def _find_binary(binary: str | Path | None) -> Path:
    if binary is not None:
        path = Path(binary).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Backend executable not found: {path}")
        return path
    roots = (Path.cwd(), Path(__file__).resolve().parents[1])
    for root in roots:
        for build in ("build-paper-release", "build-system-boost", "build"):
            for name in ("carbonfund_experiment", "carbonfund_experiment.exe"):
                candidate = root / build / name
                if candidate.is_file():
                    return candidate.resolve()
    raise FileNotFoundError("Build carbonfund_experiment or supply binary='/absolute/path/to/executable'")


def run_experiment(config: dict, output_dir: str | Path, binary: str | Path | None = None,
                   *, overwrite: bool = False, plots: bool = True,
                   progress: bool = True) -> ExperimentResult:
    """Run one complete experiment in C++; load its numerical results in Python."""
    frozen = _native(config)
    validate_config(frozen)
    executable = _find_binary(binary)
    directory = Path(output_dir).expanduser().resolve()
    if directory.exists() and any(directory.iterdir()) and not overwrite:
        raise FileExistsError("Output directory is not empty; use another directory or overwrite=True")
    directory.mkdir(parents=True, exist_ok=True)
    # Overwrite removes only generated experiment files, preserving unrelated files.
    if overwrite:
        for pattern in ("pricing.json", "pricing.png", "pricing.pdf", "pricing.svg",
                        "hedging.csv", "metadata.json", "summary.csv",
                        "paired_comparisons.csv", "repetition_statistics.csv",
                        "distributions.png", "distributions.pdf", "distributions.svg",
                        "analysis.json", "config.json", "console.log"):
            for path in directory.glob(pattern):
                if path.is_file():
                    path.unlink()
    save_config(frozen, directory / "config.json")
    command = [str(executable), "--config", str(directory / "config.json"),
               "--output", str(directory)]
    with (directory / "console.log").open("w", encoding="utf-8") as log, \
            subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                             text=True, encoding="utf-8") as process:
        try:
            for line in process.stdout:
                log.write(line)
                log.flush()
                if progress:
                    print(line, end="", flush=True)
            code = process.wait()
        except BaseException:
            process.terminate()
            process.wait()
            raise
    if code:
        raise RuntimeError(f"C++ experiment failed (exit {code}); inspect {directory / 'console.log'}")
    result = load_result(directory)
    result.save_analysis(plots)
    return result


def print_table(rows: list[dict]) -> None:
    if not rows:
        print("No comparisons available.")
        return
    fields = (["configuration", "quantity", "mean", "sample_variance", "standard_deviation", "variance_of_mean",
               "standard_error", "variance_reduction_percent", "cpu_seconds"]
              if "configuration" in rows[0] else
              ["strategy", "n", "mean", "variance", "std", "mean_se"])
    def fmt(v: Any, key: str) -> str:
        if key == "variance_reduction_percent":
            return "n/a" if not math.isfinite(v) else "0%" if v == 0 else f"{v:.6f}%"
        return f"{v:.8g}" if isinstance(v, float) else str(v)
    widths = {k: max(len(k), *(len(fmt(r[k], k)) for r in rows)) for k in fields}
    print(" | ".join(k.ljust(widths[k]) for k in fields))
    for row in rows:
        print(" | ".join(fmt(row[k], k).ljust(widths[k]) for k in fields))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--binary", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    result = run_experiment(config, args.output, args.binary,
                           overwrite=args.overwrite, plots=not args.no_plots)
    print_table(result.summary())
    print("Saved results:", result.directory)


if __name__ == "__main__":
    main()

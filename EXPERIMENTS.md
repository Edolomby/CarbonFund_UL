# Experiment interface

[README](README.md) · [Installation](INSTALLATION.md) · [Pricing benchmark](PAPER_VARIANCE_REDUCTION.md)

The Python module `carbonfund_experiments` validates a configuration, launches the `carbonfund_experiment` C++ executable and analyses its saved results. No simulation loop is implemented in Python.

## Run an experiment

After building the backend and installing the Python package as described in [INSTALLATION.md](INSTALLATION.md):

```bash
./.venv-analysis/bin/python -m carbonfund_experiments \
    --config experiments/demo_pricing.json \
    --output results/interface_pricing \
    --binary build-paper-release/carbonfund_experiment
```

Use `experiments/demo_hedging.json` and a different output directory for a hedging run. The interface creates the directory and saves the complete configuration. A nonempty destination requires explicit `--overwrite`; `--no-plots` skips figure generation.

The executable is found automatically in `build-paper-release/`, `build-system-boost/` or `build/` when `--binary` is omitted. An explicit path is preferable when keeping several builds.

## Configuration structure

Configurations use `schema_version: 1` and `experiment: "pricing"` or `"hedging"`. Both require `market`, `carbon`, `fund`, `contract`, `mortality`, `population` and `analysis`. The selected experiment additionally requires its `pricing` or `hedging` section. `notes` is optional descriptive text.

The notebooks define these sections explicitly in separate editable cells. They do not require a preset or a JSON file to choose model inputs.

### Market

| Input | Meaning |
|---|---|
| `risk_free_rate` | Constant continuously compounded cash rate. |
| `expected_returns` | Four physical stock drifts used in fund allocation. |
| `volatilities` | Four stock return volatilities. |
| `correlations` | Symmetric positive-definite 4×4 return correlation matrix with unit diagonal. |

The covariance matrix is constructed from volatilities and correlations. Financial parameters are constant. Pricing uses the risk-free financial drift under the minimal martingale measure; the expected returns still enter the allocation rule.

### Carbon

`initial`, `mean_reversion` and `volatility` are four-element vectors. The model determines the mean input:

| `model` | Mean input | Interpretation |
|---|---|---|
| `OU` | `long_run_mean` | Mean-reverting Gaussian intensity; exact transitions. |
| `GeoOU` | `long_run_log_mean` | Mean-reverting log intensity; exact transitions in log space. |
| `CIR` | `long_run_mean` | Square-root intensity; Ninomiya-Victoir splitting. |

GeoOU initial values are supplied as positive intensities, rather than logarithms. CIR initial states and means must be nonnegative, and each factor must satisfy `mean_reversion[i] * long_run_mean[i] >= volatility[i]^2 / 4`. OU can take negative values; allocation uses the positive part of carbon intensity. Carbon factors and financial noise are independent in the current model.

### Fund and contract

`fund.risk_aversion` is positive and `fund.carbon_penalty` is nonnegative. They control the carbon-sensitive allocation rule.

| Contract input | Meaning |
|---|---|
| `fund_value` | Current fund value at the start of the experiment. |
| `maturity` | Remaining term in years. |
| `minimum_guarantee` | Initial lower contractual bound. |
| `maximum_guarantee` | Initial upper contractual bound. |
| `minimum_guarantee_growth_rate` | Continuous growth rate of the lower bound. |
| `maximum_guarantee_growth_rate` | Continuous growth rate of the upper bound. |
| `endowment_death_benefit_fraction` | Nonnegative EI death-benefit multiplier. |

At elapsed time `t`, the per-notional benefit is the fund value bounded below by `minimum_guarantee * exp(minimum_guarantee_growth_rate * t)` and above by the corresponding growing maximum. Future bounds must remain ordered and distinct. Bounds can coincide initially and separate immediately through their different growth rates. Fund delta holds the contractual bounds fixed.

PE pays on survival to maturity; TI pays on death before maturity. EI combines the PE benefit with the TI benefit multiplied by the contractual fraction, on the same insured life. Benefits are ordinary nominal amounts and are discounted once.

### Mortality and population

The Gompertz-Makeham hazard for initial age `a` is

$$
\mu(a+t)=\xi+\frac{1}{b}\exp\left(\frac{a+t-m}{b}\right).
$$

`mortality` contains `xi >= 0`, finite `m` and `b > 0`. Paper-calibrated experiments use `xi = 0.0041959`, `m = 79.6921211` and `b = 11.5818911`.

`population` is a list of individual policies. Each entry supplies `type` (`PE`, `TI` or `EI`), `age` and `notional`. Ages and notionals are nonnegative. Individual lifetimes are independent, while market paths are shared by all lives in a scenario. For valuation, living policies are aggregated into age cohorts. All policies in one experiment share the contract terms.

Pricing accepts one contract type. Hedging accepts mixed populations. At fixed total exposure, a homogeneous population of `n` lives with total notional `Q` gives each life notional `Q / n`.

## Pricing settings

| Input | Meaning |
|---|---|
| `contract` | `PE`, `TI` or `EI`; agrees with every population entry. |
| `paths` | Trajectory budget per configuration and repetition. |
| `intervals` | Time intervals over the pricing horizon. |
| `configurations` | Distinct selections from the four configurations below. |
| `repetitions` | Independently seeded complete evaluations. |
| `carbon_seed`, `financial_seed` | Distinct unsigned 64-bit random-stream seeds. |

The four configurations are `standard`, `standard_antithetic`, `conditioned` and `conditioned_antithetic`. All integrate deterministic mortality analytically. Conditioned estimators additionally integrate the financial Gaussian analytically conditional on the carbon path.

Standard TI/EI pricing advances one coherent financial trajectory through the grid. Standalone PE needs only a terminal financial draw after accumulating the carbon-dependent integrated variance. These rules follow from the selected contract; there is no selectable fund-simulation mode.

Antithetic paths use Gaussian inputs of opposite sign in each model transition. Standard antithetics also negate financial inputs. Statistics use one observation per pair. If any selected configuration is antithetic, odd requests round up for all configurations to retain equal actual trajectory budgets.

### Pricing statistics

| Field | Meaning |
|---|---|
| `mean` | Price or fund delta. |
| `sample_variance` | Unbiased variance of path contributions or pair averages. |
| `standard_deviation` | Square root of that sample variance. |
| `variance_of_mean` | `sample_variance / independent_samples`. |
| `standard_error` | Square root of the variance of the mean. |
| `effective_variance` | `actual_paths * variance_of_mean`. |
| `variance_reduction_percent` | `100 * (1 - variance_of_mean / standard_variance_of_mean)`. |
| `ci_low`, `ci_high` | Approximate Student t interval for the mean. |
| `cpu_seconds` | Median CPU time of a complete price-and-delta evaluation. |

The standard estimator reports `0%`. Negative reductions are retained; without a positive standard reference, nonbaseline reductions are undefined. Raw path variances and raw pair-average variances have different statistical units and must not be compared as equal-budget estimator variances.

`summary()` reports estimates from the first repetition and median timing across repetitions. Repetitions are not pooled into a larger estimate. `repetition_statistics()` compares observed dispersion across repetitions with reported estimator uncertainty. `comparisons()` uses paired differences across repetitions and retains cross-method covariance; at least two repetitions are needed to estimate their uncertainty. Three repetitions are a workflow demonstration, not a precise variance calibration.

## Hedging settings

| Outer input | Meaning |
|---|---|
| `outer_scenarios` | Independent realized market/mortality scenarios. |
| `intervals` | Uniform outer grid for market simulation, death settlement and hedge dates. |
| `progress_every` | Print progress after this many completed scenarios; `0` disables messages. |
| `threads` | Requested outer workers; `0` uses the OpenMP runtime limit. |
| `carbon_seed`, `financial_seed`, `mortality_seed` | Separate outer random-stream seeds. |
| `initial` | Initial valuation settings. |
| `revaluation` | Settings for subsequent portfolio valuations. |

Both valuation sections specify `paths`, `intervals`, `configuration`, `carbon_seed` and `financial_seed`. The configuration is one of the four pricing estimators. Revaluation additionally accepts `minimum_intervals`, defaulting to `1`, with `1 <= minimum_intervals <= intervals`.

For original maturity `T` and valuation time `t`, the residual pricing grid uses

$$
N(t)=\max\left(N_{\min},\left\lceil N_{\mathrm{base}}\frac{T-t}{T}\right\rceil\right).
$$

Setting `minimum_intervals` equal to `intervals` fixes the residual interval count. This setting is separate from the outer hedge grid. The initial valuation uses its own specified interval count at time zero.

All seven outer/initial/revaluation seed inputs must be distinct, including financial seeds unused by conditioned pricing. Initial premium and aggregate deltas are computed once. Each subsequent valuation uses the current fund and carbon states, residual maturity, evolved guarantees and living policies. Prices and deltas already include notionals.

### Strategies and accounting

All strategies share the same premium, market observations and deaths:

- `no_hedge`: retain the premium in cash.
- `static_hedge`: scale initial aggregate deltas by surviving-notional ratios within each contract type, without market repricing.
- `dynamic_hedge`: recompute aggregate delta on each outer hedge date.

Costs are discounted to time zero: discounted claims minus initial premium minus discounted trading gains. Positive costs require additional funding; negative costs are surpluses. Deaths are settled at interval right endpoints. Outer scenarios use the physical measure; valuation uses the minimal martingale measure.

### Parallel execution and repeatability

OpenMP parallelizes complete outer scenarios. Initial valuation, within-scenario time evolution and inner Monte Carlo sampling remain sequential. Workers own their mutable states and buffers. Random streams belong to scenarios rather than workers, and CSV rows remain in scenario order.

`threads=1` selects one worker; a positive value requests that many workers, capped by the scenario count and runtime limits. A build without OpenMP accepts only `0` or `1`. Inspect `run.openmp_enabled`, `run.threads_requested` and `run.threads_used` in `metadata.json`.

For the same executable and inputs, changing thread count preserves the numerical CSV. Changing inner budgets preserves realized market and mortality scenarios. Scenario seeds are derived from the supplied stream seeds using the versioned schedule recorded in metadata. Reusing market seeds, financial inputs, carbon inputs and outer grids across portfolios supports paired comparisons. Different populations need not have identical lifetimes or premiums.

Progress counts completed scenarios, regardless of completion order. It does not alter simulation settings or checkpoint unfinished scenarios. A completed experiment exports its CSV; a failed scenario aborts the run.

Use `run.simulation_wall_seconds` for parallel elapsed-time comparisons. `run.simulation_cpu_seconds` sums process CPU work across threads and can exceed wall time. Initial valuation has separate wall/CPU timings. Top-level `wall_seconds` includes C++ export and excludes Python analysis and plotting.

## Analysis settings

| Input | Meaning |
|---|---|
| `normalization` | `aggregate`, `per_policy` or `per_notional`. |
| `confidence_level` | Mean interval level strictly between zero and one. |
| `comparison_family_size` | Bonferroni family size; `0` uses this run's contrast count. |
| `quantiles` | Distinct requested probability levels between zero and one. |
| `expected_shortfall_levels` | Upper-tail expected-shortfall levels. |
| `bins` | Histogram bin count. |
| `zoom` | Optional `[minimum, maximum]` inset range, or `None`/JSON `null`. |
| `plot_range` | Optional plot range, or `None`/JSON `null`. |
| `figure_formats` | Selected formats from `png`, `pdf`, `svg`; an empty list saves no figures. |
| `dpi` | Raster figure resolution. |
| `title` | Figure title; the Python plotter supplies a default when empty. |

Normalization changes analysis units, rather than simulations: means and standard errors divide by exposure; variances divide by exposure squared. Raw `hedging.csv` values remain aggregate amounts. Hedging summaries include means, variances, standard deviations, mean standard errors and intervals, minima/maxima, positive-cost probabilities, quantiles and expected shortfall. Paired strategy comparisons use common scenario rows.

Mean confidence intervals quantify sampling uncertainty conditional on the initial premium used. They do not include its separate estimation uncertainty or time-discretization error. A smaller Monte Carlo standard error does not establish convergence of hedge dates, death settlement or inner time integration.

## Saved files and Python API

Every run writes `config.json`, `metadata.json`, `console.log`, `summary.csv`, `paired_comparisons.csv` and `analysis.json`. Pricing additionally writes `pricing.json` and `repetition_statistics.csv`; hedging writes `hedging.csv`. Figure names are `pricing` or `distributions`, with the selected extensions.

```python
from carbonfund_experiments import run_experiment, load_result

# config is a complete dictionary assembled from explicit input cells.
result = run_experiment(
    config, "results/custom_run",
    binary="build-paper-release/carbonfund_experiment",
    overwrite=False, plots=True, progress=True,
)
```

To analyse saved data without invoking C++:

```python
result = load_result("results/custom_run")
statistics = result.summary()
comparisons = result.comparisons()
result.config["analysis"]["normalization"] = "per_notional"
result.save_analysis(plots=True)
```

`result.plot()` returns an editable Matplotlib figure. In notebooks, result tables display as HTML without pandas. The notebooks also save combined comparison tables in their experiment-specific parent directories. Keep those directories and the individual run files together for subsequent MATLAB or R analysis.

## Verification

Run the integration checks against a selected backend:

```bash
./.venv-analysis/bin/python tests/test_experiment_interface.py \
    --binary build-paper-release/carbonfund_experiment
```

Checks cover configuration validation, variance/standard-error identities, normalization, paired sampling, random-stream repeatability, discounted accounting and serial/parallel agreement. Statistical agreement of independently sampled estimates is assessed separately from deterministic regression checks.

# Python experiment interface

Python configures experiments, launches the existing C++ simulation routines and
analyses their results. All simulation loops remain in C++. The module can be
imported from a script or a future Jupyter notebook.

## Save the files

Paths below are relative to `~/Documents/codes/CarbonFund_UL`.

| File | Save as |
|---|---|
| `run_experiment.cpp` | `examples/run_experiment.cpp` |
| `carbonfund_experiments.py` | `python/carbonfund_experiments.py` |
| `pyproject.toml` | `pyproject.toml` |
| `demo_pricing.json` | `experiments/demo_pricing.json` |
| `demo_hedging.json` | `experiments/demo_hedging.json` |
| `test_experiment_interface.py` | `tests/test_experiment_interface.py` |
| `EXPERIMENTS.md` | `EXPERIMENTS.md` |

```bash
cd ~/Documents/codes/CarbonFund_UL &&
mkdir -p python experiments
```

Add these two lines to `CMakeLists.txt` beside the other example targets, outside
`if(BUILD_TESTING)`. Add them only once:

```cmake
add_executable(carbonfund_experiment examples/run_experiment.cpp)
target_link_libraries(carbonfund_experiment PRIVATE CarbonFund::carbonfund)
```

Boost.JSON's implementation is compiled in this single source file using the
installed Boost headers. No separately linked Boost.JSON library is required.
Keep existing benchmark source paths under `benchmarks/`.

## Build and install

```bash
cmake -S . -B build-paper-release \
    -DBoost_ROOT=/usr \
    -DCMAKE_BUILD_TYPE=Release \
    -DCARBONFUND_ENABLE_OPENMP=ON \
    -DBUILD_TESTING=ON &&
cmake --build build-paper-release --parallel &&
ctest --test-dir build-paper-release --output-on-failure --no-tests=error
```

Install the Python interface in the analysis environment:

```bash
python3 -m venv .venv-analysis &&
./.venv-analysis/bin/python -m pip install -e .
```

This installs NumPy, SciPy and Matplotlib as needed. SciPy supplies Student t
quantiles for confidence intervals. No C++ Python extension is needed.
An editable installation exposes subsequent Python source edits immediately.

Ignore generated package metadata when committing source files:

```bash
grep -qxF '*.egg-info/' .gitignore || printf '%s\n' '*.egg-info/' >> .gitignore
```

## First experiment: one insurance contract

The small pricing configuration contains one PE policy, age 60, notional 1.
It compares the four estimators with 2,000 actual trajectories and three
independent repetitions.

```bash
./.venv-analysis/bin/python -m carbonfund_experiments \
    --config experiments/demo_pricing.json \
    --output results/interface_pricing
```

The interface finds `build-paper-release/carbonfund_experiment` automatically.
To select another executable, add `--binary /absolute/path/to/carbonfund_experiment`.
Choose a new output directory for another run. To replace an existing
experiment's generated files, explicitly pass `--overwrite`.

Outputs include:

| Output | Contents |
|---|---|
| `config.json` | Complete submitted configuration. |
| `metadata.json` | Configuration, backend versions, actual counts and run details. |
| `console.log` | Backend progress and errors. |
| `pricing.json` | Price/delta moments and actual seeds from every repetition. |
| `summary.csv` | First-repetition estimates and median CPU times. |
| `paired_comparisons.csv` | Paired differences between methods across repetitions. |
| `repetition_statistics.csv` | Observed versus reported variance of estimates. |
| `pricing.png/pdf` | Estimates with intervals and equal-budget variances. |
| `analysis.json` | Statistical and plotting conventions. |

Repetitions are not pooled into a larger price estimate. They provide CPU
timing repetitions and independent estimates for calibration. Three repetitions
are only a small demonstration; use substantially more for useful calibration.
The implementation rotates method order across repetitions. Seeds are checked
for collisions across all repetitions and both random-stream roles; actual
values are saved.

## Use from Python or a notebook

```python
import numpy as np
from carbonfund_experiments import demo_config, run_experiment, print_table

config = demo_config("pricing", contract="EI")
config["population"] = [{"type": "EI", "age": 60.0, "notional": 1.0}]
config["contract"]["fund_value"] = 1.2
config["contract"]["maturity"] = 10.0
config["market"]["volatilities"] = np.array([0.20, 0.25, 0.18, 0.22])
config["market"]["correlations"] = np.eye(4)
config["pricing"].update(paths=100_000, intervals=100, repetitions=20)
# TI/EI use sequential fund paths; PE-only uses a terminal financial draw.

result = run_experiment(config, "results/single_EI")
print_table(result.summary())
```

In notebook cells, the returned tables have an HTML representation:

```python
result.summary()
result.comparisons()
result.repetition_statistics()
result.plot()
```

Each statement can occupy its own cell. `plot()` returns an editable Matplotlib
figure. In a regular Python script, use `matplotlib.pyplot.show()` to open it.
To reopen completed results without rerunning C++:

```python
from carbonfund_experiments import load_result
result = load_result("results/single_EI")
result.summary()
```

## Fund simulation

Standard TI/EI pricing always advances the financial factor sequentially over
the pricing grid. Standalone PE draws one financial normal at maturity after
accumulating the carbon-dependent integrated variance. Conditioned estimators
draw no financial normals. There is no configurable fund-simulation mode.

Configurations use schema version 1. The retired `fund_sampling` input is
rejected; remove it from pricing and from both hedging valuation sections.
Previously saved results remain readable with `load_result()`.

## Pricing variances

Each price/delta row contains:

| Field | Meaning |
|---|---|
| `mean` | Discounted insurance price or fund delta. |
| `sample_variance` | Variance of independent path contributions or antithetic pair averages. |
| `standard_deviation` | Square root of `sample_variance`. |
| `variance_of_mean` | `sample_variance / independent_samples`, equal to SE squared. |
| `standard_error` | Sampling uncertainty of the mean. |
| `effective_variance` | `actual_paths * variance_of_mean` for equal-trajectory comparisons. |
| `variance_reduction_percent` | Reduction in variance of the mean relative to `standard`. |
| `ci_low/high` | Approximate Student t interval at the selected confidence level. |
| `cpu_seconds` | Median CPU time of the complete price-and-delta evaluation. |

If any selected method is antithetic, odd requests round up for every selected
method to keep actual trajectory budgets equal. With M actual trajectories,
independent sampling has n=M observations and antithetic sampling has n=M/2.

The standard baseline displays 0%. Negative reductions are retained. Without
the standard estimator, or with a zero reference variance, nonbaseline
reductions are undefined and left blank.

These are estimator variances for the discretized model, with mortality
integrated analytically. They are not realized insurance-claim variances.
Increasing the path count controls sampling error; time-grid bias is separate.

`comparisons()` uses paired differences across independent repetitions,
retaining cross-method covariance. It reports individual intervals and
Bonferroni simultaneous intervals. The default simultaneous family covers the
current run: 12 contrasts for four methods and two quantities. Set
`analysis.comparison_family_size` to a larger declared family when comparing
several maturities or contracts. One repetition cannot estimate the uncertainty
of repeated-run differences, so those fields are left blank. Student t intervals
are approximations for nonnormal Monte Carlo observations.

## Hedging

```bash
./.venv-analysis/bin/python -m carbonfund_experiments \
    --config experiments/demo_hedging.json \
    --output results/interface_hedging
```

This small configuration uses one PE policy, 20 outer scenarios, 512 initial
pricing trajectories and 128 later pricing trajectories. These are editable.

To construct a mixed population:

```python
from carbonfund_experiments import demo_config, run_experiment

config = demo_config("hedging")
config["population"] = [
    {"type": kind, "age": 55.0 + i % 11, "notional": 1000.0 + 10.0 * i}
    for kind in ("PE", "TI", "EI")
    for i in range(20)
]
config["hedging"].update(outer_scenarios=1000, intervals=40, threads=8)
config["hedging"]["initial"].update(paths=8192, intervals=100)
config["hedging"]["revaluation"].update(paths=512, intervals=100)
config["analysis"]["normalization"] = "per_notional"

result = run_experiment(config, "results/mixed_portfolio")
result.summary()
result.comparisons()
result.plot()
```

The output `hedging.csv` uses the existing scenario-level schema, so the
previous `tools/analyze_hedging.py` also accepts it. The interface saves statistics
and `distributions.png/pdf` automatically.

All strategies share the computed initial premium, market observations and
deaths. Premium and deltas are always computed; manual overrides and direct
covariance inputs are not accepted. Volatilities and correlations determine
the covariance once at setup.

Outer scenarios follow the physical measure; valuations use the minimal
martingale measure. Initial and subsequent valuations may select different
estimators. All seven seed inputs are explicit and must be distinct, including
financial streams unused in conditioned mode. Changing inner path budgets
does not change realized markets or deaths.

### Parallel outer scenarios

The C++ runner parallelizes complete outer hedging scenarios with OpenMP.
Time evolution and inner Monte Carlo sampling within each scenario remain
sequential. Each worker owns its population and market-path buffers. Random
streams belong to scenarios rather than workers, and every result is written
to its scenario-indexed CSV row.

Set `hedging.threads` explicitly in the notebook or JSON:

| Value | Behaviour |
| --- | --- |
| `1` | Execute the outer scenarios on one thread. |
| `2`, `4`, `8`, ... | Request this many outer workers. |
| `0` | Use the OpenMP runtime's default maximum team size. |

The requested team is capped by the number of outer scenarios. OpenMP runtime
limits can reduce the actual team further. The demo configurations explicitly
select `threads=0`. Older configurations without this field default to one
thread. Build with `-DCARBONFUND_ENABLE_OPENMP=ON` to use multiple workers.
A build without OpenMP accepts `0` or `1` and uses one worker; an explicit
request greater than one fails with a rebuild instruction.

Initial valuation is performed once before the parallel region and shared by
all workers. Workers do not share mutable RNGs, policyholder states, path
storage or pricing workspaces. Progress messages count completed scenarios;
completion order can differ from scenario order. CSV rows remain ordered by
scenario ID. Worker exceptions are caught and rethrown after the parallel
region; a failed run does not export a partial hedging CSV.

#### Random streams and repeatability

The initial valuation retains its supplied carbon and financial seeds.
Each outer scenario receives five derived seeds: outer carbon, outer financial,
mortality, revaluation carbon and revaluation financial. They are generated
serially in scenario/role order using the SplitMix64 mixing function. Any seed
collision is resolved by incrementing, excluding the initial valuation seeds.
The versioned schedule is recorded in `metadata.json`.

For the same inputs and executable, changing only `threads` preserves every
numerical CSV value, including dynamic costs. Extending the scenario count
preserves the existing prefix. Changing inner path budgets does not change
outer markets or mortality draws.

This scenario-based schedule replaces the previous continuous cross-scenario
streams, including in one-thread runs. New hedging samples therefore differ
from older saved CSVs. Initial pricing, payoff formulas and hedge accounting
are unchanged. The PE-only optimization is not part of this change.

#### Timing and diagnostics

`metadata.json` contains these fields inside `run`:

| Field | Meaning |
| --- | --- |
| `threads_requested` | The supplied value, or `1` when omitted. |
| `threads_used` | Actual outer team size. |
| `openmp_enabled` | Whether the runner was compiled with OpenMP. |
| `initial_wall_seconds` | Elapsed time for the initial valuation. |
| `simulation_wall_seconds` | Elapsed time for outer setup and scenario execution, before CSV export. |
| `initial_cpu_seconds` | Process CPU time for initial valuation. |
| `simulation_cpu_seconds` | Process CPU time for outer setup and scenario execution. |
| `scenario_seed_schedule` | Versioned random-stream schedule. |

Compare `simulation_wall_seconds` when measuring parallel speedup. CPU time
sums work across process threads and need not fall when elapsed time falls.
The top-level `wall_seconds` measures the C++ experiment, including export;
it excludes Python analysis and plotting.

For example, after a notebook run:

```python
run = result.metadata["run"]
print("Threads:", run["threads_used"])
print("Elapsed seconds:", run["simulation_wall_seconds"])
print("CPU seconds:", run["simulation_cpu_seconds"])
```

Initial and subsequent pricing interval counts refer to the original maturity.
Later grids shrink with remaining maturity to retain approximately that step
size. The outer grid is uniform with the chosen number of hedging intervals.
This interface generates scenarios; external path injection remains available
through the existing C++ API.

Costs refer to time zero; positive means additional funding. The static hedge
scales initial aggregate deltas by surviving-notional ratios within each type.
Death benefits are settled at interval endpoints. Notionals are already
included in prices and deltas.

## Paper calibration

```python
from carbonfund_experiments import paper_config, run_experiment

config = paper_config("pricing", contract="PE", policyholders=1, maturity=20.0)
# Inspect all inputs before running; this preset starts at one million paths.
config["pricing"].update(paths=100_000, repetitions=20)
result = run_experiment(config, "results/paper_PE_T20")
```

The preset uses the financial/CIR calibration in the existing paper benchmark,
xi=0.0041959 and Gompertz exponent (age+t-m)/b.

For diversification, run PE/TI/EI separately with 1 and 1,000 policyholders and
per-policy normalization. For strategy comparisons, use homogeneous populations.
For the age study, supply the individual ages explicitly.

The preset is a starting point, not a claim to reproduce every published
number. Guarantees follow the existing variance benchmark's growing-guarantee
convention. The hedging grid and inner budgets are explicit choices, not
recovered manuscript inputs. Corrected discounted accounting and static
positions must also be considered when comparing with legacy outputs.

## Configuration scope

Only stock/carbon dimensions are fixed at four. Inputs cover financial
parameters, carbon dynamics, preferences, contracts, individual populations,
mortality parameters, simulation budgets, grids, estimators, seeds and analysis.

OU and CIR use `long_run_mean`; GeoOU uses `long_run_log_mean` with positive
initial carbon supplied in level space. CIR requires nonnegative states/means
and kappa*mean >= volatility^2/4.

Pricing selects one contract type. Hedging also accepts mixed PE/TI/EI books.
All policies in one experiment share the contract terms; different maturities
and term sets are separate experiments. The implemented mortality family and
measure assumptions are retained.

Analysis inputs include normalization, confidence level, comparison family,
quantiles, expected-shortfall levels, histogram bins/ranges, figure formats,
resolution and title. Edit `result.config["analysis"]` and call
`result.save_analysis()` to reanalyse existing results. Normalization divides
means by exposure and variances by exposure squared; it does not rerun C++.

## Verification

```bash
./.venv-analysis/bin/python tests/test_experiment_interface.py \
    --binary build-paper-release/carbonfund_experiment
```

Checks cover three carbon models, odd-budget pairing, repeatability, variance/SE
identities, normalization, paired repetitions, unique repetition seeds,
discounted hedge accounting, notional scaling, budget-independent scenarios,
standard-estimator hedging, invalid inputs, plots and deterministic cases.
They also check identical serial/parallel CSVs for all three carbon models and
all four estimators, thread controls, scenario-prefix stability, progress
counts, worker error handling and timing metadata.

To compare both build variants, additionally pass an executable compiled with
OpenMP disabled:

```bash
python tests/test_experiment_interface.py \
    --binary build-paper-release/carbonfund_experiment \
    --serial-binary build-no-openmp/carbonfund_experiment
```

When existing `example_petiei_hedging` and
`benchmark_petiei_variance_reduction` binaries are beside the runner, the checks
also compare initial hedging premiums with the original example and pricing moments
with all 96 rows of the existing paper benchmark.

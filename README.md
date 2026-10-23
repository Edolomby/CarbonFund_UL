# CarbonFund-UL

**Pricing and risk-minimizing hedging of carbon-sensitive unit-linked insurance portfolios.**

CarbonFund_UL is a C++20 research library implementing the numerical framework developed in:

> Katia Colaneri, Alessandra Cretarola, Edoardo Lombardo and Daniele Mancinelli (2026), [*Carbon-Sensitive Fund Construction and Hedging for Green Unit-Linked Life Insurance*](https://arxiv.org/abs/2509.05676).

The fund adjusts its investment allocation in response to firms' carbon intensities. These investment choices affect both the insurance benefits and the insurer's hedge. The library connects fund construction, insurance valuation and portfolio simulation to study how effectively financial trading controls risk when carbon intensity and mortality introduce additional, non-traded sources of uncertainty.

The C++ engine computes prices, fund deltas and realized hedging costs. A Python interface and Jupyter notebooks configure experiments, analyse results and export data for further processing.

## Capabilities

- **Carbon-sensitive fund construction:** four risky assets and a risk-free account, with user-specified returns, volatilities, correlations, risk aversion and carbon penalty.
- **Carbon dynamics:** Ornstein-Uhlenbeck (OU), geometric OU and Cox-Ingersoll-Ross (CIR) models. OU transitions are exact; geometric OU transitions are exact in log intensity; CIR uses a Ninomiya-Victoir splitting scheme.
- **Insurance portfolios:** individual policyholders or valuation cohorts, with configurable ages and notionals and Gompertz-Makeham mortality.
- **Pricing and delta estimation:** standard Monte Carlo, conditional Monte Carlo and antithetic sampling, with estimator variances, standard errors and timing measurements.
- **Hedging experiments:** no hedge, mortality-adjusted static hedge and dynamic delta hedge, evaluated on common realized market paths and deaths.

The supported contracts are:

| Contract | Payment |
|---|---|
| Pure endowment (PE) | A survival benefit at maturity. |
| Term insurance (TI) | A death benefit before maturity. |
| Endowment insurance (EI) | A survival benefit and a death benefit, with a configurable death-benefit fraction. |

Benefits depend on the fund value, subject to a lower guarantee and an upper cap. The guarantee and cap can grow at different contractual rates. Portfolio prices and deltas already include policy notionals, mortality weights and discounting.

## Quick start

The C++ library requires a C++20 compiler, CMake 3.22 or newer, Eigen 3.4 or newer and Boost 1.88 or newer. The Python interface requires Python 3.10 or newer. See [INSTALLATION.md](INSTALLATION.md) for dependency installation and environment setup.

Run the following commands from the repository root. This Release build enables OpenMP for outer hedging scenarios:

```bash
cmake -S . -B build-paper-release \
    -DCMAKE_BUILD_TYPE=Release \
    -DCARBONFUND_ENABLE_OPENMP=ON \
    -DBUILD_TESTING=ON &&
cmake --build build-paper-release --parallel &&
ctest --test-dir build-paper-release --output-on-failure --no-tests=error
```

Create the Python environment and install the experiment interface:

```bash
python3 -m venv .venv-analysis &&
./.venv-analysis/bin/python -m pip install -e .
```

### Compare the four pricing estimators

```bash
./.venv-analysis/bin/python -m carbonfund_experiments \
    --config experiments/demo_pricing.json \
    --output results/quickstart_pricing \
    --binary build-paper-release/carbonfund_experiment
```

This small example reports price and delta estimates, variances, standard errors, variance reduction percentages and CPU times for:

| Configuration | Financial randomness | Sampling |
|---|---|---|
| `standard` | Simulated | Independent trajectories |
| `standard_antithetic` | Simulated | Antithetic trajectory pairs |
| `conditioned` | Integrated analytically conditional on carbon | Independent carbon trajectories |
| `conditioned_antithetic` | Integrated analytically conditional on carbon | Antithetic carbon pairs |

The selected contract, model parameters, grids, path counts, repetitions and random seeds are specified in the configuration file.

### Simulate hedging costs

```bash
./.venv-analysis/bin/python -m carbonfund_experiments \
    --config experiments/demo_hedging.json \
    --output results/quickstart_hedging \
    --binary build-paper-release/carbonfund_experiment
```

The engine computes an initial premium and delta, then simulates the three hedging strategies on each outer scenario. Initial valuation, subsequent revaluation and outer simulation have separate numerical budgets and random seeds.

With OpenMP enabled, `hedging.threads` controls parallel execution of independent outer scenarios; `0` uses the OpenMP runtime limit, subject to the number of scenarios. Time steps within a scenario and inner pricing evaluations remain sequential. Requested and actual thread counts are recorded in `metadata.json`.

The demo configurations use small illustrative budgets. They check the workflow; the research notebooks specify the inputs and simulation budgets for their experiments. Each command creates its output directory. Use a new directory for another run, or pass `--overwrite` to replace that run's generated files.

## Results and subsequent analysis

The Python interface saves numerical results and the inputs used to generate them:

| File | Contents |
|---|---|
| `config.json` | Experiment inputs. |
| `metadata.json` | Configuration, run settings, timing and build information. |
| `summary.csv` | Price/delta statistics or hedging-cost statistics. |
| `paired_comparisons.csv` | Paired comparisons between estimators or hedging strategies. |
| `pricing.json` | Pricing estimates and Monte Carlo statistics for each repetition; pricing runs only. |
| `repetition_statistics.csv` | Statistics across repeated pricing runs; pricing runs only. |
| `hedging.csv` | One row per outer scenario, including strategy costs, discounted claims and trading gains; hedging runs only. |
| `analysis.json`, `console.log` | Analysis settings and execution log. |

Plots are saved in the formats selected in `analysis.figure_formats`. The demo configurations request PNG and PDF. CSV files can be read directly by MATLAB, R or other analysis tools.

Saved runs can also be loaded in Python without repeating the simulations:

```python
from carbonfund_experiments import load_result

result = load_result("results/quickstart_hedging")
statistics = result.summary()
```

See [EXPERIMENTS.md](EXPERIMENTS.md) for the configuration schema, Python API and analysis options, and [PAPER_VARIANCE_REDUCTION.md](PAPER_VARIANCE_REDUCTION.md) for the pricing benchmark and statistical agreement checks.

## Notebooks

The notebooks expose model, contract, population, simulation and analysis settings in editable cells. Select the Python environment containing the installed `carbonfund-experiments` package as the notebook kernel.

| Notebook | Experiment |
|---|---|
| [Pricing_example.ipynb](notebook/Pricing_example.ipynb) | Prices and deltas with the four estimators, including variance, standard error and runtime. |
| [Hedging_example.ipynb](notebook/Hedging_example.ipynb) | A configurable insurance portfolio and the three hedging strategies. |
| [Policyholder_pooling_1_vs_1000.ipynb](notebook/Policyholder_pooling_1_vs_1000.ipynb) | The experiment underlying Table 6.2, comparing PE, TI and EI portfolios. |
| [Policyholder_pooling_1_vs_1000.ipynb](notebook/Policyholder_pooling_1_vs_1000.ipynb) | One versus 1,000 policyholders at equal total notional, following Figure 6.1. |
| [Homogeneous_vs_heterogeneous.ipynb](notebook/Homogeneous_vs_heterogeneous.ipynb) | Homogeneous and heterogeneous age groups, following Figure 6.3. |
| [OU and GeoOU hedging experiments](notebook/OU_GeoOU_hedging.ipynb): Calibrate OU and GeoOU carbon models to match the CIR model’s cumulative carbon mean and variance. Compare no hedging, static hedging and dynamic hedging for PE, TI and EI portfolios, with an optional CIR baseline. All inputs are editable; the notebook exports scenario-level CSV data, statistical summaries, calibration diagnostics and figures.

The main paper develops the fund construction and hedging framework, with numerical hedging experiments in Section 6. Section E of the supplementary material describes fund simulation and the conditional pricing method. Reproduction requires matching both the model inputs and the numerical settings; the quick-start demos use their own illustrative parameters.


## C++ use

The numerical library is header-only. C++ applications link to the CMake interface target `CarbonFund::carbonfund`, which supplies include paths, the C++20 requirement and dependencies.

See [examples/petiei_monte_carlo.cpp](examples/petiei_monte_carlo.cpp) for direct pricing and delta calculations, [examples/petiei_hedging.cpp](examples/petiei_hedging.cpp) for scenario-level hedging, and [examples/run_experiment.cpp](examples/run_experiment.cpp) for the configurable backend used by Python.

## Citation

When using the framework or numerical methods in research, cite:

```bibtex
@misc{colaneri2026carbonsensitivefundconstructionhedging,
      title={Carbon-Sensitive Fund Construction and Hedging for Green Unit-Linked Life Insurance},
      author={Katia Colaneri and Alessandra Cretarola and Edoardo Lombardo and Daniele Mancinelli},
      year={2026},
      eprint={2509.05676},
      archivePrefix={arXiv},
      primaryClass={q-fin.RM},
      url={https://arxiv.org/abs/2509.05676},
}
```

# Pricing variance-reduction experiment

[README](README.md) · [Installation](INSTALLATION.md) · [Experiment interface](EXPERIMENTS.md)

This experiment compares the four pricing estimators for the PE, TI and EI contracts using the calibration of *Carbon-Sensitive Fund Construction and Hedging for Green Unit-Linked Life Insurance* and its supplementary material. It measures uncertainty in prices and deltas, rather than variance of realized hedging costs.

## Estimators

| Configuration | Financial randomness | Carbon sampling |
|---|---|---|
| `standard` | Simulated | Independent trajectories |
| `standard_antithetic` | Simulated with opposite financial inputs in each pair | Opposite Gaussian inputs in each pair |
| `conditioned` | Integrated analytically conditional on carbon | Independent trajectories |
| `conditioned_antithetic` | Integrated analytically conditional on carbon | Opposite Gaussian inputs in each pair |

All four integrate deterministic mortality analytically and use the same time grid, contractual guarantees, mortality weights and carbon model. Each antithetic trajectory advances through its own transition, including for CIR; no OU reflection identity is used.

Standard TI/EI pricing simulates a coherent sequential fund path. Standalone PE needs a terminal financial draw after integrating variance along the carbon path. Conditioned estimators integrate that financial randomness analytically. There is no fund-simulation selector.

## Benchmark calibration

The standalone executable `benchmark_petiei_variance_reduction` uses fixed model inputs and exposes numerical budgets through its command line. Use [Pricing_example.ipynb](Pricing_example.ipynb) or the Python configuration interface to choose all model inputs explicitly.

| Input | Benchmark value |
|---|---|
| Maturities | 5, 10, 20, 30 years |
| Intervals | `N = 5T`: 25, 50, 100, 150 |
| Policy | One age-60 policy, unit notional, evaluated separately for PE/TI/EI |
| Initial fund | 1 |
| Risk-free rate | 0.05 |
| Risk aversion | 1 |
| Carbon penalty | 0.0025 |
| Initial CIR intensities | 5000, 4000, 3000, 1000 |
| Long-run CIR means | 2500, 2000, 1500, 500 |
| Mean-reversion coefficients | 0.05 for each factor |
| CIR volatilities | 3 for each factor |
| Stock expected returns | 0.25, 0.15, 0.10, 0.08 |
| Stock volatilities | 0.30, 0.25, 0.20, 0.16 |
| Mortality | `xi = 0.0041959`, `m = 79.6921211`, `b = 11.5818911` |
| Lower guarantee | `exp(0.05 * t)` |
| Upper guarantee | `exp(0.50 * t)` |
| EI death-benefit fraction | 1 |

The benchmark uses the rounded correlation matrix printed in Table 4.1:

```text
1.00  0.44  0.39  0.32
0.44  1.00  0.30  0.33
0.39  0.30  1.00  0.31
0.32  0.33  0.31  1.00
```

The legacy code also contains more precise correlations: 0.4397, 0.3900, 0.3168, 0.2954, 0.3261 and 0.3134. Several hedging notebooks explicitly use these values. Record the actual matrix when comparing results; the rounded and full-precision calibrations are distinct inputs.

The Gompertz-Makeham exponent is `(age + t - m) / b`. The corrected Makeham value is `0.0041959`; older outputs using `0.00279` target a different mortality model. Original historical standard TI/EI variance entries also used a different dependence between financial draws at successive dates. They are not exact acceptance targets for the coherent sequential estimator.

## Build and run

Use a serial Release build for comparable process CPU timings:

```bash
cmake -S . -B build-paper-benchmark \
    -DCMAKE_BUILD_TYPE=Release \
    -DCARBONFUND_ENABLE_OPENMP=OFF \
    -DBUILD_TESTING=ON &&
cmake --build build-paper-benchmark --parallel &&
ctest --test-dir build-paper-benchmark --output-on-failure --no-tests=error
```

Start with a smaller run:

```bash
mkdir -p results &&
./build-paper-benchmark/benchmark_petiei_variance_reduction \
    --paths 10000 \
    --repetitions 1 \
    --seed 20261015 \
    --batches 0 \
    --output-prefix results/variance_smoke
```

For a larger experiment with repeated timings and statistical agreement checks:

```bash
mkdir -p results &&
./build-paper-benchmark/benchmark_petiei_variance_reduction \
    --paths 1000000 \
    --repetitions 3 \
    --seed 20261015 \
    --batches 100 \
    --batch-paths 2000 \
    --output-prefix results/paper_variance
```

Each invocation evaluates all four configurations, three contracts and four maturities. `--paths` is the per-configuration trajectory budget. Odd requests round up to an even count for all configurations. `--repetitions` controls timing repetitions; reported estimates come from the first repetition and CPU time is the median. Repetitions are not pooled into one larger estimate.

`--batches 0` skips agreement checks, so validation counts print `0/0`; this does not indicate a failed test. With agreement checks enabled, use at least two batches. Choose a new output prefix to retain a previous run. Create the parent directory before launching the standalone benchmark.

## Variances, standard errors and efficiency

For `M` actual trajectories, standard sampling has `n = M` independent observations. Antithetic sampling has `n = M/2` independent pair averages. In either case:

$$
\mathrm{SE}^2=\frac{s^2}{n}.
$$

The sample variance `s²` therefore has different observation units in the two sampling modes. At equal trajectory budgets, compare the variance of the estimated mean, or equivalently

$$
V_{\mathrm{effective}}=M\,\mathrm{SE}^2.
$$

This equals the single-path variance for independent sampling and twice the pair-average variance for antithetic sampling. Price and delta have separate variance estimates.

The variance reduction percentage is

$$
100\left(1-\frac{\mathrm{SE}_{\mathrm{method}}^2}
{\mathrm{SE}_{\mathrm{standard}}^2}\right).
$$

The standard row reports `0%`; negative values indicate increased estimator variance. Undefined ratios are left blank. The CPU efficiency ratio additionally compares `CPU_seconds * SE²`, so a reduction in variance can be assessed together with its computational cost. Price and delta are computed together and share the evaluation's CPU time.

## Agreement checks

Within each independently seeded validation batch, the configurations share carbon inputs and the standard modes share financial inputs. Comparisons use differences between estimates and retain the covariance induced by these common random inputs.

The benchmark saves individual and Bonferroni simultaneous Student t intervals for calibration grand-mean differences and for the main equal-budget differences. With all maturities, contracts, quantities and pairwise contrasts, the simultaneous family contains 144 comparisons. Coverage is an approximation at finite batch sizes.

Independent estimates can differ even when both are correct. Confidence-interval exclusions are reported as statistical outcomes rather than deterministic unit-test failures. The calibration output also compares empirical dispersion across batches with the standard errors reported within batches.

These checks compare the same discretized pricing target. They do not measure CIR discretization error, fund-integration error or mortality quadrature bias. Assess those separately by refining grids while controlling sampling uncertainty.

## Output files

For `--output-prefix results/paper_variance`, the benchmark writes:

| Suffix | Contents |
|---|---|
| `_summary.csv` | Price/delta means, observation variances, standard errors, effective variances, variance reduction, CPU times and efficiency. |
| `_timings.csv` | Individual timing repetitions. |
| `_parameters.json` | Model inputs, numerical settings, seed root and build information. |
| `_batches.csv` | Retained validation-batch estimates. |
| `_calibration.csv` | Empirical batch dispersion versus reported standard errors. |
| `_agreement.csv` | Calibration grand-mean difference intervals. |
| `_main_agreement.csv` | Main equal-budget estimate difference intervals. |
| `_table_e1.csv` | Comparison fields for the historical supplementary table. |

With `--batches 0`, validation CSVs may contain only headers. Keep the parameters JSON alongside the CSVs. Historical values should be compared only after reconciling mortality, correlation precision, financial-path dependence and numerical conventions.

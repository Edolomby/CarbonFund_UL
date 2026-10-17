# ASTIN four-configuration variance-reduction experiment

Measured on 15 October 2026 with GCC 13.3.0, Eigen 3.4.0, Boost 1.90.0 and C++20, in Release mode (`-O3 -DNDEBUG`, without fast-math or native-architecture flags). Simulation was serial, with OpenMP disabled for these measurements.

The library now supports standard and conditioned Monte Carlo, each with independent and antithetic sampling. The experiment was run at **one million actual trajectories per configuration**, for all three contracts and all four maturities. The confirmed mortality input is **xi = 0.0041959**, with exponent **(age + t - m) / b**.

The sequential measurements retained below are historical results of the 15 October study. All 144 sequential price/delta comparisons contained zero in their simultaneous intervals, using the historical declared family of 288 comparisons. The current benchmark uses one sequential fund construction and a family of 144 comparisons. The historical intervals have not been relabelled as new 144-family measurements.

## Four configurations

| Configuration | Financial treatment | Antithetic inputs |
|---|---|---|
| Standard | Simulate the financial Gaussian | None |
| Standard + antithetic | Simulate the financial Gaussian | Negate both carbon and financial normals |
| Formula | Integrate the financial Gaussian analytically | None |
| Formula + antithetic | Integrate the financial Gaussian analytically | Negate carbon normals |

Mortality is integrated analytically in every configuration, consistently with the standard estimators in Section E.2 of the supplement. Each of the three contracts is evaluated separately for timing, with a unit-notional age-60 policy.

For pure endowment alone, standard Monte Carlo uses E3: accumulate the trapezoidal integrated variance along the carbon path, then draw one financial Gaussian at maturity. The formula likewise needs only its terminal payoff evaluation. TI and EI evaluate benefits at the grid dates.

Standard and conditioned pricing share the same prepared transitions, allocation inputs, contractual bounds, discounted mortality weights, cohort notionals and quadrature. Conditional Monte Carlo remains the default API. Separate carbon and financial streams ensure that extra financial draws do not change the carbon paths.

## Exact benchmark inputs

- Maturities: T = 5, 10, 20, 30 years; N = 5T = 25, 50, 100, 150 intervals.
- Fund and policy notional: X0 = 1, notional = 1; initial age 60.
- Risk-free rate r = 0.05; risk aversion delta = 1; carbon penalty alpha = 0.0025.
- Initial carbon: (5000, 4000, 3000, 1000).
- Long-run carbon: (2500, 2000, 1500, 500).
- All carbon mean-reversion coefficients: 0.05; all CIR volatilities: 3.
- Asset expected returns: (0.25, 0.15, 0.10, 0.08).
- Asset volatilities: (0.30, 0.25, 0.20, 0.16).
- Asset covariance: diag(volatility) * correlation * diag(volatility), using the rounded correlation matrix printed in Table 4.1:

```
1.00  0.44  0.39  0.32
0.44  1.00  0.30  0.33
0.39  0.30  1.00  0.31
0.32  0.33  0.31  1.00
```

- Mortality: xi = 0.0041959, m = 79.6921211, scale b = 11.5818911. The library receives growth rate 1/b.
- Guarantees: lower = exp(r*t), upper = exp(10*r*t), anchored at the original contract fund of one. These contractual amounts stay fixed when differentiating with respect to the current fund value.
- EI death-benefit fraction: 1.
- CIR transition: the existing four-factor Ninomiya-Victoir scheme; fund variance and mortality integrals use the same trapezoidal grid in all modes.
- RNG: Boost xoshiro256pp plus Boost Ziggurat for every configuration; seed root 20261015. SplitMix64 derives the recorded per-case/run seeds and separate carbon/financial stream seeds. AS241 is not added to the production project.

The original C++ source uses more precise off-diagonal correlations (0.4397, 0.3900, 0.3168, 0.2954, 0.3261, 0.3134). The parameter reconciliation below tests those values too. The primary results above deliberately use the publicly printed matrix, and the complete chosen inputs are written to each parameters JSON.

## Sequential fund simulation

Standard TI/EI pricing always implements the sequential scheme E2:

```math
\Delta v_j=\frac h2(q_j+q_{j+1}),\qquad
X_{j+1}=X_j\exp\left(rh-\frac12\Delta v_j+\sqrt{\Delta v_j}F_{j+1}\right),
\qquad q_j=\pi_j^\top\Sigma\Sigma^\top\pi_j.
```

Standalone PE uses E3: it accumulates integrated variance over the carbon path
and draws one financial Gaussian at maturity. Conditioned estimators integrate
financial randomness analytically. These rules are fixed by the selected
contract and estimator; there is no fund-simulation setting or command-line flag.

The original published standard TI/EI variance entries used a different
financial dependence between dates, so exact equality with those historical
standard variances is not a validation target for the sequential estimator.

## Statistics and confidence checks

Each ordinary configuration uses M independent observations. Each antithetic configuration uses M/2 independent pair means, with both trajectories advanced through the normal carbon stepper. No OU reflection formula is used.

The sample variance column refers to one independent observation. Comparisons at equal actual trajectory counts use

$$
V_{\mathrm{effective}}=M\,\mathrm{SE}^2
=\begin{cases}s_{\mathrm{path}}^2,&\text{ordinary},\\
2s_{\mathrm{pair}}^2,&\text{antithetic}.
\end{cases}
$$

Variance reduction is `100 * (1 - SE_configuration^2 / SE_standard^2)` at the common actual budget. CPU efficiency is estimated by `(CPU_standard * SE_standard^2) / (CPU_configuration * SE_configuration^2)`. Undefined ratios are left blank in the CSV.

The million-trajectory measurements use one full-size CPU timing per case. Preparation and stream construction are excluded; all modes are warmed up and results are made observable. The program supports repeated timings and rotates execution order; use `--repetitions 3` to obtain a median on your hardware. Price and delta are evaluated together, so their rows share the same measured CPU cost. CPU times here describe this execution environment.

For the historical sequential study, 100 separately seeded calibration batches of 2,000 actual trajectories per configuration were retained. Within a batch the modes share carbon inputs, and standard modes share financial inputs. Across batches fresh streams are used.

For each comparison, D_b is the difference of the two batch estimates, so its sample variance s_D^2 includes their covariance. With m calibration trajectories and M main trajectories, the main-estimate difference has estimated SE

$$
\widehat{\mathrm{SE}}(D_M)=\sqrt{\frac mM\,s_D^2}.
$$

The main confidence interval is `D_M +/- t_(R-1) * SE(D_M)`. A separate CSV reports intervals for the calibration grand-mean difference, `mean(D_b) +/- t_(R-1) * s_D / sqrt(R)`. The historical intervals below used a declared 288-comparison family for each inference collection. The current benchmark declares 144 comparisons: four maturities x three contracts x two quantities x six contrasts, with tail probability `0.05 / (2 * 144)`. Student-t coverage is approximate at these finite Monte Carlo batch sizes.

| Fund construction | Million-budget individual 95% containing zero | Million-budget simultaneous 95% containing zero | Calibration grand-mean individual 95% containing zero | Calibration grand-mean simultaneous 95% containing zero |
|---|---:|---:|---:|---:|
| Sequential E2 | 126/144 | 144/144 | 134/144 | 144/144 |

The nominal exclusions are correlated and are not independent failures. No seeds or observations were discarded, and no fresh runs were selected until an interval passed.

Observed standard errors were also checked against the spread across independent batches:

- Sequential E2: observed batch SD / RMS reported SE ranged from **0.8625 to 1.1337** across prices, deltas and configurations.

The finite number of batches produces sampling uncertainty in these SD ratios. The earlier exact pair reconstruction and the new deterministic tests provide additional checks of the statistics.

An independent Python/SciPy two-pass calculation verified the means, variances, effective variances, standard errors and both kinds of confidence interval. Maximum numerical discrepancy was 2.23e-16. The sequential main-estimate intervals were derived from its retained measurements after the C++ exporter was extended; pricing was not rerun or selected.

## Example: 20-year EI, sequential fund

| Configuration | Price | Price SE | Delta | Delta SE | CPU seconds |
|---|---:|---:|---:|---:|---:|
| Standard | 1.180211730 | 3.214182e-04 | 0.588823540 | 6.392472e-04 | 10.140 |
| Standard + antithetic | 1.180017143 | 2.701560e-04 | 0.588761200 | 3.742381e-04 | 8.044 |
| Formula | 1.180231313 | 8.146243e-06 | 0.588969805 | 4.072243e-06 | 14.681 |
| Formula + antithetic | 1.180241662 | 1.696041e-06 | 0.588974981 | 8.480099e-07 | 10.882 |

Relative to standard Monte Carlo at the same million-trajectory budget:

- Formula alone: price SE 39.46x smaller; delta SE 156.98x smaller.
- Formula plus antithetic: price SE 189.51x smaller; delta SE 753.82x smaller.
- Price variance reductions: 99.9358% for the formula, 99.9972% for the combination.

Across all sequential-fund price cases, conditioning alone reduced variance by 99.9321% to 99.9738%; conditioning plus antithetic reduced it by 99.9972% to 99.9996%.

The confidence checks compare the same discretized pricing target. They quantify Monte Carlo uncertainty; the common CIR/time-grid and mortality-quadrature approximation errors are separate.

## Updated variance tables: sequential E2 fund

Entries are effective price variances at equal actual trajectory budgets. Ordinary entries are the sample variances of single contributions; antithetic entries are twice the sample variances of pair means.

### PE

| Configuration | T = 5 | T = 10 | T = 20 | T = 30 |
|---|---:|---:|---:|---:|
| Standard | 2.2995162e-02 | 4.1444336e-02 | 3.5208150e-02 | 4.2599965e-03 |
| Standard + antithetic | 1.5139179e-02 | 2.9458176e-02 | 2.7577864e-02 | 3.5518276e-03 |
| Formula | 7.1842040e-06 | 2.0272185e-05 | 2.0373804e-05 | 2.0702443e-06 |
| Formula + antithetic | 1.5994609e-07 | 7.4723849e-07 | 9.9619910e-07 | 1.0111404e-07 |

### TI

| Configuration | T = 5 | T = 10 | T = 20 | T = 30 |
|---|---:|---:|---:|---:|
| Standard | 1.0801380e-04 | 1.3574301e-03 | 2.1795619e-02 | 8.3579212e-02 |
| Standard + antithetic | 5.7587090e-05 | 7.8512805e-04 | 1.4185279e-02 | 5.8022844e-02 |
| Formula | 2.8349158e-08 | 6.1668263e-07 | 1.4074431e-05 | 5.6742902e-05 |
| Formula + antithetic | 4.1208077e-10 | 1.5884971e-08 | 5.3508575e-07 | 2.3539631e-06 |

### EI

| Configuration | T = 5 | T = 10 | T = 20 | T = 30 |
|---|---:|---:|---:|---:|
| Standard | 2.5837553e-02 | 5.5543900e-02 | 1.0330968e-01 | 1.1844056e-01 |
| Standard + antithetic | 1.6663472e-02 | 3.7778061e-02 | 7.2984261e-02 | 8.3096072e-02 |
| Formula | 8.0926442e-06 | 2.7547460e-05 | 6.6361270e-05 | 7.8602737e-05 |
| Formula + antithetic | 1.7488870e-07 | 9.6279862e-07 | 2.8765562e-06 | 3.2862083e-06 |

## Reconciliation with the published Table E1

The old code's active Makeham constant was 0.00279. Repeating its input profile with the source's full-precision correlations reproduced the 12 conditioned price variances in Table E1 using 200,000 carbon trajectories. Relative differences ranged from -0.3087% to 0.5322%. This supports the explanation that the reported table came from the old code's mortality input and matrix precision.

With a fixed financial setup, changing xi rescales the PE survival weight, and therefore

$$
\mathrm{Var}_{\mathrm{PE,new}}
=\mathrm{Var}_{\mathrm{PE,old}}
\exp\{-2(0.0041959-0.00279)T\}.
$$

For TI and EI the death-density weights also change, so a single terminal rescaling is insufficient. Their updated variances were recomputed above. Updated outputs should therefore replace, rather than be forced to match, the old raw table entries.

## Implementation and validation

The existing default call remains conditioned/antithetic:

```cpp
const auto result = Pricer::evaluate(prepared, M, carbon_rng);
```

An explicit standard call adds a separate financial stream:

```cpp
random::SimulationRng carbon_rng{20261015};
random::SimulationRng financial_rng{20261016};
const auto result = Pricer::evaluate(
    prepared, M, carbon_rng, financial_rng,
    pricing::PetieiEstimator::standard,
    pricing::PetieiSampling::antithetic,
    pricing::PetieiContract::endowment_insurance);
```

Both streams advance by reference. `PetieiEstimator::conditioned` selects the formula and does not advance the financial stream. The optional contract selection also supports PE, TI and all three together; unselected result fields are zero.

For fund delta, guarantees stay fixed. The standard estimator uses `(X_t / X0) * indicator(lower < X_t < upper)` with the same discounted mortality weights as the conditioned derivative. At an isolated payoff kink a symmetric derivative is used.

One boundary correction was required: when floor and cap coincide at t=0, the immediate payoff is constant and its fund delta is zero. The old routine used one there. Prices and variances are unchanged analytically; TI/EI delta loses the erroneous deterministic initial quadrature contribution. With the paper's unit-notional inputs and fraction one, that contribution is `(h / 2) * mortality_hazard(age=60)` for both TI and EI.

The deterministic validation tests cover Gaussian quadrature of the realized payoff/delta versus the conditional formula; independent scalar OU, GeoOU and CIR transitions with a general allocation solve; sequential fund simulation; standalone PE E3; antithetic means/variances and scalar draw counts; contract selection; fixed-guarantee finite differences; and the corrected initial endpoint. An odd-budget CLI smoke test verified rounding from 101 requested to 102 actual trajectories in every configuration.

Regression checks retain the previous sequential price/delta moments and hedging outcomes across OU, GeoOU and CIR. Removing the sampling selector preserves those results, including antithetic pair statistics and the terminal-only PE draw count.

## Install the files

Keep the pricing header in `include/carbonfund/pricing/`, estimator tests in
`tests/`, and `benchmark_petiei_variance_reduction.cpp` in `benchmarks/`.
The Python runner, module, configurations and integration checks are described
in [EXPERIMENTS.md](EXPERIMENTS.md). Existing CMake target definitions remain
valid when their source paths point to these locations.

Then configure, build and test with system Boost:

```bash
cd ~/Documents/codes/CarbonFund_UL

cmake -S . -B build-paper-release \
    -DBoost_ROOT=/usr \
    -DCMAKE_BUILD_TYPE=Release \
    -DCARBONFUND_ENABLE_OPENMP=OFF \
    -DBUILD_TESTING=ON &&

cmake --build build-paper-release --parallel &&
ctest --test-dir build-paper-release --output-on-failure --no-tests=error
```

Exact settings of the measured sequential run:

```bash
mkdir -p results

./build-paper-release/benchmark_petiei_variance_reduction \
    --paths 1000000 --repetitions 1 --seed 20261015 \
    --batches 100 --batch-paths 2000 \
    --output-prefix results/paper_path
```

For median timings use `--repetitions 3`. A shorter initial check can use `--paths 100000 --repetitions 1 --batches 20 --batch-paths 2000`. Every invocation evaluates all four configurations, all three contracts and all four maturities. Odd trajectory requests are rounded up for all modes so the actual budget stays equal. `--batches 0` performs timings/main estimates without the repeated-batch confidence checks.

The benchmark returns zero when it completes successfully; it prints and records statistical exclusions rather than making random confidence tests into unit-test assertions. Output prefixes must differ if you want to retain two runs.

## Saved results

### Sequential fund

- [paper_path_summary.csv](sandbox:/workspace/scratch/7ab786f8e272/paper-variance-study/results/paper_path_summary.csv): Means, SEs, variances, CPU times and efficiency.
- [paper_path_main_agreement.csv](sandbox:/workspace/scratch/7ab786f8e272/paper-variance-study/results/paper_path_main_agreement.csv): Covariance-aware million-budget comparison intervals.
- [paper_path_agreement.csv](sandbox:/workspace/scratch/7ab786f8e272/paper-variance-study/results/paper_path_agreement.csv): Calibration grand-mean comparison intervals.
- [paper_path_calibration.csv](sandbox:/workspace/scratch/7ab786f8e272/paper-variance-study/results/paper_path_calibration.csv): Observed versus reported SE.
- [paper_path_batches.csv](sandbox:/workspace/scratch/7ab786f8e272/paper-variance-study/results/paper_path_batches.csv): All retained calibration observations.
- [paper_path_timings.csv](sandbox:/workspace/scratch/7ab786f8e272/paper-variance-study/results/paper_path_timings.csv): Individual measured CPU times.
- [paper_path_table_e1.csv](sandbox:/workspace/scratch/7ab786f8e272/paper-variance-study/results/paper_path_table_e1.csv): Raw comparison with original Table E1.
- [paper_path_parameters.json](sandbox:/workspace/scratch/7ab786f8e272/paper-variance-study/results/paper_path_parameters.json): Complete parameters and run settings.

- [four_configuration_comparison.csv](sandbox:/workspace/scratch/7ab786f8e272/paper-variance-study/results/four_configuration_comparison.csv)
- [parameter_reconciliation.csv](sandbox:/workspace/scratch/7ab786f8e272/paper-variance-study/results/parameter_reconciliation.csv)
- [legacy_standard_checks.csv](sandbox:/workspace/scratch/7ab786f8e272/paper-variance-study/results/legacy_standard_checks.csv)

Sources: [Main paper, Table 4.1 and Section 6](sandbox:/workspace/scratch/7ab786f8e272/upload/ASTIN_rev1.pdf); [Supplement, Section E and Table E1](sandbox:/workspace/scratch/7ab786f8e272/upload/ASTIN_Supl_Mat.pdf). The mortality sign and Makeham value follow the author's corrections in this conversation.

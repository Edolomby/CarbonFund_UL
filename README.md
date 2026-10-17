# CarbonFund_UL

**Pricing and risk-minimizing hedging of life insurance portfolios linked to a fund with carbon preferences.**

CarbonFund_UL is a C++20 research library for studying the risks faced by an insurer issuing unit-linked life insurance contracts. The policy benefits depend on an investment fund whose allocation responds to the carbon intensity of the firms in which it invests.

The central question is how effectively the insurer can hedge this portfolio when financial risk is tradable, while carbon intensity and policyholder mortality introduce additional, non-traded risks. The library connects fund construction, insurance valuation and hedging simulation so that their effects can be studied together.

It implements numerical building blocks from *Carbon-Sensitive Fund Construction and Hedging for Green Unit-Linked Life Insurance* and its supplementary material, together with antithetic sampling, individual policyholder bookkeeping and tools for comparing hedging-cost distributions.

## What the library is designed to investigate

- How carbon preferences change investment allocations and the fund underlying the insurance benefits.
- How guarantees, mortality, ages and notionals affect portfolio prices and hedge positions.
- How dynamic hedging compares with an initial-delta strategy and with holding the premium in cash.
- How pooling independent policyholders reduces the relative effect of individual mortality risk, while financial and carbon scenarios remain common to the portfolio.
- How conditioning and antithetic variables reduce the computational effort needed to estimate prices and deltas accurately.

Two different variances matter in these experiments. **Estimator variance** measures Monte Carlo uncertainty in a computed price or delta. **Hedging-cost variance** measures the insurer's residual risk across realized scenarios. The pricing benchmarks study the first; the hedging engine and Python analysis study the second.

## The underlying fund and its carbon preferences

The current implementation uses four risky assets and a risk-free account. Each firm has a stochastic carbon-intensity factor. The fund's risky-asset weights solve

```math
\left[\delta A+\alpha\,\mathrm{diag}
\left(A_{11}C_1^+,\ldots,A_{44}C_4^+\right)\right]\pi^*(C)
=\mu-r\mathbf{1},
\qquad C_i^+=\max(C_i,0).
```

Here, $A$ is the asset-return covariance matrix, $\delta$ is risk aversion, and $\alpha$ controls the carbon penalty. These correspond to `covariance`, `risk_aversion` and `carbon_penalty` in `CarbonAllocationParameters4`. The remaining wealth is allocated to the risk-free account. Setting the penalty to zero recovers the corresponding allocation without carbon preferences.

Carbon preferences therefore enter the allocation rule itself. They change the fund's exposures and, through the fund, the insurer's liabilities and hedge positions.

The available carbon models are:

| Model | Implementation |
|---|---|
| Ornstein-Uhlenbeck | Exact transitions for the carbon factors. |
| Geometric Ornstein-Uhlenbeck | Exact transitions in log carbon intensity. |
| Cox-Ingersoll-Ross | Ninomiya-Victoir splitting scheme. |

The CIR scheme requires $\kappa_i\bar C_i\geq\sigma_i^2/4$. Carbon-path simulation and integration of the fund's variance remain separate numerical steps, including when the carbon transitions are exact.

## The insurer's portfolio

The library supports three contract types, abbreviated PE, TI and EI:

| Contract | Benefit |
|---|---|
| Pure endowment (PE) | Pays at maturity if the policyholder survives. |
| Term insurance (TI) | Pays on death before maturity. |
| Endowment insurance (EI) | Combines a survival benefit with a death benefit, optionally scaled by a contractual fraction. |

Benefits use a fund payoff with a minimum guarantee and an upper cap:

```math
H(t,x)=\min\!\left(\max(x,L(t)),U(t)\right),
\qquad
L(t)=L_0e^{g_Lt},\quad U(t)=U_0e^{g_Ut}.
```

The payoff is expressed per unit of policy notional. Prices and deltas returned for a portfolio already include its notionals, mortality weights and discounting.

Two representations serve different purposes:

- `CohortBook` groups living policies by age and aggregate notional for valuation. Policies in a cohort may have different individual notionals.
- `PolicyholderPopulation` retains individual ages, notionals, contract types and survival states for realized deaths and benefit payments during hedging.

Mortality follows a deterministic Gompertz-Makeham law, independently of financial and carbon risk. Individual lifetimes are independent in the population simulation. A pricing preparation uses common fund and contract terms; books with different terms require separate preparations.

## Hedging in an incomplete market

The paper uses local risk-minimization and the Föllmer-Schweizer decomposition to identify the tradable financial exposure and the residual cost arising from unspanned risks. The numerical objective is to approximate this hedge and examine its performance on portfolios of policies.

Valuation uses the **minimal martingale measure**: the fund's financial drift becomes the risk-free rate, while the carbon and mortality laws are unchanged under the independence assumptions. The asset expected returns still enter the fund's allocation rule.

The hedge is expressed in fund units through the portfolio delta

```math
\Delta_t=\frac{\partial V}{\partial x}
\left(t,X_t,C_t,\text{living portfolio}\right).
```

Holding $\Delta_t$ units of the fund gives risky-asset exposures $\Delta_t X_t\pi_i^*(C_t)$. The hedge account also holds cash. As time passes, valuation uses the observed fund and carbon state, the remaining maturity, the evolved contractual guarantees and the surviving policyholders.

The engine compares three strategies on the **same market path and realized deaths**, starting from the same initial premium:

| Strategy | Fund position |
|---|---|
| No hedge | Zero fund units; the premium is held in the cash account. |
| Mortality-adjusted static hedge | Initial aggregate deltas, scaled by the surviving-notional ratio within each contract type. No market-based repricing. |
| Dynamic hedge | Aggregate delta recalculated from the current state and living portfolio at each rebalancing date. |

The static strategy is an explicitly defined benchmark. Its mortality adjustment also applies to portfolios with heterogeneous ages and notionals; it does not recompute cohort-specific deltas.

All reported costs are **discounted to time zero**, with positive values meaning additional funding required:

```math
\text{cost}
=\sum_{k=1}^{K}e^{-rt_k}D_k-V_0
-\sum_{k=1}^{K}\theta_{k-1}
\left(e^{-rt_k}X_{t_k}-e^{-rt_{k-1}}X_{t_{k-1}}\right).
```

Here, $D_k$ is the benefit paid at a settlement date, $V_0$ is the initial premium and $\theta_{k-1}$ is the fund holding over the preceding interval. Dynamic capital additions and withdrawals are recorded at their discounted values. A negative cost is a net surplus.

Dynamic hedging targets the financial exposure; carbon and mortality risk still contribute to residual costs. Its numerical performance is assessed from the resulting distributions, together with the effects of rebalancing frequency and pricing accuracy.

## Efficient pricing and delta estimation

Conditional on a carbon path, the allocation path is known and the fund has a lognormal conditional distribution at each payment date. Its integrated variance is determined by that allocation. This allows the min/max payoff and its fund derivative to be evaluated analytically, as developed in Section E of the supplementary material.

The pricer integrates mortality analytically in both estimator families:

| Configuration | Financial noise | Sampling |
|---|---|---|
| `standard` | Simulated | Independent trajectories |
| `standard_antithetic` | Simulated | Antithetic trajectory pairs |
| `conditioned` | Integrated analytically | Independent carbon trajectories |
| `conditioned_antithetic` | Integrated analytically | Antithetic carbon pairs |

Conditioned antithetic pricing is the default. At each carbon step, a four-dimensional Gaussian vector $Z$ drives one trajectory and $-Z$ drives its partner. Each trajectory is advanced through its own model transition, so pairing works for OU, GeoOU and CIR without an OU reflection identity. Standard antithetic sampling also negates the financial Gaussian inputs.

An antithetic pair contributes **one pair average** to the statistics. `sample_variance` is the variance of these independent observations, and

```math
\mathrm{SE}(\widehat V)=\sqrt{s^2/n},
```

where $n$ is the number of independent paths or pairs. The same convention applies to delta. Odd antithetic requests round up to the next even trajectory count.

Conditioning preserves the expectation of the discretized estimator and reduces its variance. The benefit from antithetic sampling is measured by the benchmarks. Increasing the path count controls sampling uncertainty; refining the time grid addresses discretization error.

## Build and run

The library is header-only and requires C++20, CMake 3.22 or newer, Eigen 3.4 and Boost 1.88 or newer. Boost 1.90 is supported. See [INSTALLATION.md](INSTALLATION.md) for dependency installation and platform-specific instructions.

Run project commands from the repository root. For a serial Release build using installed dependencies:

```bash
cmake -S . -B build-paper-release \
    -DCMAKE_BUILD_TYPE=Release \
    -DCARBONFUND_ENABLE_OPENMP=OFF \
    -DBUILD_TESTING=ON &&
cmake --build build-paper-release --parallel &&
ctest --test-dir build-paper-release --output-on-failure --no-tests=error
```

### Price contracts and calculate deltas

```bash
./build-paper-release/example_petiei_monte_carlo
```

The complete setup is in [examples/petiei_monte_carlo.cpp](examples/petiei_monte_carlo.cpp). It constructs PE, TI and EI cohort books, prepares the carbon model and contract inputs, then returns prices and fund deltas.

Within an application, after constructing those inputs:

```cpp
using Pricer = carbonfund::pricing::PetieiMonteCarloPricer4<
    carbonfund::models::OUExactStepper4>;

const auto prepared = Pricer::prepare(
    contract, current_carbon, carbon, allocation, portfolio, mortality);

carbonfund::random::SimulationRng rng{20261015};
const auto result = Pricer::evaluate(prepared, 10000, rng);

const double price = result.endowment_insurance.price.mean;
const double delta = result.endowment_insurance.delta.mean;
const double price_se = result.endowment_insurance.price.standard_error;
const double delta_se = result.endowment_insurance.delta.standard_error;
```

Delta is the derivative with respect to the supplied current fund value, holding the contractual guarantees fixed. It already includes aggregate policy notionals. A new valuation is prepared when the observed fund, carbon state, remaining term or living book changes.

### Generate hedging-cost distributions

```bash
./build-paper-release/example_petiei_hedging 100 128 results/hedging_CIR
```

The arguments are the number of outer scenarios, the inner path count for subsequent valuations, and the output prefix. The example creates the output directory and writes:

- `results/hedging_CIR.csv`: one row per scenario, with the three strategy costs, discounted claims, trading gains, terminal fund and initial premium.
- `results/hedging_CIR_parameters.json`: model and contract inputs, initial population, grids, path counts and seeds.

The initial premium uses a separate 8,192-path valuation. The example has illustrative CIR parameters and a mixed portfolio; it is not the paper's full hedging experiment. Its complete setup is in [examples/petiei_hedging.cpp](examples/petiei_hedging.cpp).

Outer scenarios follow the physical measure and contain realized carbon and fund paths. The engine accepts these through `MarketPath4`, including externally generated paths. Inner valuation uses the minimal martingale measure and sees only the current observation and living portfolio. Separate market, mortality and pricing streams keep realized scenarios independent of the chosen inner simulation budget.

The small command above checks the workflow. Distribution and tail estimates require substantially more outer scenarios. Increasing the inner path count improves the precision of the hedge positions; it does not increase the number of observed hedging outcomes.

### Analyse and plot the results

Create the Python environment once:

```bash
python3 -m venv .venv-analysis &&
./.venv-analysis/bin/python -m pip install numpy matplotlib
```

Then run:

```bash
./.venv-analysis/bin/python tools/analyze_hedging.py results/hedging_CIR.csv
```

The analyser writes statistics and PNG/PDF plots under `hedging_analysis/hedging_CIR/`. Results include means, sample variances, standard deviations, mean standard errors, quantiles, expected shortfall and paired strategy comparisons. Plots show histograms, empirical distribution functions and box plots.

For comparisons between portfolio sizes, use a consistent exposure normalization, available through `--divide-by`. Mortality diversification concerns relative risk: increasing the number of policies also increases aggregate exposure.

### Compare pricing variance and CPU time

```bash
mkdir -p results &&
./build-paper-release/benchmark_petiei_variance_reduction \
    --paths 100000 \
    --repetitions 3 \
    --seed 20261015 \
    --batches 0 \
    --output-prefix results/paper_path
```

This compares the four configurations for PE, TI and EI at maturities 5, 10, 20 and 30 years. It prints prices, deltas, sample variances, standard errors, variance reduction percentages and CPU times, and saves the detailed results and parameters. The benchmark source belongs in `benchmarks/`.

At equal actual trajectory count, the reported reduction for price or delta is

```math
\mathrm{VR}(\%)=100\left(1-
\frac{\mathrm{SE}_{\text{method}}^2}
     {\mathrm{SE}_{\text{standard}}^2}\right).
```

The standard row reports `0%`. This definition accounts for the smaller number of independent observations in antithetic mode. A raw path variance and a raw pair-average variance have different sample units and should not be compared directly.

`--batches 0` skips statistical agreement checks. To enable batch-based confidence intervals for price and delta differences, repeat the command with, for example, `--batches 100 --batch-paths 2000`. Standard TI/EI pricing always uses a coherent sequential fund trajectory. Standalone PE simulates only its terminal financial value after the carbon path; conditioned estimators integrate financial randomness analytically.

The paper-parameter benchmark uses the corrected Makeham constant $\xi=0.0041959$ and the Gompertz exponent $(\text{age}+t-m)/b$. Examples have their own inputs; benchmark results should be interpreted using the parameters saved with each run.

## Project components

Paths below are relative to `include/carbonfund/`.

| Component | Responsibility |
|---|---|
| `models/` | Four-factor carbon dynamics. |
| `portfolio/` | Carbon-sensitive allocation and instantaneous fund variance. |
| `contracts/` | Aggregate cohorts and individual policyholder states. |
| `mortality/` | Gompertz-Makeham hazard and survival probabilities. |
| `pricing/` | Financial payoff kernels, insurance prices, deltas and Monte Carlo statistics. |
| `simulation/` | Physical carbon/fund scenarios and externally supplied market paths. |
| `hedging/` | Revaluation, strategy positions and discounted cash accounting. |
| `io/` | Scenario-level hedging CSV export. |
| `random/` | Boost xoshiro256++ streams and Ziggurat normal sampling. |
| `utils/` | Tabular console output. |

Link an application through the CMake interface target:

```cmake
add_executable(my_pricer path/to/my_pricer.cpp)
target_link_libraries(my_pricer PRIVATE CarbonFund::carbonfund)
```

`CarbonFund::carbonfund` supplies the include paths, C++20 requirement and dependencies.

## Numerical scope

The implemented pricing and hedging loops are serial. The model uses constant financial parameters, independent carbon factors and financial/carbon noise, independent deterministic-intensity mortality, and frictionless trading with a common borrowing and lending rate. Dynamic hedging is performed on a finite grid; deaths are settled at interval endpoints. Inner Monte Carlo error and time discretization both affect the realized numerical hedge.

Automated tests cover the carbon models, allocation, RNG, mortality, contracts, payoff kernels, estimators, population bookkeeping and hedging accounts. Pricing checks include delta finite differences and antithetic statistics; hedging checks include discounted cash identities and notional scaling. Statistical agreement and computational performance are evaluated separately by the benchmarks.

## Research reference

Katia Colaneri, Alessandra Cretarola, Edoardo Lombardo and Daniele Mancinelli (2026), *Carbon-Sensitive Fund Construction and Hedging for Green Unit-Linked Life Insurance*, and the accompanying supplementary material.

The main paper develops carbon-sensitive fund construction and the insurance hedging framework, with numerical hedging experiments in Section 6. Section E of the supplementary material describes fund simulation, conditioning formulas and the variance-reduction study. The library supports experiments based on this framework; parameter choices, numerical conventions and simulation budgets must be specified for each comparison.

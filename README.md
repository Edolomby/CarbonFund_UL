# CarbonFund_UL

Monte Carlo pricing and fund deltas for insurance contracts with carbon-sensitive investment strategies.

The current library prices three contract types:

| Contract | Benefit |
|---|---|
| Pure endowment (PE) | A benefit at maturity if the policyholder survives. |
| Term insurance (TI) | A benefit on death before maturity. |
| Endowment insurance (EI) | A death benefit before maturity and a survival benefit at maturity. |

Benefits use a fund payoff with a minimum guarantee and a maximum guarantee. The investment allocation depends on four carbon factors.

The implemented pricer is serial conditional Monte Carlo. It returns portfolio prices, fund deltas, sample variances and standard errors. The outer simulation of policyholder deaths, hedge rebalancing and trading gains is a subsequent layer; it is not implemented in this pricer.

## Installation and build

See [INSTALLATION.md](INSTALLATION.md) for dependencies, platform setup, build commands, automated tests and Release builds.

## Example 1: price PE, TI and EI portfolios

The following complete program is provided in `examples/petiei_monte_carlo.cpp` and is built as `example_petiei_monte_carlo`.

```cpp
#include <carbonfund/pricing/petiei_monte_carlo.hpp>
#include <carbonfund/random/simulation_rng.hpp>

#include <iomanip>
#include <iostream>

int main()
{
    using namespace carbonfund;
    using Pricer = pricing::PetieiMonteCarloPricer4<models::OUExactStepper4>;

    const contracts::CohortBook pe{{{60.0, 100000.0}}};
    const contracts::CohortBook ti{{{60.0, 100000.0}}};
    const contracts::CohortBook ei{{{60.0, 100000.0}}};
    const contracts::PetieiPortfolioView portfolio{
        pe.cohorts(), ti.cohorts(), ei.cohorts()};

    const mortality::GompertzMakeham mortality{
        {0.00279, 79.6921211, 1.0 / 11.5818911}};

    pricing::PetieiMonteCarloParameters contract;
    contract.fund_value = 1.0;
    contract.minimum_guarantee = 1.0;
    contract.maximum_guarantee = 1.0;
    contract.minimum_guarantee_growth_rate = 0.0;
    contract.maximum_guarantee_growth_rate = 0.05;
    contract.endowment_death_benefit_fraction = 1.0;
    contract.time_to_maturity = 5.0;
    contract.time_intervals = 60;

    const Eigen::Vector4d current_carbon{1.0, 1.2, 0.8, 1.1};
    models::OUParameters4 carbon;
    carbon.long_run_mean = current_carbon;
    carbon.mean_reversion.setConstant(0.5);
    carbon.volatility.setConstant(0.2);

    portfolio::CarbonAllocationParameters4 allocation;
    allocation.risk_free_rate = 0.03;
    allocation.risk_aversion = 2.0;
    allocation.carbon_penalty = 1.0;
    allocation.expected_returns << 0.06, 0.08, 0.07, 0.09;
    allocation.covariance =
        Eigen::Vector4d{0.04, 0.06, 0.05, 0.08}.asDiagonal();

    const auto prepared = Pricer::prepare(
        contract, current_carbon, carbon, allocation, portfolio, mortality);

    random::SimulationRng rng{20261011};
    const auto result = Pricer::evaluate(prepared, 1000, rng);

    const auto print = [](
        const char* name, const pricing::PetieiContractEstimate& value) {
        std::cout << name
                  << ": price=" << value.price.mean
                  << ", price standard error=" << value.price.standard_error
                  << ", delta=" << value.delta.mean
                  << ", delta standard error=" << value.delta.standard_error
                  << '\n';
    };

    std::cout << std::fixed << std::setprecision(6);
    print("PE", result.pure_endowment);
    print("TI", result.term_insurance);
    print("EI", result.endowment_insurance);
}
```

The example uses one age-60 cohort of total notional 100,000 for each contract type. These are three separate example portfolios. The fund and guarantees are expressed per unit of policy notional.

The mortality parameters are annual quantities. `gompertz_growth_rate` is the reciprocal of the scale used in the earlier Gompertz-Makeham parameterization.

With the validated configuration, the output is:

```text
PE: price=87258.346513, price standard error=1.187195, delta=30281.933712, delta standard error=18.923561
TI: price=10509.401796, price standard error=0.058239, delta=2600.719298, delta standard error=1.411592
EI: price=97767.748309, price standard error=1.242668, delta=32882.653011, delta standard error=20.219508
```

These are results for the illustrative inputs above. Last digits can differ across compiler and dependency versions.

### Read the results

Each contract estimate contains a `price` and a `delta`, both with these fields:

| Field | Meaning |
|---|---|
| `mean` | Estimated discounted portfolio price or its fund delta. |
| `sample_variance` | Unbiased variance of one conditional path contribution. |
| `standard_error` | Estimated Monte Carlo uncertainty of the mean: `sqrt(sample_variance / simulations)`. |

For example, the PE portfolio price is approximately 87,258.35. Its delta is

$$
\Delta_{PE}=\frac{\partial\,\text{price}_{PE}}{\partial X_0}.
$$

A small increase of the current fund value from 1.00 to 1.01 changes the PE price by approximately `30281.933712 * 0.01 = 302.82`, with guarantees and other valuation inputs held fixed. The derivative is with respect to the fund normalization used in `fund_value`; hedge quantities must use that same normalization.

Use the values directly in an application:

```cpp
const double ei_price = result.endowment_insurance.price.mean;
const double ei_delta = result.endowment_insurance.delta.mean;
const double ei_price_se = result.endowment_insurance.price.standard_error;
```

The PETIEI prices already include mortality, aggregate notional and discounting. The PETIEI deltas already include those same weights.

In this example, the identical cohorts and EI death-benefit fraction of one imply `EI price = PE price + TI price`, and likewise for delta. EI variance is calculated from the combined benefit on each path, retaining the covariance between its death and maturity contributions.

Standard error measures Monte Carlo sampling uncertainty. Time-grid error in carbon/variance integration and mortality quadrature is separate. Increasing the path count by a factor of four should approximately halve standard error; refining the time grid addresses discretization error.

Integrated fund variance describes the conditional financial distribution along one carbon path. `price.sample_variance` and `delta.sample_variance` describe variation of the conditional contributions across many carbon paths; they are different quantities.

## Example 2: several cohorts, or one contract type

Replace the cohort declarations in Example 1 to price different ages and notionals:

```cpp
const contracts::CohortBook pe{
    {{55.0, 120000.0}, {60.0, 80000.0}}};
const contracts::CohortBook ti{
    {{60.0, 90000.0}, {70.0, 110000.0}}};
const contracts::CohortBook ei{
    {{62.0, 250000.0}}};
const contracts::PetieiPortfolioView portfolio{
    pe.cohorts(), ti.cohorts(), ei.cohorts()};
```

A cohort stores current age and the aggregate notional of living policyholders sharing the same pricing terms and mortality law. Individual notionals can differ: only their sum is needed for this valuation. A book stores aggregate rows; it does not track individual deaths.

To price only EI, use empty books for PE and TI:

```cpp
const contracts::CohortBook pe;
const contracts::CohortBook ti;
const contracts::CohortBook ei{{{60.0, 100000.0}}};
const contracts::PetieiPortfolioView portfolio{
    pe.cohorts(), ti.cohorts(), ei.cohorts()};
```

`PetieiPortfolioView` borrows read-only spans from the books. Keep the books alive while calling `prepare()`. The prepared pricer owns the resulting valuation data and does not retain those spans.

One preparation uses common fund, guarantee and investment parameters for the three contract books. Portfolios with different contract terms need separate preparations.

## Example 3: choose GeoOU or CIR carbon dynamics

The model is selected at compile time. To use GeoOU, replace the pricer alias and carbon-parameter block in Example 1 with:

```cpp
using Pricer = pricing::PetieiMonteCarloPricer4<models::GeoOUExactStepper4>;

models::GeoOUParameters4 carbon;
carbon.long_run_log_mean = current_carbon.array().log().matrix();
carbon.mean_reversion.setConstant(0.5);
carbon.volatility.setConstant(0.2);
```

GeoOU evolves `log(Y)`. Its initial carbon state must be positive, and `long_run_log_mean` is a mean in log space.

For CIR:

```cpp
using Pricer = pricing::PetieiMonteCarloPricer4<models::CIRNVStepper4>;

models::CIRParameters4 carbon;
carbon.long_run_mean = current_carbon;
carbon.mean_reversion.setConstant(0.5);
carbon.volatility.setConstant(0.2);
```

CIR uses the Ninomiya-Victoir split. Its initial state must be nonnegative and each factor must satisfy `kappa * long_run_mean >= volatility^2 / 4`. OU and GeoOU have exact transitions; CIR NV is an approximation. All three still use a time grid for the integrated fund variance and death-benefit quadrature.

## Example 4: a single financial payoff and delta

`MinMaxPayoff` is the underlying financial kernel. It accepts integrated variance for one future payment date and returns an undiscounted conditional expected payoff and fund delta per unit of notional.

Save this complete additional example as `examples/min_max_cashflow.cpp`:

```cpp
#include <carbonfund/pricing/min_max_payoff.hpp>

#include <cmath>
#include <iomanip>
#include <iostream>

int main()
{
    using carbonfund::pricing::MinMaxPayoff;

    const double rate = 0.03;
    const double payment_time = 5.0;
    const double integrated_variance = 0.1;

    const auto prepared = MinMaxPayoff::prepare({
        1.0,                              // Current fund value.
        std::exp(rate * payment_time),    // Risk-free growth.
        1.0,                              // Guarantee at payment time.
        std::exp(0.05 * payment_time)      // Cap at payment time.
    });

    const auto value = MinMaxPayoff::evaluate(
        prepared, integrated_variance);
    const double discount = std::exp(-rate * payment_time);

    std::cout << std::fixed << std::setprecision(6)
              << "Discounted price per unit of notional: "
              << discount * value.expected_payoff << '\n'
              << "Discounted delta per unit of notional: "
              << discount * value.delta << '\n';
}
```

For these inputs, the output is:

```text
Discounted price per unit of notional: 0.973858
Discounted delta per unit of notional: 0.299272
```

Here the variance is supplied directly; the example has no carbon-path simulation or mortality weighting. Integrated variance must be finite and nonnegative. Multiplying its outputs by a policy notional gives a financial cashflow's aggregate price and delta. Example 1 performs the additional carbon and insurance calculations.

Add this target to `CMakeLists.txt`, outside `if(BUILD_TESTING)`:

```cmake
add_executable(example_min_max_cashflow examples/min_max_cashflow.cpp)
target_link_libraries(example_min_max_cashflow PRIVATE CarbonFund::carbonfund)
```

After [reconfiguring your existing build directory](INSTALLATION.md#configure-build-and-test):

```bash
cmake --build build/debug --target example_min_max_cashflow
./build/debug/example_min_max_cashflow
```

On Windows, run `.\build\debug\example_min_max_cashflow.exe`.

## How the valuation works

### Prepare once for the current valuation

`Pricer::prepare()` validates setup inputs and constructs:

1. Carbon transition coefficients for the chosen step size.
2. Allocation coefficients and a factorization of the financial covariance.
3. Discounted mortality weights, including cohort notionals and death quadrature.
4. A financial payoff preparation for each future grid date.
5. The initial allocation-variance contribution and current death-quadrature endpoint.

The current fund value is `contract.fund_value`, corresponding to $X_0$ in the original routine. At each future date $t_i$, the financial kernel stores the forward $F_i=X_0\exp(r t_i)$.

Guarantees grow from their current contractual amounts:

$$
L(t)=L_0\exp(g_L t),\qquad U(t)=U_0\exp(g_U t).
$$

Every future payment requires $0\leq L(t_i)<U(t_i)$. Equal guarantees are allowed at the current endpoint only when they become strictly ordered at every future grid date, as in Example 1. The current death-quadrature endpoint preserves the original delta convention of one at either boundary. The future zero-variance kernel uses half the risk-free growth as its boundary delta convention.

### Simulate carbon and integrate fund variance

For each path, `evaluate()` resets carbon to the supplied current state, then advances it using four standard normal draws per step.

The allocation solves

$$
\left[\gamma\Sigma+
\alpha\,\operatorname{diag}\!\left(\Sigma_{ii}\max(Y_i,0)\right)\right]\pi
=\mu-r\mathbf{1},
$$

where `risk_aversion` is $\gamma$, `carbon_penalty` is $\alpha$ and `covariance` is $\Sigma$. The covariance must be symmetric positive definite; risk aversion must be positive and the penalty nonnegative.

The instantaneous fund variance is $q_i=\pi_i^\top\Sigma\pi_i$. With step size $h$, integrated variance is accumulated by

$$
V_i=V_{i-1}+\frac{h}{2}(q_{i-1}+q_i),\qquad V_0=0.
$$

The simulated carbon state remains the output of its model. Its positive part is used only in the allocation penalty.

### Evaluate conditional benefits and accumulate statistics

The financial payoff at a payment date is

$$
H_t=\min\!\left(\max(X_t,L(t)),U(t)\right).
$$

Conditional on the carbon path, the fund is treated as lognormal with the prepared forward and integrated variance. This assumes the financial noise is independent of the carbon noise. `MinMaxPayoff` integrates that financial noise analytically and calculates both expected payoff and its fund derivative.

Independent Gompertz-Makeham mortality is handled through the prepared survival and death-density weights. Death payments are integrated on the time grid using trapezoidal quadrature. Mortality events are not sampled inside this valuation loop.

The pricer combines maturity and death benefits into one contribution per contract per carbon path, then estimates their means and variances using Welford accumulation. Consequently, only carbon randomness contributes to this conditional estimator's Monte Carlo sampling error.

## Current fund value and later valuations

`fund_value` stays fixed within one prepared valuation. It changes when a new valuation is prepared from an observed fund state.

For example, add the following fragment after Example 1's first valuation. Suppose half a year has elapsed, the observed fund value is 1.04, and the surviving notionals are supplied by the outer policyholder bookkeeping:

```cpp
// This fragment also needs: #include <cmath>
const double elapsed = 0.5;

auto next_contract = contract;
next_contract.fund_value = 1.04;
next_contract.minimum_guarantee *=
    std::exp(contract.minimum_guarantee_growth_rate * elapsed);
next_contract.maximum_guarantee *=
    std::exp(contract.maximum_guarantee_growth_rate * elapsed);
next_contract.time_to_maturity -= elapsed;
next_contract.time_intervals = 54;

const Eigen::Vector4d observed_carbon{1.05, 1.15, 0.85, 1.0};
const contracts::CohortBook surviving_pe{{{60.5, 99000.0}}};
const contracts::CohortBook surviving_ti{{{60.5, 98500.0}}};
const contracts::CohortBook surviving_ei{{{60.5, 99200.0}}};
const contracts::PetieiPortfolioView surviving_portfolio{
    surviving_pe.cohorts(),
    surviving_ti.cohorts(),
    surviving_ei.cohorts()};

const auto next_prepared = Pricer::prepare(
    next_contract, observed_carbon, carbon, allocation,
    surviving_portfolio, mortality);

random::SimulationRng next_rng{20261012};
const auto next_result = Pricer::evaluate(next_prepared, 1000, next_rng);

std::cout << "New EI price: "
          << next_result.endowment_insurance.price.mean << '\n'
          << "New EI delta: "
          << next_result.endowment_insurance.delta.mean << '\n';
```

The guarantee update follows the contract's growth rates; it is independent of the observed fund value. The cohort ages and surviving notionals refer to the new valuation date. The observed values here are illustrative inputs; this snippet does not generate a hedge path or deaths.

The pricer requires strictly positive remaining maturity. Settlement at maturity belongs to the outer contract/hedging layer.

## Random streams and simulation counts

`SimulationRng` owns a Boost `xoshiro256pp` engine and Boost's Ziggurat normal sampler. AS241 is not included in the production project; it was used through a temporary adapter in separate old/new comparison benchmarks.

Create a stream once and pass it by reference. Each `evaluate()` call advances its state. A freshly constructed stream with the same seed reproduces the same draws for the same implementation and inputs.

`time_intervals` controls the grid size $N$. The second argument of `evaluate()` controls the path count $M$. At least two paths are required for an unbiased sample variance.

For example, reuse the same prepared valuation with a larger path count:

```cpp
random::SimulationRng larger_rng{20261011};
const auto larger_result = Pricer::evaluate(prepared, 4000, larger_rng);
```

Preparation is read-only during evaluation. Future parallel callers should each own a separate mutable RNG stream; the current inner path loop is serial.

## Project components

| Component | Responsibility |
|---|---|
| `models/carbon_models_4d.hpp` | Four-factor OU, GeoOU and CIR NV transitions. |
| `random/simulation_rng.hpp` | Owned random engine and normal/uniform/Bernoulli draws. |
| `portfolio/carbon_allocation_4d.hpp` | Carbon-sensitive allocation and instantaneous fund variance. |
| `contracts/cohorts.hpp` | Aggregate living-policyholder rows and borrowed portfolio views. |
| `mortality/gompertz_makeham.hpp` | Hazard, survival probabilities and death density. |
| `pricing/min_max_payoff.hpp` | Conditional expected financial payoff and fund delta. |
| `pricing/petiei_mortality_weights.hpp` | Mortality, discounting, notionals and quadrature weights. |
| `pricing/petiei_monte_carlo.hpp` | Serial PE/TI/EI pricing, deltas and Monte Carlo statistics. |

Header paths in the table are relative to `include/carbonfund/`.

To link your own executable, add it to the root CMake file:

```cmake
add_executable(my_pricer path/to/my_pricer.cpp)
target_link_libraries(my_pricer PRIVATE CarbonFund::carbonfund)
```

`CarbonFund::carbonfund` is an alias of the interface target `carbonfund`. It supplies the project headers, C++20 requirement, Eigen, Boost headers and optional OpenMP dependency.

## Numerical validation

CTest currently contains eight tests: carbon models, RNG, allocation, Gompertz-Makeham mortality, cohorts, PETIEI mortality weights, min/max payoff and the serial PETIEI pricer.

The pricing checks include fixed-draw comparisons against independent equations, delta finite differences with common random draws, sample variance and standard error, deterministic and zero-variance cases, model selection, RNG advancement and input validation.

All eight Debug tests and the portfolio example passed on the reported Ubuntu 26.04.1 / GCC 15.2 / system Boost 1.90.0 setup. Additional checks used GCC 13.3.0, CMake 3.28.3, Eigen 3.4.0 and Boost 1.90.0, including Release pricing tests.

Generated build files belong in `build/`; dependencies remain outside the repository.

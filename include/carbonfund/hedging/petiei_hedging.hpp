#pragma once

#include <carbonfund/contracts/policyholders.hpp>
#include <carbonfund/pricing/petiei_monte_carlo.hpp>
#include <carbonfund/simulation/market_scenarios_4d.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>

namespace carbonfund::hedging {

struct PetieiHedgeValuation {
    double price = 0.0;
    // Already include aggregate notional: PE, TI, EI, respectively.
    std::array<double, 3> deltas{};

    [[nodiscard]] double total_delta() const noexcept
    { return deltas[0] + deltas[1] + deltas[2]; }
};

struct PetieiHedgeOutcome {
    std::uint64_t scenario_id = 0;
    double no_hedge = 0.0;
    double static_hedge = 0.0;
    double dynamic_hedge = 0.0;
    double claims_pv = 0.0;
    double static_trading_gains_pv = 0.0;
    double dynamic_trading_gains_pv = 0.0;
    double fund_terminal = 0.0;
    double initial_price = 0.0;
};

// All monetary state is discounted to time zero. Negative cash can be financed
// at the same rate as positive cash; there are no transaction costs.
class DiscountedHedgeAccount {
public:
    DiscountedHedgeAccount(double premium, double initial_fund, double shares)
        : wealth_(premium), previous_fund_(initial_fund), shares_(shares) {}

    void advance(double discounted_fund, double discounted_claim) noexcept
    {
        const double gain = shares_ * (discounted_fund - previous_fund_);
        gains_ += gain;
        wealth_ += gain - discounted_claim;
        previous_fund_ = discounted_fund;
    }

    // Inject or withdraw capital to match the current discounted liability.
    // Summing these DISCOUNTED additions fixes the legacy mixed-date cost.
    void fund_to(double discounted_liability) noexcept
    {
        funding_ += discounted_liability - wealth_;
        wealth_ = discounted_liability;
    }

    void rebalance(double shares) noexcept { shares_ = shares; }
    [[nodiscard]] double wealth() const noexcept { return wealth_; }
    [[nodiscard]] double funding() const noexcept { return funding_; }
    [[nodiscard]] double trading_gains() const noexcept { return gains_; }

private:
    double wealth_, previous_fund_, shares_;
    double gains_ = 0.0, funding_ = 0.0;
};

namespace detail {

inline void validate_valuation(const PetieiHedgeValuation& value)
{
    if (!std::isfinite(value.price) || !std::isfinite(value.total_delta()) ||
        !std::all_of(value.deltas.begin(), value.deltas.end(),
                     [](double d) { return std::isfinite(d); }))
        throw std::invalid_argument("Hedge valuation must be finite.");
}

inline std::array<double, 3> notionals(const contracts::PetieiPopulationTotals& totals)
{
    return {totals.pure_endowment.notional, totals.term_insurance.notional,
            totals.endowment_insurance.notional};
}

inline void validate_terms(const pricing::PetieiMonteCarloParameters& p, double rate)
{
    const std::array<double, 8> inputs{p.fund_value, p.minimum_guarantee, p.maximum_guarantee,
        p.minimum_guarantee_growth_rate, p.maximum_guarantee_growth_rate,
        p.endowment_death_benefit_fraction, p.time_to_maturity, rate};
    if (!std::all_of(inputs.begin(), inputs.end(), [](double x) { return std::isfinite(x); }) ||
        p.fund_value <= 0.0 || p.time_to_maturity <= 0.0 || p.minimum_guarantee < 0.0 ||
        p.maximum_guarantee < p.minimum_guarantee || p.endowment_death_benefit_fraction < 0.0)
        throw std::invalid_argument("Invalid hedging contract terms.");
    const double lower = p.minimum_guarantee == 0.0 ? 0.0 : p.minimum_guarantee *
        std::exp(p.minimum_guarantee_growth_rate * p.time_to_maturity);
    const double upper = p.maximum_guarantee * std::exp(
        p.maximum_guarantee_growth_rate * p.time_to_maturity);
    const double discount = std::exp(-rate * p.time_to_maturity);
    if (!std::isfinite(lower) || !std::isfinite(upper) || upper < lower ||
        !std::isfinite(discount) || discount <= 0.0)
        throw std::invalid_argument("Guarantees or discount factors are not representable/ordered.");
}

inline double realised_payoff(const pricing::PetieiMonteCarloParameters& p,
                              double time, double fund)
{
    const double lower = p.minimum_guarantee == 0.0 ? 0.0 :
        p.minimum_guarantee * std::exp(p.minimum_guarantee_growth_rate * time);
    const double upper = p.maximum_guarantee * std::exp(p.maximum_guarantee_growth_rate * time);
    // This implements the contractual min/max payoff on the realised fund.
    return fund < lower ? lower : (fund > upper ? upper : fund);
}

} // namespace detail

[[nodiscard]] inline double mortality_adjusted_static_delta(
    const PetieiHedgeValuation& initial,
    const contracts::PetieiPopulationTotals& initial_population,
    const contracts::PetieiPopulationTotals& living_population) noexcept
{
    const auto start = detail::notionals(initial_population);
    const auto living = detail::notionals(living_population);
    double shares = 0.0;
    for (std::size_t i = 0; i < 3; ++i) {
        if (start[i] > 0.0)
            shares += initial.deltas[i] * (living[i] / start[i]);
    }
    return shares;
}

// Preserve the full-horizon time-step until the selected minimum refines the
// residual grid. The minimum counts intervals, not grid points.
[[nodiscard]] inline std::size_t petiei_revaluation_intervals(
    std::size_t full_horizon_intervals, double remaining_time,
    double original_maturity, std::size_t minimum_intervals = 1)
{
    if (full_horizon_intervals == 0 ||
        full_horizon_intervals == std::numeric_limits<std::size_t>::max() ||
        minimum_intervals == 0 || minimum_intervals > full_horizon_intervals)
        throw std::invalid_argument(
            "Revaluation requires 1 <= minimum_intervals <= full_horizon_intervals, "
            "with a representable interval count.");
    if (!std::isfinite(remaining_time) || !std::isfinite(original_maturity) ||
        original_maturity <= 0.0 || remaining_time <= 0.0 ||
        remaining_time > original_maturity)
        throw std::invalid_argument(
            "Revaluation requires 0 < remaining_time <= original_maturity.");
    const long double desired = std::ceil(
        static_cast<long double>(full_horizon_intervals) *
        (static_cast<long double>(remaining_time) / original_maturity));
    const auto scaled = desired >= static_cast<long double>(full_horizon_intervals)
        ? full_horizon_intervals : static_cast<std::size_t>(desired);
    return std::max(minimum_intervals, scaled);
}

// Existing conditioned MC pricer adapted to a valuation at an observed state.
// Guarantees remain anchored to the original contract, independent of fund.
// Keep approximately the initial pricing time-step, with a configurable floor.
template<class CarbonStepper, class PricingRng>
[[nodiscard]] PetieiHedgeValuation price_petiei_at(
    const pricing::PetieiMonteCarloParameters& original,
    double time, double observed_fund, const Eigen::Vector4d& observed_carbon,
    const typename CarbonStepper::Parameters& carbon_parameters,
    const portfolio::CarbonAllocationParameters4& allocation,
    const contracts::PetieiPortfolioView& living,
    const mortality::GompertzMakeham& mortality,
    std::size_t simulations, PricingRng& rng,
    pricing::PetieiSampling sampling = pricing::PetieiSampling::antithetic,
    std::size_t minimum_intervals = 1)
{
    detail::validate_terms(original, allocation.risk_free_rate);
    if (original.time_intervals == 0 ||
        original.time_intervals == std::numeric_limits<std::size_t>::max())
        throw std::invalid_argument("Initial pricing interval count must be positive and representable.");
    if (!std::isfinite(time) || time < 0.0 || time >= original.time_to_maturity)
        throw std::invalid_argument("Revaluation requires 0 <= time < maturity.");
    auto p = original;
    p.fund_value = observed_fund;
    p.time_to_maturity = original.time_to_maturity - time;
    p.minimum_guarantee = original.minimum_guarantee == 0.0 ? 0.0 :
        original.minimum_guarantee * std::exp(original.minimum_guarantee_growth_rate * time);
    p.maximum_guarantee = original.maximum_guarantee *
        std::exp(original.maximum_guarantee_growth_rate * time);
    p.time_intervals = petiei_revaluation_intervals(
        original.time_intervals, p.time_to_maturity,
        original.time_to_maturity, minimum_intervals);
    using Pricer = pricing::PetieiMonteCarloPricer4<CarbonStepper>;
    const auto prepared = Pricer::prepare(p, observed_carbon, carbon_parameters,
                                          allocation, living, mortality);
    const auto value = Pricer::evaluate(prepared, simulations, rng, sampling);
    return {value.pure_endowment.price.mean + value.term_insurance.price.mean +
            value.endowment_insurance.price.mean,
            {value.pure_endowment.delta.mean, value.term_insurance.delta.mean,
             value.endowment_insurance.delta.mean}};
}

// Inject a precomputed physical market path. Draw mortality ONCE for all three
// strategies. value_at(time, fund, carbon, living_view) returns a valuation;
// it sees only the current observation and is never called at maturity.
// initial_value must have been priced at the common initial market/population.
// The population is reset for every run. Use a separate RNG for inner pricing.
template<class MortalityRng, class Valuator>
[[nodiscard]] PetieiHedgeOutcome run_petiei_hedge(
    std::uint64_t scenario_id, const simulation::MarketPath4& path,
    const pricing::PetieiMonteCarloParameters& terms, double risk_free_rate,
    contracts::PolicyholderPopulation& population,
    const mortality::GompertzMakeham& mortality,
    const PetieiHedgeValuation& initial_value,
    MortalityRng& mortality_rng, Valuator&& value_at)
{
    path.validate();
    detail::validate_terms(terms, risk_free_rate);
    detail::validate_valuation(initial_value);
    if (path.times.back() != terms.time_to_maturity || path.fund.front() != terms.fund_value)
        throw std::invalid_argument("Scenario maturity/initial fund must match the initial contract valuation.");
    population.reset();
    const auto initial_population = population.initial_totals();
    DiscountedHedgeAccount no_hedge(initial_value.price, path.fund.front(), 0.0);
    DiscountedHedgeAccount fixed(initial_value.price, path.fund.front(), initial_value.total_delta());
    DiscountedHedgeAccount dynamic(initial_value.price, path.fund.front(), initial_value.total_delta());
    double claims_pv = 0.0;
    for (std::size_t k = 1; k < path.times.size(); ++k) {
        const double time = path.times[k];
        const double discount = std::exp(-risk_free_rate * time);
        const auto deaths = population.advance_to(time, mortality, mortality_rng);
        const auto living = population.living_totals();
        const double payoff = detail::realised_payoff(terms, time, path.fund[k]);
        double claims = payoff * (deaths.term_insurance.notional +
            terms.endowment_death_benefit_fraction * deaths.endowment_insurance.notional);
        const bool maturity = k + 1 == path.times.size();
        if (maturity)
            claims += payoff * (living.pure_endowment.notional + living.endowment_insurance.notional);
        const double discounted_claims = discount * claims;
        claims_pv += discounted_claims;
        const double discounted_fund = discount * path.fund[k];
        no_hedge.advance(discounted_fund, discounted_claims);
        fixed.advance(discounted_fund, discounted_claims);
        dynamic.advance(discounted_fund, discounted_claims);
        if (!maturity) {
            PetieiHedgeValuation current{};
            if (living.pure_endowment.notional > 0.0 || living.term_insurance.notional > 0.0 ||
                living.endowment_insurance.notional > 0.0) {
                current = value_at(time, path.fund[k], path.carbon[k], population.living_portfolio());
                detail::validate_valuation(current);
            }
            dynamic.fund_to(discount * current.price);
            dynamic.rebalance(current.total_delta());
            fixed.rebalance(mortality_adjusted_static_delta(initial_value, initial_population, living));
        }
    }
    // Final settlement includes the last death interval and surviving PE/EI.
    no_hedge.fund_to(0.0);
    fixed.fund_to(0.0);
    dynamic.fund_to(0.0);
    return {scenario_id, no_hedge.funding(), fixed.funding(), dynamic.funding(),
        claims_pv, fixed.trading_gains(), dynamic.trading_gains(), path.fund.back(), initial_value.price};
}

} // namespace carbonfund::hedging

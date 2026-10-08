#pragma once

#include <carbonfund/contracts/cohorts.hpp>
#include <carbonfund/mortality/gompertz_makeham.hpp>

#include <cmath>
#include <cstddef>
#include <limits>
#include <numbers>
#include <span>
#include <stdexcept>
#include <vector>

namespace carbonfund::pricing {

// All weights include aggregate notional and discounting from the current
// valuation date. Death weights also include trapezoidal quadrature.
// The endowment death-benefit percentage is applied separately by the pricer.
struct PetieiMortalityWeights {
    double step_size = 0.0;
    double pure_endowment_maturity = 0.0;
    double endowment_insurance_maturity = 0.0;
    std::vector<double> term_insurance_death;
    std::vector<double> endowment_insurance_death;
};

namespace detail {

inline void validate_mortality_cohorts(
    std::span<const contracts::Cohort> cohorts)
{
    double total = 0.0;
    for (const auto& cohort : cohorts) {
        if (!std::isfinite(cohort.current_age) || cohort.current_age < 0.0 ||
            !std::isfinite(cohort.total_notional) || cohort.total_notional < 0.0) {
            throw std::invalid_argument(
                "Cohort ages and notionals must be finite and nonnegative.");
        }
        total += cohort.total_notional;
        if (!std::isfinite(total)) {
            throw std::invalid_argument("Total cohort notional overflows.");
        }
    }
}

inline double checked_mortality_weight(long double value)
{
    if (!std::isfinite(value) || value > std::numeric_limits<double>::max()) {
        throw std::overflow_error(
            "Mortality weight is outside the double range.");
    }
    return static_cast<double>(value);
}

inline double maturity_mortality_weight(
    std::span<const contracts::Cohort> cohorts,
    const mortality::GompertzMakeham& model,
    double rate,
    double maturity)
{
    long double total = 0.0L;

    if (maturity == 0.0) {
        for (const auto& cohort : cohorts) {
            total += cohort.total_notional;
        }
        return checked_mortality_weight(total);
    }

    for (const auto& cohort : cohorts) {
        if (cohort.total_notional == 0.0) {
            continue;
        }
        total += std::exp(
            std::log(static_cast<long double>(cohort.total_notional)) +
            model.log_survival_probability(cohort.current_age, maturity) -
            static_cast<long double>(rate) * maturity);
    }
    return checked_mortality_weight(total);
}

inline void fill_death_mortality_weights(
    std::span<const contracts::Cohort> cohorts,
    const mortality::GompertzMakeham& model,
    double rate,
    double maturity,
    double step_size,
    std::vector<double>& weights)
{
    const std::size_t intervals = weights.size() - 1;
    const long double log_step =
        std::log(static_cast<long double>(step_size));

    for (const auto& cohort : cohorts) {
        if (cohort.total_notional == 0.0) {
            continue;
        }

        const long double log_notional =
            std::log(static_cast<long double>(cohort.total_notional));

        for (std::size_t i = 0; i <= intervals; ++i) {
            // Avoid accumulated time-step error and use the exact maturity.
            const double time = i == intervals
                ? maturity : static_cast<double>(i) * step_size;

            const long double log_quadrature = log_step -
                ((i == 0 || i == intervals)
                    ? std::numbers::ln2_v<long double> : 0.0L);

            const long double contribution = std::exp(
                log_notional +
                model.log_death_density(cohort.current_age, time) -
                static_cast<long double>(rate) * time +
                log_quadrature);

            weights[i] = checked_mortality_weight(
                static_cast<long double>(weights[i]) + contribution);
        }
    }
}

} // namespace detail

// Prepare once per valuation, before simulating financial paths.
// Ages refer to this valuation date and maturity is the remaining time in years.
// The rate is continuously compounded; finite negative rates are allowed.
[[nodiscard]] inline PetieiMortalityWeights prepare_petiei_mortality_weights(
    const contracts::PetieiPortfolioView& portfolio,
    const mortality::GompertzMakeham& model,
    double risk_free_rate,
    double time_to_maturity,
    std::size_t time_intervals)
{
    if (!std::isfinite(risk_free_rate) || !std::isfinite(time_to_maturity) ||
        time_to_maturity < 0.0 || time_intervals == 0 ||
        time_intervals == std::numeric_limits<std::size_t>::max()) {
        throw std::invalid_argument(
            "Rate must be finite, maturity finite and nonnegative, "
            "and intervals positive.");
    }

    detail::validate_mortality_cohorts(portfolio.pure_endowment);
    detail::validate_mortality_cohorts(portfolio.term_insurance);
    detail::validate_mortality_cohorts(portfolio.endowment_insurance);

    PetieiMortalityWeights result;
    result.step_size =
        time_to_maturity / static_cast<double>(time_intervals);

    if (time_to_maturity > 0.0 && result.step_size == 0.0) {
        throw std::invalid_argument("Time step underflows to zero.");
    }

    result.pure_endowment_maturity = detail::maturity_mortality_weight(
        portfolio.pure_endowment, model, risk_free_rate, time_to_maturity);

    result.endowment_insurance_maturity = detail::maturity_mortality_weight(
        portfolio.endowment_insurance, model,
        risk_free_rate, time_to_maturity);

    result.term_insurance_death.resize(time_intervals + 1, 0.0);
    result.endowment_insurance_death.resize(time_intervals + 1, 0.0);

    // At maturity only PE/EI survival benefits remain;
    // there is no death interval.
    if (time_to_maturity == 0.0) {
        return result;
    }

    detail::fill_death_mortality_weights(
        portfolio.term_insurance, model, risk_free_rate,
        time_to_maturity, result.step_size, result.term_insurance_death);

    detail::fill_death_mortality_weights(
        portfolio.endowment_insurance, model, risk_free_rate,
        time_to_maturity, result.step_size,
        result.endowment_insurance_death);

    return result;
}

} // namespace carbonfund::pricing
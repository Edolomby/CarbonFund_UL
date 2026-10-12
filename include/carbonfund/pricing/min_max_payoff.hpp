#pragma once

#include <algorithm>
#include <cmath>
#include <limits>
#include <numbers>
#include <stdexcept>

namespace carbonfund::pricing {

struct PayoffDelta {
    double expected_payoff = 0.0;
    double delta = 0.0;
};

// Inputs for one future payment date, per unit of policy notional.
// risk_free_growth = exp(r*t), where t is measured from the current valuation.
// Guarantees are fixed when differentiating with respect to fund_value.
struct MinMaxPayoffParameters {
    double fund_value = 1.0;
    double risk_free_growth = 1.0;
    double minimum_guarantee = 0.0;
    double maximum_guarantee = 1.0;
};

namespace detail {

inline double payoff_log_ratio(double numerator, double denominator)
{
    const double relative_difference =
        (numerator - denominator) / denominator;

    if (std::abs(relative_difference) < 0.5) {
        return std::log1p(relative_difference);
    }

    return std::log(numerator) - std::log(denominator);
}

inline double payoff_normal_cdf(double x) noexcept
{
    constexpr double inverse_sqrt_two =
        std::numbers::sqrt2_v<double> / 2.0;

    return 0.5 * std::erfc(-x * inverse_sqrt_two);
}

inline double payoff_normal_survival(double x) noexcept
{
    constexpr double inverse_sqrt_two =
        std::numbers::sqrt2_v<double> / 2.0;

    return 0.5 * std::erfc(x * inverse_sqrt_two);
}

inline double payoff_normal_interval(
    double lower,
    double upper,
    double width) noexcept
{
    const double midpoint = 0.5 * (lower + upper);

    // Integrate the normal density over a narrow interval instead of
    // subtracting almost equal CDFs. width is computed independently of
    // upper-lower so that nearby guarantee values retain their precision.
    if (width < 1e-2 / (1.0 + std::abs(midpoint))) {
        constexpr double inverse_sqrt_two_pi =
            std::numbers::inv_sqrtpi_v<double> /
            std::numbers::sqrt2_v<double>;

        const double scaled_midpoint = midpoint * width;
        const double width_squared = width * width;
        const double scaled_squared = scaled_midpoint * scaled_midpoint;

        const double correction =
            (scaled_squared - width_squared) / 24.0 +
            (scaled_squared * scaled_squared -
             6.0 * scaled_squared * width_squared +
             3.0 * width_squared * width_squared) / 1920.0;

        return inverse_sqrt_two_pi *
               std::exp(-0.5 * midpoint * midpoint) *
               width * (1.0 + correction);
    }

    // Use upper tails when both CDFs would be close to one.
    return lower >= 0.0
        ? payoff_normal_survival(lower) -
          payoff_normal_survival(upper)
        : payoff_normal_cdf(upper) -
          payoff_normal_cdf(lower);
}

} // namespace detail

struct MinMaxPayoff {
    using Parameters = MinMaxPayoffParameters;

    class Prepared {
        friend struct MinMaxPayoff;

        double forward_;
        double growth_;
        double minimum_;
        double maximum_;
        double log_minimum_forward_ratio_;
        double log_maximum_forward_ratio_;
        double log_guarantee_ratio_;

        Prepared(
            double forward,
            double growth,
            double minimum,
            double maximum,
            double log_minimum,
            double log_maximum,
            double log_guarantees)
            : forward_(forward),
              growth_(growth),
              minimum_(minimum),
              maximum_(maximum),
              log_minimum_forward_ratio_(log_minimum),
              log_maximum_forward_ratio_(log_maximum),
              log_guarantee_ratio_(log_guarantees)
        {
        }
    };

    [[nodiscard]] static Prepared prepare(const Parameters& parameters)
    {
        const double x = parameters.fund_value;
        const double growth = parameters.risk_free_growth;
        const double minimum = parameters.minimum_guarantee;
        const double maximum = parameters.maximum_guarantee;

        if (!std::isfinite(x) || !std::isfinite(growth) ||
            !std::isfinite(minimum) || !std::isfinite(maximum) ||
            x <= 0.0 || growth <= 0.0 ||
            minimum < 0.0 || maximum <= minimum) {
            throw std::invalid_argument(
                "Fund and growth must be positive and finite; "
                "guarantees finite with 0 <= min < max.");
        }

        const double forward = x * growth;

        if (!std::isfinite(forward) || forward == 0.0) {
            throw std::invalid_argument(
                "Fund forward overflows or underflows to zero.");
        }


        const double log_minimum = minimum == 0.0
            ? -std::numeric_limits<double>::infinity()
            : detail::payoff_log_ratio(minimum, forward);

        const double log_maximum =
            detail::payoff_log_ratio(maximum, forward);

        const double log_guarantees = minimum == 0.0
            ? std::numeric_limits<double>::infinity()
            : detail::payoff_log_ratio(maximum, minimum);

        return Prepared{
            forward, growth, minimum, maximum,
            log_minimum, log_maximum, log_guarantees};
    }

    // Undiscounted conditional expected payoff and its fund delta.
    // Caller precondition: integrated_variance is finite and nonnegative.
    // Allocation-free; prepared remains unchanged.
    [[nodiscard]] static PayoffDelta evaluate(
        const Prepared& prepared,
        double integrated_variance) noexcept
    {
        if (integrated_variance == 0.0) {
            if (prepared.forward_ < prepared.minimum_) {
                return {prepared.minimum_, 0.0};
            }

            if (prepared.forward_ > prepared.maximum_) {
                return {prepared.maximum_, 0.0};
            }

            const bool boundary =
                prepared.forward_ == prepared.minimum_ ||
                prepared.forward_ == prepared.maximum_;

            return {
                prepared.forward_,
                prepared.growth_ * (boundary ? 0.5 : 1.0)};
        }

        const double half_deviation = 0.5 * std::sqrt(integrated_variance);
        const double inv_deviation = 0.5 / half_deviation;

        const double lower_scaled =
            prepared.log_minimum_forward_ratio_ * inv_deviation;
        const double upper_scaled =
            prepared.log_maximum_forward_ratio_ * inv_deviation;

        const double a_lower = lower_scaled - half_deviation;
        const double a_upper = upper_scaled - half_deviation;
        const double b_lower = lower_scaled + half_deviation;
        const double b_upper = upper_scaled + half_deviation;

        const double probability =
                    prepared.minimum_ == 0.0
                        ? detail::payoff_normal_cdf(a_upper)
                        : detail::payoff_normal_interval(
                            a_lower,
                            a_upper,
                            prepared.log_guarantee_ratio_ * inv_deviation);

        const double payoff =
            prepared.minimum_ * detail::payoff_normal_cdf(b_lower) +
            prepared.maximum_ * detail::payoff_normal_survival(b_upper) +
            prepared.forward_ * probability;

        // The expected payoff lies between the guarantees; clamp roundoff.
        return {payoff, prepared.growth_ * probability};
    }
};

} // namespace carbonfund::pricing
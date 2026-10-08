#pragma once

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace carbonfund::mortality {

// Ages and horizons are in years; intensities and growth rate are per year.
struct GompertzMakehamParameters {
    double makeham_intensity = 0.0;     // phi >= 0
    double gompertz_location_age = 0.0; // m
    double gompertz_growth_rate = 0.0;  // Set b > 0; reciprocal of the old scale
};

// Deterministic mortality law, conditional on being alive at the supplied age.
// Compute probabilities during setup, outside individual sampling loops.
class GompertzMakeham {
public:
    explicit GompertzMakeham(const GompertzMakehamParameters& parameters)
        : parameters_(parameters)
    {
        if (!std::isfinite(parameters_.makeham_intensity) ||
            !std::isfinite(parameters_.gompertz_location_age) ||
            !std::isfinite(parameters_.gompertz_growth_rate) ||
            parameters_.makeham_intensity < 0.0 ||
            parameters_.gompertz_growth_rate <= 0.0) {
            throw std::invalid_argument(
              "Mortality parameters must be finite, with phi >= 0 and b > 0.");
        }
    }

    // lambda(a) = phi + b * exp(b * (a - m)).
    [[nodiscard]] double hazard(double age) const
    {
        require_nonnegative_finite(age);

        const long double b = parameters_.gompertz_growth_rate;
        const long double log_gompertz_hazard =
            std::log(b) + b * (static_cast<long double>(age) -
                 								parameters_.gompertz_location_age);

        return to_double(
            parameters_.makeham_intensity +
            std::exp(log_gompertz_hazard));
    }

    // H_a(t) = phi*t + exp(b*(a-m)) * (exp(b*t) - 1).
    [[nodiscard]] double cumulative_hazard(double age, double horizon) const
    {
        require_nonnegative_finite(age);
        require_nonnegative_finite(horizon);

        if (horizon == 0.0) {
            return 0.0;
        }

        const long double b = parameters_.gompertz_growth_rate;
        const long double t = horizon;
        const long double bt = b * t;

        // Equivalent Gompertz term:
        // exp(b*(a+t-m)) * (1 - exp(-b*t)).
        //
        // Evaluate its logarithm to avoid multiplying zero by infinity.
        // The bt==0 branch covers product underflow on platforms where
        // long double has the same range as double.
        const long double log_interval = bt == 0.0L
            ? std::log(b) + std::log(t)
            : std::log(-std::expm1(-bt));

        const long double log_gompertz_integral =
            b * (static_cast<long double>(age) + t -
                 parameters_.gompertz_location_age) +
            log_interval;

        return to_double(
            parameters_.makeham_intensity * t +
            std::exp(log_gompertz_integral));
    }

    [[nodiscard]] double log_survival_probability(
        double age,
        double horizon) const
    {
        return -cumulative_hazard(age, horizon);
    }

    [[nodiscard]] double survival_probability(
        double age,
        double horizon) const
    {
        return std::exp(log_survival_probability(age, horizon));
    }

    [[nodiscard]] double death_probability(
        double age,
        double horizon) const
    {
        // Accurate even when survival rounds to exactly one.
        return -std::expm1(log_survival_probability(age, horizon));
    }

		// log(f_a(t)), where f_a(t) = lambda(a+t) * S_a(t).
		// This is a density per year, not an interval death probability.
		[[nodiscard]] double log_death_density(double age, double horizon) const
		{
				const double integrated_hazard = cumulative_hazard(age, horizon);
				if (std::isinf(integrated_hazard)) {
						return -std::numeric_limits<double>::infinity();
				}

				const long double b = parameters_.gompertz_growth_rate;
				const long double log_gompertz_hazard =
						std::log(b) + b * (static_cast<long double>(age) + horizon -
															parameters_.gompertz_location_age);

				long double log_hazard = log_gompertz_hazard;
				if (parameters_.makeham_intensity > 0.0) {
						const long double log_makeham =
								std::log(static_cast<long double>(parameters_.makeham_intensity));
						const long double larger = std::max(log_makeham, log_gompertz_hazard);
						const long double smaller = std::min(log_makeham, log_gompertz_hazard);
						log_hazard = larger + std::log1p(std::exp(smaller - larger));
				}

				return to_double(log_hazard - integrated_hazard);
		}



private:
    GompertzMakehamParameters parameters_;

    static void require_nonnegative_finite(double value)
    {
        if (!std::isfinite(value) || value < 0.0) {
            throw std::invalid_argument(
                "Age and horizon must be finite and nonnegative.");
        }
    }

    // Saturate values outside the double range, including negative log densities.
		static double to_double(long double value) noexcept
		{
				if (value > std::numeric_limits<double>::max()) {
						return std::numeric_limits<double>::infinity();
				}
				if (value < -std::numeric_limits<double>::max()) {
						return -std::numeric_limits<double>::infinity();
				}
				return static_cast<double>(value);
		}
};

} // namespace carbonfund::mortality
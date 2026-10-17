#pragma once

#include <carbonfund/models/carbon_models_4d.hpp>
#include <carbonfund/portfolio/carbon_allocation_4d.hpp>

#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

namespace carbonfund::simulation {

// One realised market path, including the initial observation at time zero.
// This owning container can also be filled by an external scenario source.
struct MarketPath4 {
    std::vector<double> times;
    std::vector<double> fund;
    std::vector<Eigen::Vector4d> carbon;

    void validate() const
    {
        if (times.size() < 2 || fund.size() != times.size() ||
            carbon.size() != times.size() || times.front() != 0.0) {
            throw std::invalid_argument("Market path needs matching arrays and a grid starting at zero.");
        }
        for (std::size_t k = 0; k < times.size(); ++k) {
            if (!std::isfinite(times[k]) || !std::isfinite(fund[k]) ||
                fund[k] <= 0.0 || !carbon[k].allFinite() ||
                (k > 0 && times[k] <= times[k - 1])) {
                throw std::invalid_argument("Market path has invalid times, fund values or carbon states.");
            }
        }
    }
};

// Physical-measure scenarios with independent carbon and financial streams.
// Trapezoidal log-drift/variance integration matches the legacy fund scheme.
template<class CarbonStepper>
struct MarketScenarioGenerator4 {
    class Prepared {
        friend struct MarketScenarioGenerator4;
        double initial_fund_, rate_, step_;
        Eigen::Vector4d initial_carbon_, excess_returns_;
        typename CarbonStepper::Prepared transition_;
        portfolio::CarbonAllocation4::Prepared allocation_;
        std::vector<double> times_;

        Prepared(double fund, double rate, double step,
                 const Eigen::Vector4d& carbon, const Eigen::Vector4d& excess,
                 typename CarbonStepper::Prepared transition,
                 portfolio::CarbonAllocation4::Prepared allocation,
                 std::vector<double> times)
            : initial_fund_(fund), rate_(rate), step_(step),
              initial_carbon_(carbon), excess_returns_(excess),
              transition_(std::move(transition)), allocation_(std::move(allocation)),
              times_(std::move(times)) {}
    };

    [[nodiscard]] static Prepared prepare(
        double initial_fund, const Eigen::Vector4d& initial_carbon,
        const typename CarbonStepper::Parameters& carbon_parameters,
        const portfolio::CarbonAllocationParameters4& allocation_parameters,
        double maturity, std::size_t intervals)
    {
        if (!std::isfinite(initial_fund) || initial_fund <= 0.0 ||
            !initial_carbon.allFinite() || !std::isfinite(maturity) || maturity <= 0.0 ||
            intervals == 0 || intervals == std::numeric_limits<std::size_t>::max()) {
            throw std::invalid_argument("Market generation needs finite states, positive fund/maturity/intervals.");
        }
        if constexpr (std::is_same_v<CarbonStepper, models::GeoOUExactStepper4>) {
            if ((initial_carbon.array() <= 0.0).any())
                throw std::invalid_argument("GeoOU initial carbon must be positive.");
        } else if constexpr (std::is_same_v<CarbonStepper, models::CIRNVStepper4>) {
            if ((initial_carbon.array() < 0.0).any())
                throw std::invalid_argument("CIR initial carbon must be nonnegative.");
        }
        const double step = maturity / static_cast<double>(intervals);
        auto transition = CarbonStepper::prepare(step, carbon_parameters);
        auto allocation = portfolio::CarbonAllocation4::prepare(allocation_parameters);
        std::vector<double> times(intervals + 1);
        for (std::size_t k = 0; k <= intervals; ++k) {
            times[k] = k == intervals ? maturity : static_cast<double>(k) * step;
            if (k > 0 && times[k] <= times[k - 1])
                throw std::invalid_argument("Market time grid is not representable.");
        }
        return Prepared{initial_fund, allocation_parameters.risk_free_rate, step,
            initial_carbon,
            (allocation_parameters.expected_returns.array() - allocation_parameters.risk_free_rate).matrix(),
            std::move(transition), std::move(allocation), std::move(times)};
    }

    // Reuse the same output buffers between scenarios. No full scenario cube.
    // Arithmetic representability is a sampling-loop precondition; the hedger
    // validates the completed path before using it.
    template<class CarbonRng, class FinancialRng>
    static void generate(const Prepared& prepared, CarbonRng& carbon_rng,
                         FinancialRng& financial_rng, MarketPath4& output)
    {
        output.times = prepared.times_;
        output.fund.resize(output.times.size());
        output.carbon.resize(output.times.size());
        output.fund[0] = prepared.initial_fund_;
        output.carbon[0] = prepared.initial_carbon_;
        portfolio::CarbonAllocation4::Workspace workspace;
        portfolio::CarbonAllocation4::evaluate(output.carbon[0], prepared.allocation_, workspace);
        double previous_variance = workspace.instantaneous_variance;
        double previous_excess = workspace.weights.dot(prepared.excess_returns_);
        Eigen::Vector4d normals;
        for (std::size_t k = 1; k < output.times.size(); ++k) {
            output.carbon[k] = output.carbon[k - 1];
            carbon_rng.fill_std_normals(normals);
            CarbonStepper::step(output.carbon[k], prepared.transition_, normals);
            portfolio::CarbonAllocation4::evaluate(output.carbon[k], prepared.allocation_, workspace);
            const double variance = workspace.instantaneous_variance;
            const double excess = workspace.weights.dot(prepared.excess_returns_);
            const double integrated_variance = 0.5 * prepared.step_ * (previous_variance + variance);
            const double integrated_excess = 0.5 * prepared.step_ * (previous_excess + excess);
            output.fund[k] = output.fund[k - 1] * std::exp(
                prepared.rate_ * prepared.step_ + integrated_excess - 0.5 * integrated_variance +
                std::sqrt(integrated_variance) * financial_rng.std_normal());
            previous_variance = variance;
            previous_excess = excess;
        }
    }
};

} // namespace carbonfund::simulation

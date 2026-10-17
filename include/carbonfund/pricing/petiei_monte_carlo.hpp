#pragma once

#include <carbonfund/models/carbon_models_4d.hpp>
#include <carbonfund/portfolio/carbon_allocation_4d.hpp>
#include <carbonfund/pricing/min_max_payoff.hpp>
#include <carbonfund/pricing/petiei_mortality_weights.hpp>

#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

namespace carbonfund::pricing {

// All values refer to the current valuation date. Guarantees are contractual
// amounts, independent of fund_value. The risk-free rate comes from the
// CarbonAllocationParameters4 supplied to prepare(), for both pricing and
// allocation. Only a strictly positive remaining maturity is priced here.
struct PetieiMonteCarloParameters {
    double fund_value = 1.0;
    double minimum_guarantee = 0.0;
    double maximum_guarantee = 1.0;
    double minimum_guarantee_growth_rate = 0.0;
    double maximum_guarantee_growth_rate = 0.0;
    double endowment_death_benefit_fraction = 1.0;
    double time_to_maturity = 1.0;
    std::size_t time_intervals = 1;
};

struct MonteCarloEstimate {
    double mean = 0.0;
    // Unbiased variance of one independent observation, not of its mean:
    // a path contribution in independent mode, a pair mean in antithetic mode.
    double sample_variance = 0.0;
    double standard_error = 0.0;
};

struct PetieiContractEstimate {
    MonteCarloEstimate price;
    MonteCarloEstimate delta;
};

struct PetieiMonteCarloResult {
    // Actual carbon trajectories evaluated. Odd antithetic requests round up.
    std::size_t simulations = 0;
    PetieiContractEstimate pure_endowment;
    PetieiContractEstimate term_insurance;
    PetieiContractEstimate endowment_insurance;
    // Number of observations used for variance and standard-error estimation.
    std::size_t independent_samples = 0;
};

enum class PetieiSampling {
    independent,
    antithetic
};

enum class PetieiEstimator { standard, conditioned };
enum class PetieiContract { all, pure_endowment, term_insurance, endowment_insurance };

namespace detail {

// Welford accumulation avoids subtracting almost equal sums of squares.
// One inverse sample count is shared by all six price/delta statistics.
struct PetieiRunningMoments {
    PayoffDelta mean;
    PayoffDelta sum_squared_deviations;

    void add(const PayoffDelta& sample, double inverse_count) noexcept
    {
        const double price_difference = sample.expected_payoff - mean.expected_payoff;
        const double delta_difference = sample.delta - mean.delta;

        mean.expected_payoff += price_difference * inverse_count;
        mean.delta += delta_difference * inverse_count;

        sum_squared_deviations.expected_payoff +=
            price_difference * (sample.expected_payoff - mean.expected_payoff);
        sum_squared_deviations.delta +=
            delta_difference * (sample.delta - mean.delta);
    }

    [[nodiscard]] PetieiContractEstimate finish(
        double inverse_degrees_of_freedom,
        double inverse_count) const noexcept
    {
        const double price_variance =
            sum_squared_deviations.expected_payoff * inverse_degrees_of_freedom;
        const double delta_variance =
            sum_squared_deviations.delta * inverse_degrees_of_freedom;

        return {
            {mean.expected_payoff, price_variance,
             std::sqrt(price_variance * inverse_count)},
            {mean.delta, delta_variance,
             std::sqrt(delta_variance * inverse_count)}};
    }
};

} // namespace detail

// Serial standard/conditional pricing of PE, TI and EI. Mortality is
// integrated analytically in both estimators. The conditioned estimator also
// integrates the financial Gaussian. CarbonStepper is selected at compile time.
template<class CarbonStepper>
struct PetieiMonteCarloPricer4 {
    using Parameters = PetieiMonteCarloParameters;

    class Prepared {
        friend struct PetieiMonteCarloPricer4;

        typename CarbonStepper::Prepared carbon_transition_;
        portfolio::CarbonAllocation4::Prepared allocation_;
        Eigen::Vector4d initial_carbon_state_;
        PetieiMortalityWeights mortality_weights_;
        // Index zero corresponds to the first future payment date, t = h.
        std::vector<MinMaxPayoff::Prepared> future_payoffs_;
        PayoffDelta initial_term_insurance_;
        PayoffDelta initial_endowment_death_;
        double half_step_;
        double initial_half_variance_;
        double endowment_death_benefit_fraction_;

        Prepared(
            typename CarbonStepper::Prepared carbon_transition,
            portfolio::CarbonAllocation4::Prepared allocation,
            const Eigen::Vector4d& initial_carbon_state,
            PetieiMortalityWeights mortality_weights,
            std::vector<MinMaxPayoff::Prepared> future_payoffs,
            const PayoffDelta& initial_payoff,
            double initial_half_variance,
            double endowment_death_benefit_fraction)
            : carbon_transition_(std::move(carbon_transition)),
              allocation_(std::move(allocation)),
              initial_carbon_state_(initial_carbon_state),
              mortality_weights_(std::move(mortality_weights)),
              future_payoffs_(std::move(future_payoffs)),
              initial_term_insurance_{
                  initial_payoff.expected_payoff * mortality_weights_.term_insurance_death[0],
                  initial_payoff.delta * mortality_weights_.term_insurance_death[0]},
              initial_endowment_death_{
                  initial_payoff.expected_payoff * mortality_weights_.endowment_insurance_death[0],
                  initial_payoff.delta * mortality_weights_.endowment_insurance_death[0]},
              half_step_(0.5 * mortality_weights_.step_size),
              initial_half_variance_(initial_half_variance),
              endowment_death_benefit_fraction_(endowment_death_benefit_fraction)
        {
        }
    };

    // Prepare once per valuation. The resulting object owns its data; cohort
    // views are not retained. Rebuild it when the observed fund/carbon state,
    // remaining maturity, guarantees or living cohorts change during hedging.
    [[nodiscard]] static Prepared prepare(
        const Parameters& parameters,
        const Eigen::Vector4d& initial_carbon_state,
        const typename CarbonStepper::Parameters& carbon_parameters,
        const portfolio::CarbonAllocationParameters4& allocation_parameters,
        const contracts::PetieiPortfolioView& portfolio_view,
        const mortality::GompertzMakeham& mortality_model)
    {
        if (!std::isfinite(parameters.fund_value) || parameters.fund_value <= 0.0 ||
            !std::isfinite(parameters.minimum_guarantee) ||
            !std::isfinite(parameters.maximum_guarantee) ||
            parameters.minimum_guarantee < 0.0 ||
            parameters.maximum_guarantee < parameters.minimum_guarantee ||
            !std::isfinite(parameters.minimum_guarantee_growth_rate) ||
            !std::isfinite(parameters.maximum_guarantee_growth_rate) ||
            !std::isfinite(parameters.endowment_death_benefit_fraction) ||
            parameters.endowment_death_benefit_fraction < 0.0 ||
            !std::isfinite(parameters.time_to_maturity) ||
            parameters.time_to_maturity <= 0.0 ||
            parameters.time_intervals == 0 ||
            parameters.time_intervals == std::numeric_limits<std::size_t>::max()) {
            throw std::invalid_argument(
                "Pricing requires a positive finite fund and maturity, ordered "
                "nonnegative guarantees, finite growth rates, a nonnegative "
                "death-benefit fraction and positive intervals.");
        }

        if (!initial_carbon_state.allFinite()) {
            throw std::invalid_argument("Initial carbon state must be finite.");
        }
        if constexpr (std::is_same_v<CarbonStepper, models::GeoOUExactStepper4>) {
            if ((initial_carbon_state.array() <= 0.0).any()) {
                throw std::invalid_argument("GeoOU requires a positive initial carbon state.");
            }
        } else if constexpr (std::is_same_v<CarbonStepper, models::CIRNVStepper4>) {
            if ((initial_carbon_state.array() < 0.0).any()) {
                throw std::invalid_argument("CIR requires a nonnegative initial carbon state.");
            }
        }

        const auto allocation = portfolio::CarbonAllocation4::prepare(allocation_parameters);
        auto mortality_weights = prepare_petiei_mortality_weights(
            portfolio_view, mortality_model, allocation_parameters.risk_free_rate,
            parameters.time_to_maturity, parameters.time_intervals);
        const double step = mortality_weights.step_size;
        const auto carbon_transition = CarbonStepper::prepare(step, carbon_parameters);

        std::vector<MinMaxPayoff::Prepared> future_payoffs;
        future_payoffs.reserve(parameters.time_intervals);

        for (std::size_t i = 1; i <= parameters.time_intervals; ++i) {
            const double time = i == parameters.time_intervals
                ? parameters.time_to_maturity : static_cast<double>(i) * step;
            const double minimum = parameters.minimum_guarantee == 0.0
                ? 0.0
                : parameters.minimum_guarantee *
                  std::exp(parameters.minimum_guarantee_growth_rate * time);
            const double maximum = parameters.maximum_guarantee *
                std::exp(parameters.maximum_guarantee_growth_rate * time);

            // The kernel enforces minimum < maximum at every future date.
            future_payoffs.push_back(MinMaxPayoff::prepare({
                parameters.fund_value,
                std::exp(allocation_parameters.risk_free_rate * time),
                minimum, maximum}));
        }

        // Guarantees are fixed in fund delta. At t=0 an equal floor and cap
        // give a constant immediate payoff, with delta zero. Use the symmetric
        // derivative at a single kink, consistently with the payoff kernel.
        PayoffDelta initial_payoff{parameters.fund_value, 1.0};
        if (parameters.minimum_guarantee == parameters.maximum_guarantee) {
            initial_payoff = {parameters.minimum_guarantee, 0.0};
        } else if (parameters.fund_value < parameters.minimum_guarantee) {
            initial_payoff = {parameters.minimum_guarantee, 0.0};
        } else if (parameters.fund_value > parameters.maximum_guarantee) {
            initial_payoff = {parameters.maximum_guarantee, 0.0};
        } else if (parameters.fund_value == parameters.minimum_guarantee ||
                   parameters.fund_value == parameters.maximum_guarantee) {
            initial_payoff.delta = 0.5;
        }

        portfolio::CarbonAllocation4::Workspace initial_allocation;
        portfolio::CarbonAllocation4::evaluate(
            initial_carbon_state, allocation, initial_allocation);
        const double initial_half_variance =
            0.5 * step * initial_allocation.instantaneous_variance;
        if (!std::isfinite(initial_half_variance)) {
            throw std::overflow_error("Initial integrated-variance contribution overflows.");
        }

        return Prepared{
            carbon_transition, allocation, initial_carbon_state,
            std::move(mortality_weights), std::move(future_payoffs),
            initial_payoff, initial_half_variance,
            parameters.endowment_death_benefit_fraction};
    }

private:
    struct PathState {
        Eigen::Vector4d carbon;
        double integrated_variance;
        double previous_half_variance;
        PayoffDelta term_insurance;
        PayoffDelta endowment_death;
        PayoffDelta payoff;
        double log_financial_factor = 0.0;
    };

    struct PathContributions {
        PayoffDelta pure_endowment;
        PayoffDelta term_insurance;
        PayoffDelta endowment_insurance;
    };

    struct NoFinancialStream {
        double std_normal() noexcept { return 0.0; }
    };

    template<PetieiContract Contract>
    static constexpr bool has_pe = Contract == PetieiContract::all ||
        Contract == PetieiContract::pure_endowment;
    template<PetieiContract Contract>
    static constexpr bool has_ti = Contract == PetieiContract::all ||
        Contract == PetieiContract::term_insurance;
    template<PetieiContract Contract>
    static constexpr bool has_ei = Contract == PetieiContract::all ||
        Contract == PetieiContract::endowment_insurance;

    [[nodiscard]] static PathState initial_path(const Prepared& prepared)
    {
        return {prepared.initial_carbon_state_, 0.0, prepared.initial_half_variance_,
                prepared.initial_term_insurance_, prepared.initial_endowment_death_, {}};
    }

    static double advance_carbon(
        PathState& path, const Prepared& prepared,
        const Eigen::Vector4d& normals,
        portfolio::CarbonAllocation4::Workspace& workspace)
    {
        CarbonStepper::step(path.carbon, prepared.carbon_transition_, normals);
        portfolio::CarbonAllocation4::evaluate(path.carbon, prepared.allocation_, workspace);
        const double current_half = prepared.half_step_ * workspace.instantaneous_variance;
        const double increment = path.previous_half_variance + current_half;
        // Keep the existing summation order for the conditional estimator.
        path.integrated_variance += path.previous_half_variance;
        path.integrated_variance += current_half;
        path.previous_half_variance = current_half;
        return increment;
    }

    template<bool Standard>
    static void evaluate_date(
        PathState& path, const MinMaxPayoff::Prepared& payoff,
        double increment, double financial_normal)
    {
        if constexpr (Standard) {
            path.log_financial_factor += -0.5 * increment +
                std::sqrt(increment) * financial_normal;
            path.payoff = MinMaxPayoff::evaluate_sample(payoff, path.log_financial_factor);
        } else {
            path.payoff = MinMaxPayoff::evaluate(payoff, path.integrated_variance);
        }
    }

    template<PetieiContract Contract>
    static void accumulate_death(PathState& path, const Prepared& prepared, std::size_t date)
    {
        const auto& weights = prepared.mortality_weights_;
        if constexpr (has_ti<Contract>) {
            path.term_insurance.expected_payoff +=
                path.payoff.expected_payoff * weights.term_insurance_death[date + 1];
            path.term_insurance.delta += path.payoff.delta * weights.term_insurance_death[date + 1];
        }
        if constexpr (has_ei<Contract>) {
            path.endowment_death.expected_payoff +=
                path.payoff.expected_payoff * weights.endowment_insurance_death[date + 1];
            path.endowment_death.delta += path.payoff.delta * weights.endowment_insurance_death[date + 1];
        }
    }

    template<PetieiContract Contract>
    [[nodiscard]] static PathContributions finish_path(const PathState& path, const Prepared& prepared)
    {
        PathContributions result{};
        const auto& weights = prepared.mortality_weights_;
        if constexpr (has_pe<Contract>) {
            result.pure_endowment = {
                path.payoff.expected_payoff * weights.pure_endowment_maturity,
                path.payoff.delta * weights.pure_endowment_maturity};
        }
        if constexpr (has_ti<Contract>) result.term_insurance = path.term_insurance;
        if constexpr (has_ei<Contract>) {
            result.endowment_insurance = {
                prepared.endowment_death_benefit_fraction_ * path.endowment_death.expected_payoff +
                    path.payoff.expected_payoff * weights.endowment_insurance_maturity,
                prepared.endowment_death_benefit_fraction_ * path.endowment_death.delta +
                    path.payoff.delta * weights.endowment_insurance_maturity};
        }
        return result;
    }

    [[nodiscard]] static PayoffDelta pair_mean(const PayoffDelta& a, const PayoffDelta& b) noexcept
    {
        return {0.5 * a.expected_payoff + 0.5 * b.expected_payoff,
                0.5 * a.delta + 0.5 * b.delta};
    }

    template<bool Standard, bool Antithetic, PetieiContract Contract,
             class CarbonStream, class FinancialStream>
    [[nodiscard]] static PetieiMonteCarloResult evaluate_samples(
        const Prepared& prepared, std::size_t observations,
        CarbonStream& carbon_stream, FinancialStream& financial_stream)
    {
        detail::PetieiRunningMoments pe, ti, ei;
        portfolio::CarbonAllocation4::Workspace workspace;
        Eigen::Vector4d normals;
        constexpr bool intermediate_payoffs = has_ti<Contract> || has_ei<Contract>;
        for (std::size_t sample = 0; sample < observations; ++sample) {
            PathState plus = initial_path(prepared);
            PathState minus = initial_path(prepared);
            for (std::size_t date = 0; date < prepared.future_payoffs_.size(); ++date) {
                carbon_stream.fill_std_normals(normals);
                const double plus_increment = advance_carbon(plus, prepared, normals, workspace);
                double minus_increment = 0.0;
                if constexpr (Antithetic) {
                    const Eigen::Vector4d opposite = -normals;
                    minus_increment = advance_carbon(minus, prepared, opposite, workspace);
                }
                if constexpr (intermediate_payoffs) {
                    double financial = 0.0;
                    if constexpr (Standard) financial = financial_stream.std_normal();
                    evaluate_date<Standard>(plus, prepared.future_payoffs_[date],
                                                         plus_increment, financial);
                    accumulate_death<Contract>(plus, prepared, date);
                    if constexpr (Antithetic) {
                        evaluate_date<Standard>(minus, prepared.future_payoffs_[date],
                                                             minus_increment, -financial);
                        accumulate_death<Contract>(minus, prepared, date);
                    }
                }
            }
            if constexpr (!intermediate_payoffs) {
                // PE-only uses E3: one financial normal after the carbon path.
                double financial = 0.0;
                if constexpr (Standard) financial = financial_stream.std_normal();
                evaluate_date<Standard>(
                    plus, prepared.future_payoffs_.back(), plus.integrated_variance, financial);
                if constexpr (Antithetic) {
                    evaluate_date<Standard>(
                        minus, prepared.future_payoffs_.back(), minus.integrated_variance, -financial);
                }
            }
            auto value = finish_path<Contract>(plus, prepared);
            if constexpr (Antithetic) {
                const auto opposite = finish_path<Contract>(minus, prepared);
                value.pure_endowment = pair_mean(value.pure_endowment, opposite.pure_endowment);
                value.term_insurance = pair_mean(value.term_insurance, opposite.term_insurance);
                value.endowment_insurance = pair_mean(value.endowment_insurance, opposite.endowment_insurance);
            }
            const double inverse = 1.0 / static_cast<double>(sample + 1);
            if constexpr (has_pe<Contract>) pe.add(value.pure_endowment, inverse);
            if constexpr (has_ti<Contract>) ti.add(value.term_insurance, inverse);
            if constexpr (has_ei<Contract>) ei.add(value.endowment_insurance, inverse);
        }
        const double inverse = 1.0 / static_cast<double>(observations);
        const double inverse_df = 1.0 / static_cast<double>(observations - 1);
        return {Antithetic ? 2 * observations : observations,
                pe.finish(inverse_df, inverse), ti.finish(inverse_df, inverse),
                ei.finish(inverse_df, inverse), observations};
    }

    template<bool Standard, bool Antithetic,
             class CarbonStream, class FinancialStream>
    [[nodiscard]] static PetieiMonteCarloResult select_contract(
        const Prepared& prepared, std::size_t observations,
        CarbonStream& carbon, FinancialStream& financial, PetieiContract contract)
    {
        switch (contract) {
        case PetieiContract::all:
            return evaluate_samples<Standard, Antithetic, PetieiContract::all>(
                prepared, observations, carbon, financial);
        case PetieiContract::pure_endowment:
            return evaluate_samples<Standard, Antithetic, PetieiContract::pure_endowment>(
                prepared, observations, carbon, financial);
        case PetieiContract::term_insurance:
            return evaluate_samples<Standard, Antithetic, PetieiContract::term_insurance>(
                prepared, observations, carbon, financial);
        case PetieiContract::endowment_insurance:
            return evaluate_samples<Standard, Antithetic, PetieiContract::endowment_insurance>(
                prepared, observations, carbon, financial);
        }
        throw std::invalid_argument("Unknown PETIEI contract selection.");
    }

    template<bool Standard, class CarbonStream, class FinancialStream>
    [[nodiscard]] static PetieiMonteCarloResult select_sampling(
        const Prepared& prepared, std::size_t simulations,
        CarbonStream& carbon, FinancialStream& financial,
        PetieiSampling sampling, PetieiContract contract)
    {
        switch (sampling) {
        case PetieiSampling::independent:
            if (simulations < 2) throw std::invalid_argument("Independent sampling requires at least two trajectories.");
            return select_contract<Standard, false>(prepared, simulations, carbon, financial, contract);
        case PetieiSampling::antithetic:
            if (simulations < 3) throw std::invalid_argument("Antithetic sampling requires at least two complete pairs.");
            if (simulations == std::numeric_limits<std::size_t>::max()) {
                throw std::overflow_error("Rounded antithetic trajectory count overflows.");
            }
            return select_contract<Standard, true>(
                prepared, simulations / 2 + simulations % 2, carbon, financial, contract);
        }
        throw std::invalid_argument("Unknown PETIEI sampling mode.");
    }

public:
    // Conditioned estimator: existing calls retain their signature and default.
    // CarbonStream supplies fill_std_normals(Eigen::Vector4d&).
    // simulations counts actual trajectories; odd antithetic requests round up.
    // Variance/SE use one observation per path or per complete pair.
    // For a selected contract, the other result fields are zero. PE-only avoids
    // intermediate payoff evaluations. Rebuild Prepared after any valuation change.
    template<class CarbonStream>
    [[nodiscard]] static PetieiMonteCarloResult evaluate(
        const Prepared& prepared, std::size_t simulations, CarbonStream& carbon,
        PetieiSampling sampling = PetieiSampling::antithetic,
        PetieiContract contract = PetieiContract::all)
    {
        NoFinancialStream unused;
        return select_sampling<false>(
            prepared, simulations, carbon, unused, sampling, contract);
    }

    // Explicit estimator selection. Separate, independently seeded streams keep
    // extra financial draws from altering the carbon paths. FinancialStream must
    // supply std_normal(). It is not advanced by the conditioned estimator.
    // Standard antithetics negate both carbon and financial Gaussian inputs.
    // PE-only uses E3: one financial normal at maturity. TI/EI use the
    // sequential fund scheme E2 at every required payment date.
    // Caller precondition: all simulated arithmetic remains finite/representable.
    template<class CarbonStream, class FinancialStream>
    [[nodiscard]] static PetieiMonteCarloResult evaluate(
        const Prepared& prepared, std::size_t simulations,
        CarbonStream& carbon, FinancialStream& financial, PetieiEstimator estimator,
        PetieiSampling sampling = PetieiSampling::antithetic,
        PetieiContract contract = PetieiContract::all)
    {
        switch (estimator) {
        case PetieiEstimator::conditioned:
            return evaluate(prepared, simulations, carbon, sampling, contract);
        case PetieiEstimator::standard:
            return select_sampling<true>(
                prepared, simulations, carbon, financial, sampling, contract);
        }
        throw std::invalid_argument("Unknown PETIEI estimator.");
    }
};

} // namespace carbonfund::pricing

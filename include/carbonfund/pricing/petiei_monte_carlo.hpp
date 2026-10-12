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

	// Unbiased variance of one conditional path contribution,
	// not of its mean.
	double sample_variance = 0.0;
	double standard_error = 0.0;
};

struct PetieiContractEstimate {
	MonteCarloEstimate price;
	MonteCarloEstimate delta;
};

struct PetieiMonteCarloResult {
	std::size_t simulations = 0;
	PetieiContractEstimate pure_endowment;
	PetieiContractEstimate term_insurance;
	PetieiContractEstimate endowment_insurance;
};

namespace detail {

// Welford accumulation avoids subtracting almost equal sums of squares.
// One inverse sample count is shared by all six price/delta statistics.
struct PetieiRunningMoments {
	PayoffDelta mean;
	PayoffDelta sum_squared_deviations;

	void add(
		const PayoffDelta& sample,
		double inverse_count) noexcept
	{
		const double price_difference =
				sample.expected_payoff - mean.expected_payoff;
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
				sum_squared_deviations.expected_payoff *
				inverse_degrees_of_freedom;

		const double delta_variance =
				sum_squared_deviations.delta *
				inverse_degrees_of_freedom;

		return {
			{
				mean.expected_payoff,
				price_variance,
				std::sqrt(price_variance * inverse_count)
			},
			{
				mean.delta,
				delta_variance,
				std::sqrt(delta_variance * inverse_count)
			}
		};
	}
};

} // namespace detail

// Serial conditional Monte Carlo pricing of PE, TI and EI together.
// CarbonStepper is selected at compile time. The financial Gaussian and
// independent mortality are integrated analytically; only carbon is sampled.
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
						initial_payoff.expected_payoff *
								mortality_weights_.term_insurance_death[0],
						initial_payoff.delta *
								mortality_weights_.term_insurance_death[0]
				},
				initial_endowment_death_{
						initial_payoff.expected_payoff *
								mortality_weights_.endowment_insurance_death[0],
						initial_payoff.delta *
								mortality_weights_.endowment_insurance_death[0]
				},
				half_step_(0.5 * mortality_weights_.step_size),
				initial_half_variance_(initial_half_variance),
				endowment_death_benefit_fraction_(
						endowment_death_benefit_fraction)
		{
		}
	};

	// Prepare once per valuation. The resulting object owns its data;
	// cohort views are not retained. Rebuild it when the observed
	// fund/carbon state, remaining maturity, guarantees or living cohorts
	// change during hedging.
	[[nodiscard]] static Prepared prepare(
			const Parameters& parameters,
			const Eigen::Vector4d& initial_carbon_state,
			const typename CarbonStepper::Parameters& carbon_parameters,
			const portfolio::CarbonAllocationParameters4& allocation_parameters,
			const contracts::PetieiPortfolioView& portfolio_view,
			const mortality::GompertzMakeham& mortality_model)
	{
		if (
				!std::isfinite(parameters.fund_value) ||
				parameters.fund_value <= 0.0 ||
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
				parameters.time_intervals ==
						std::numeric_limits<std::size_t>::max()
		) {
				throw std::invalid_argument(
						"Pricing requires a positive finite fund and maturity, "
						"ordered nonnegative guarantees, finite growth rates, "
						"a nonnegative death-benefit fraction and positive intervals."
				);
		}

		if (!initial_carbon_state.allFinite()) {
				throw std::invalid_argument(
						"Initial carbon state must be finite."
				);
		}

		if constexpr (
				std::is_same_v<CarbonStepper, models::GeoOUExactStepper4>
		) {
				if ((initial_carbon_state.array() <= 0.0).any()) {
						throw std::invalid_argument(
								"GeoOU requires a positive initial carbon state."
						);
				}
		} else if constexpr (
				std::is_same_v<CarbonStepper, models::CIRNVStepper4>
		) {
				if ((initial_carbon_state.array() < 0.0).any()) {
						throw std::invalid_argument(
								"CIR requires a nonnegative initial carbon state."
						);
				}
		}

		const auto allocation =
				portfolio::CarbonAllocation4::prepare(allocation_parameters);

		auto mortality_weights = prepare_petiei_mortality_weights(
				portfolio_view,
				mortality_model,
				allocation_parameters.risk_free_rate,
				parameters.time_to_maturity,
				parameters.time_intervals
		);

		const double step = mortality_weights.step_size;

		const auto carbon_transition =
				CarbonStepper::prepare(step, carbon_parameters);

		std::vector<MinMaxPayoff::Prepared> future_payoffs;
		future_payoffs.reserve(parameters.time_intervals);

		for (
				std::size_t i = 1;
				i <= parameters.time_intervals;
				++i
		) {
				const double time =
						i == parameters.time_intervals
								? parameters.time_to_maturity
								: static_cast<double>(i) * step;

				const double minimum =
						parameters.minimum_guarantee == 0.0
								? 0.0
								: parameters.minimum_guarantee *
									std::exp(
											parameters.minimum_guarantee_growth_rate * time
									);

				const double maximum =
						parameters.maximum_guarantee *
						std::exp(
								parameters.maximum_guarantee_growth_rate * time
						);

				// The kernel enforces minimum < maximum at every future date.
				future_payoffs.push_back(
						MinMaxPayoff::prepare({
								parameters.fund_value,
								std::exp(
										allocation_parameters.risk_free_rate * time
								),
								minimum,
								maximum
						})
				);
		}

		// Preserve the original t=0 quadrature convention:
		// delta is one at either boundary.
		//
		// Equal guarantees are allowed only at this immediate endpoint,
		// e.g. both start at one and subsequently grow at different rates.
		PayoffDelta initial_payoff{
				parameters.fund_value,
				1.0
		};

		if (parameters.fund_value < parameters.minimum_guarantee) {
				initial_payoff = {parameters.minimum_guarantee, 0.0};
		} else if (parameters.fund_value > parameters.maximum_guarantee) {
				initial_payoff = {parameters.maximum_guarantee, 0.0};
		}

		portfolio::CarbonAllocation4::Workspace initial_allocation;

		portfolio::CarbonAllocation4::evaluate(
				initial_carbon_state,
				allocation,
				initial_allocation
		);

		const double initial_half_variance =
				0.5 * step * initial_allocation.instantaneous_variance;

		if (!std::isfinite(initial_half_variance)) {
				throw std::overflow_error(
						"Initial integrated-variance contribution overflows."
				);
		}

		return Prepared{
				carbon_transition,
				allocation,
				initial_carbon_state,
				std::move(mortality_weights),
				std::move(future_payoffs),
				initial_payoff,
				initial_half_variance,
				parameters.endowment_death_benefit_fraction
		};
	}

	// RandomStream supplies fill_std_normals(Eigen::Vector4d&),
	// as SimulationRng does. Pass an owned stream by reference;
	// this call advances that stream.
	//
	// Each path uses exactly four normal draws per carbon time step.
	// The prepared input remains unchanged.
	// Workspace is local to this call.
	//
	// Caller preconditions: transitions, variances and benefit arithmetic
	// stay finite and representable throughout the simulated paths.
	template<class RandomStream>
	[[nodiscard]] static PetieiMonteCarloResult evaluate(
			const Prepared& prepared,
			std::size_t simulations,
			RandomStream& random_stream)
	{
		if (simulations < 2) {
			throw std::invalid_argument(
					"At least two simulations are required to estimate variance.");
		}

		detail::PetieiRunningMoments pe_moments;
		detail::PetieiRunningMoments ti_moments;
		detail::PetieiRunningMoments ei_moments;

		portfolio::CarbonAllocation4::Workspace allocation_workspace;
		Eigen::Vector4d carbon_state;
		Eigen::Vector4d standard_normals;

		const auto& weights = prepared.mortality_weights_;

		for (std::size_t path = 0; path < simulations; ++path) {
			carbon_state = prepared.initial_carbon_state_;

			double integrated_variance = 0.0;
			double previous_half_variance = prepared.initial_half_variance_;

			PayoffDelta ti_sample       = prepared.initial_term_insurance_;

			PayoffDelta ei_death_sample = prepared.initial_endowment_death_;

			PayoffDelta payoff;

			for (std::size_t i = 0;i < prepared.future_payoffs_.size();++i)
			{
				random_stream.fill_std_normals(standard_normals);

				CarbonStepper::step(
						carbon_state,
						prepared.carbon_transition_,
						standard_normals
				);

				portfolio::CarbonAllocation4::evaluate(
						carbon_state,
						prepared.allocation_,
						allocation_workspace
				);

				const double current_half_variance =
						prepared.half_step_ *
						allocation_workspace.instantaneous_variance;

				integrated_variance   += previous_half_variance;
				integrated_variance   +=  current_half_variance;
				previous_half_variance =  current_half_variance;

				payoff = MinMaxPayoff::evaluate(
						prepared.future_payoffs_[i],
						integrated_variance
				);

				ti_sample.expected_payoff += payoff.expected_payoff *
																weights.term_insurance_death[i + 1];

				ti_sample.delta += payoff.delta *
														weights.term_insurance_death[i + 1];

				ei_death_sample.expected_payoff += payoff.expected_payoff *
															weights.endowment_insurance_death[i + 1];

				ei_death_sample.delta += payoff.delta *
															weights.endowment_insurance_death[i + 1];
			}

			const PayoffDelta pe_sample{
				payoff.expected_payoff * weights.pure_endowment_maturity,
									payoff.delta * weights.pure_endowment_maturity};

			const PayoffDelta ei_sample{
				prepared.endowment_death_benefit_fraction_ *
						ei_death_sample.expected_payoff +
						payoff.expected_payoff *
						weights.endowment_insurance_maturity,

				prepared.endowment_death_benefit_fraction_ *
						ei_death_sample.delta +
						payoff.delta *
						weights.endowment_insurance_maturity };

			// EI death and maturity benefits are combined on each
			// common path before estimating variance, retaining
			// their covariance.
			const double inverse_count =
					1.0 / static_cast<double>(path + 1);

			pe_moments.add(pe_sample, inverse_count);
			ti_moments.add(ti_sample, inverse_count);
			ei_moments.add(ei_sample, inverse_count);
		}

		const double inverse_count =
				1.0 / static_cast<double>(simulations);

		const double inverse_degrees_of_freedom =
				1.0 / static_cast<double>(simulations - 1);

		return {
			simulations,
			pe_moments.finish(inverse_degrees_of_freedom, inverse_count),
			ti_moments.finish(inverse_degrees_of_freedom, inverse_count),
			ei_moments.finish(inverse_degrees_of_freedom, inverse_count)};
	}
};

} // namespace carbonfund::pricing
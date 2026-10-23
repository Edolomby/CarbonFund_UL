#include <carbonfund/contracts/cohorts.hpp>
#include <carbonfund/models/carbon_models_4d.hpp>
#include <carbonfund/mortality/gompertz_makeham.hpp>
#include <carbonfund/portfolio/carbon_allocation_4d.hpp>
#include <carbonfund/pricing/petiei_monte_carlo.hpp>
#include <carbonfund/random/simulation_rng.hpp>
#include <carbonfund/utils/TablePrinter.hpp>

#include <exception>
#include <iostream>
#include <string>
#include <vector>

int main()
{
    try {
        namespace cf = carbonfund;
        using Stepper = cf::models::CIRNVStepper4;
        using Pricer = cf::pricing::PetieiMonteCarloPricer4<Stepper>;

        cf::pricing::PetieiMonteCarloParameters terms;
        terms.fund_value = 1.2;
        terms.minimum_guarantee = 1.0;
        terms.maximum_guarantee = 1.5;
        terms.minimum_guarantee_growth_rate = 0.01;
        terms.maximum_guarantee_growth_rate = 0.04;
        terms.endowment_death_benefit_fraction = 0.8;
        terms.time_to_maturity = 5.0;
        terms.time_intervals = 50;

        const Eigen::Vector4d initial_carbon{5.0, 4.0, 3.0, 1.0};
        cf::models::CIRParameters4 carbon;
        carbon.long_run_mean << 2.5, 2.0, 1.5, 0.5;
        carbon.mean_reversion.setConstant(0.1);
        carbon.volatility.setConstant(0.2);

        cf::portfolio::CarbonAllocationParameters4 allocation;
        allocation.risk_free_rate = 0.03;
        allocation.risk_aversion = 2.0;
        allocation.carbon_penalty = 0.1;
        allocation.expected_returns << 0.08, 0.10, 0.07, 0.09;
        const Eigen::Vector4d volatilities{0.20, 0.25, 0.18, 0.22};
        const Eigen::Matrix4d correlations = Eigen::Matrix4d::Identity();
        allocation.covariance = volatilities.asDiagonal() * correlations *
                                volatilities.asDiagonal();

        const cf::mortality::GompertzMakeham mortality{
            {0.0041959, 79.6921211, 1.0 / 11.5818911}};
        const cf::contracts::CohortBook pe{{{60.0, 1.0}}};
        const cf::contracts::CohortBook ti{{{60.0, 1.0}}};
        const cf::contracts::CohortBook ei{{{60.0, 1.0}}};
        const cf::contracts::PetieiPortfolioView portfolio{
            pe.cohorts(), ti.cohorts(), ei.cohorts()};

        const auto prepared = Pricer::prepare(
            terms, initial_carbon, carbon, allocation, portfolio, mortality);

        constexpr std::size_t paths = 10000;
        cf::random::SimulationRng carbon_rng{20261015};
        const auto result = Pricer::evaluate(
            prepared, paths, carbon_rng,
            cf::pricing::PetieiSampling::antithetic,
            cf::pricing::PetieiContract::all);

        std::vector<double> prices, price_variances, price_errors;
        std::vector<double> deltas, delta_variances, delta_errors;
        for (const auto* estimate : {&result.pure_endowment,
                                     &result.term_insurance,
                                     &result.endowment_insurance}) {
            prices.push_back(estimate->price.mean);
            price_variances.push_back(estimate->price.sample_variance);
            price_errors.push_back(estimate->price.standard_error);
            deltas.push_back(estimate->delta.mean);
            delta_variances.push_back(estimate->delta.sample_variance);
            delta_errors.push_back(estimate->delta.standard_error);
        }

        std::cout << "Conditioned antithetic pricing\n"
                  << "Actual trajectories: " << result.simulations
                  << "; independent pair averages: " << result.independent_samples
                  << "\nPrices and deltas include each cohort's aggregate notional.\n";
        cf::utils::printVectors(
            cf::utils::TableConfig{cf::utils::FloatFormat::Scientific, 6},
            {"Contract", "Price", "Price variance", "Price SE",
             "Delta", "Delta variance", "Delta SE"},
            std::vector<std::string>{"PE", "TI", "EI"},
            prices, price_variances, price_errors,
            deltas, delta_variances, delta_errors);
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "Pricing example error: " << error.what() << '\n';
        return 1;
    }
}

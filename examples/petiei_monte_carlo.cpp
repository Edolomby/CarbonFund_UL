#include <carbonfund/pricing/petiei_monte_carlo.hpp>
#include <carbonfund/random/simulation_rng.hpp>
#include <carbonfund/utils/TablePrinter.hpp>

#include <string>
#include <vector>

int main()
{
    using namespace carbonfund;

    using Pricer =
        pricing::PetieiMonteCarloPricer4<
            models::OUExactStepper4
        >;

    const contracts::CohortBook pe{
        {{60.0, 100000.0}}
    };

    const contracts::CohortBook ti{
        {{60.0, 100000.0}}
    };

    const contracts::CohortBook ei{
        {{60.0, 100000.0}}
    };

    const contracts::PetieiPortfolioView portfolio{
        pe.cohorts(),
        ti.cohorts(),
        ei.cohorts()
    };

    const mortality::GompertzMakeham mortality{
        {0.00279, 79.6921211, 1.0 / 11.5818911}
    };

    pricing::PetieiMonteCarloParameters contract;

    contract.fund_value = 1.0;
    contract.minimum_guarantee = 1.0;
    contract.maximum_guarantee = 1.0;
    contract.minimum_guarantee_growth_rate = 0.0;
    contract.maximum_guarantee_growth_rate = 0.05;
    contract.endowment_death_benefit_fraction = 1.0;
    contract.time_to_maturity = 5.0;
    contract.time_intervals = 60;

    const Eigen::Vector4d current_carbon{
        1.0, 1.2, 0.8, 1.1
    };

    models::OUParameters4 carbon;

    carbon.long_run_mean = current_carbon;
    carbon.mean_reversion.setConstant(0.5);
    carbon.volatility.setConstant(0.2);

    portfolio::CarbonAllocationParameters4 allocation;

    allocation.risk_free_rate = 0.03;
    allocation.risk_aversion = 2.0;
    allocation.carbon_penalty = 1.0;
    allocation.expected_returns <<
        0.06, 0.08, 0.07, 0.09;

    allocation.covariance =
        (
            Eigen::Vector4d{
                0.04, 0.06, 0.05, 0.08
            }
        ).asDiagonal();

    const auto prepared = Pricer::prepare(
        contract,
        current_carbon,
        carbon,
        allocation,
        portfolio,
        mortality
    );

    random::SimulationRng rng{20261011};

    const auto result =
        Pricer::evaluate(prepared, 1000, rng);

    const std::vector<std::string> names{"PE", "TI", "EI"};
    std::vector<double> prices, price_errors, deltas, delta_errors;
    for (const auto& value : {result.pure_endowment, result.term_insurance,
                              result.endowment_insurance}) {
        prices.push_back(value.price.mean);
        price_errors.push_back(value.price.standard_error);
        deltas.push_back(value.delta.mean);
        delta_errors.push_back(value.delta.standard_error);
    }
    utils::printVectors(
        utils::TableConfig{utils::FloatFormat::Fixed, 6},
        {"Contract", "Price", "Price SE", "Delta", "Delta SE"},
        names, prices, price_errors, deltas, delta_errors);
}

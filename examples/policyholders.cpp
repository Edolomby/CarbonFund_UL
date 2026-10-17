#include <carbonfund/contracts/policyholders.hpp>
#include <carbonfund/random/simulation_rng.hpp>
#include <carbonfund/utils/TablePrinter.hpp>

#include <exception>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

int main()
{
    try {
        using namespace carbonfund;
        using contracts::PolicyType;

        // Each row below represents a different insured life. Different
        // notionals at the same age can share one pricing cohort.
        std::vector<contracts::Policyholder> policies;
        policies.reserve(3000);
        for (int i = 0; i < 1000; ++i) {
            const double age = 60.0 + static_cast<double>(i % 5);
            policies.push_back({age, 1000.0, PolicyType::pure_endowment});
            policies.push_back({age, 1500.0, PolicyType::term_insurance});
            policies.push_back({age, i % 2 == 0 ? 800.0 : 1200.0,
                                PolicyType::endowment_insurance});
        }
        contracts::PolicyholderPopulation population{std::move(policies)};
        const mortality::GompertzMakeham mortality{
            {0.0041959, 79.6921211, 1.0 / 11.5818911}};
        // Use a separate stream from market paths and inner pricing simulations.
        random::SimulationRng mortality_rng{20261015};

        std::vector<int> years;
        std::vector<std::string> types;
        std::vector<std::size_t> death_counts, living_counts, pricing_rows;
        std::vector<double> death_notionals, living_notionals;
        const auto append = [&](int year, const char* type,
                                const contracts::PolicyholderTotals& deaths,
                                const contracts::PolicyholderTotals& living,
                                std::size_t cohorts) {
            years.push_back(year);
            types.emplace_back(type);
            death_counts.push_back(deaths.count);
            death_notionals.push_back(deaths.notional);
            living_counts.push_back(living.count);
            living_notionals.push_back(living.notional);
            pricing_rows.push_back(cohorts);
        };

        for (int year = 1; year <= 5; ++year) {
            const auto deaths = population.advance_to(
                static_cast<double>(year), mortality, mortality_rng);
            const auto living = population.living_totals();
            // For revaluation before maturity, pass this directly as the
            // portfolio argument to PetieiMonteCarloPricer4::prepare(...).
            const auto portfolio = population.living_portfolio();
            append(year, "PE", deaths.pure_endowment, living.pure_endowment,
                   portfolio.pure_endowment.size());
            append(year, "TI", deaths.term_insurance, living.term_insurance,
                   portfolio.term_insurance.size());
            append(year, "EI", deaths.endowment_insurance, living.endowment_insurance,
                   portfolio.endowment_insurance.size());
        }

        utils::printVectors(
            utils::TableConfig{utils::FloatFormat::Fixed, 2},
            {"Year", "Contract", "Deaths", "Death notional", "Living",
             "Living notional", "Pricing rows"},
            years, types, death_counts, death_notionals, living_counts,
            living_notionals, pricing_rows);

        population.reset();
        std::cout << "Reset: " << population.size() << " initial lives restored.\n";
        return 0;
    } catch (const std::exception& exception) {
        std::cerr << "Example error: " << exception.what() << '\n';
        return 1;
    }
}

#include <carbonfund/hedging/petiei_hedging.hpp>
#include <carbonfund/io/hedging_csv.hpp>
#include <carbonfund/random/simulation_rng.hpp>
#include <carbonfund/utils/TablePrinter.hpp>

#include <charconv>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <locale>
#include <string>
#include <string_view>
#include <vector>

namespace {
std::size_t count(const char* argument)
{
    const std::string_view text(argument);
    std::size_t value = 0;
    const auto [end, error] = std::from_chars(text.data(), text.data() + text.size(), value);
    if (error != std::errc{} || end != text.data() + text.size())
        throw std::invalid_argument("Scenario/pricing counts must be integers.");
    return value;
}
}

int main(int argc, char** argv)
{
    try {
        if (argc > 4) throw std::invalid_argument(
            "Usage: example_petiei_hedging [outer_scenarios=100] [pricing_paths=128] [output_prefix=petiei_hedging]");
        const std::size_t scenarios = argc > 1 ? count(argv[1]) : 100;
        const std::size_t pricing_paths = argc > 2 ? count(argv[2]) : 128;
        const std::string prefix = argc > 3 ? argv[3] : "petiei_hedging";
        if (scenarios < 2 || pricing_paths < 4)
            throw std::invalid_argument("Use at least 2 outer scenarios and 4 inner paths.");
        using namespace carbonfund;
        using Stepper = models::CIRNVStepper4;
        using Generator = simulation::MarketScenarioGenerator4<Stepper>;

        pricing::PetieiMonteCarloParameters terms;
        terms.fund_value = 1.2; // Explicit X0; never implicitly reset to one.
        terms.minimum_guarantee = 1.0;
        terms.maximum_guarantee = 1.5;
        terms.minimum_guarantee_growth_rate = 0.01;
        terms.maximum_guarantee_growth_rate = 0.04;
        terms.endowment_death_benefit_fraction = 0.8;
        terms.time_to_maturity = 5.0;
        terms.time_intervals = 50; // Initial inner pricing grid.
        constexpr std::size_t hedge_intervals = 20;
        constexpr std::size_t initial_pricing_paths = 8192;
        const Eigen::Vector4d initial_carbon{5.0, 4.0, 3.0, 1.0};
        models::CIRParameters4 carbon;
        carbon.long_run_mean << 2.5, 2.0, 1.5, 0.5;
        carbon.mean_reversion.setConstant(0.1);
        carbon.volatility.setConstant(0.2);
        portfolio::CarbonAllocationParameters4 allocation;
        allocation.risk_free_rate = 0.03;
        allocation.risk_aversion = 2.0;
        allocation.carbon_penalty = 0.1;
        allocation.expected_returns << 0.08, 0.10, 0.07, 0.09;
        const Eigen::Vector4d asset_volatility{0.20, 0.25, 0.18, 0.22};
        allocation.covariance = asset_volatility.array().square().matrix().asDiagonal();
        const mortality::GompertzMakeham mortality{
            {0.0041959, 79.6921211, 1.0 / 11.5818911}};
        std::vector<contracts::Policyholder> people;
        for (int type = 0; type < 3; ++type)
            for (int i = 0; i < 10; ++i)
                people.push_back({60.0 + i % 3, 1000.0 + 100.0 * i,
                                  static_cast<contracts::PolicyType>(type)});
        contracts::PolicyholderPopulation population{std::move(people)};

        // Fixed, distinct streams: inner effort cannot change markets/deaths.
        random::SimulationRng carbon_rng{2026101601}, financial_rng{2026101602},
            mortality_rng{2026101603}, pricing_rng{2026101604}, initial_rng{2026101605};
        const auto initial_value = hedging::price_petiei_at<Stepper>(
            terms, 0.0, terms.fund_value, initial_carbon, carbon, allocation,
            population.living_portfolio(), mortality, initial_pricing_paths, initial_rng);
        const auto prepared = Generator::prepare(terms.fund_value, initial_carbon,
            carbon, allocation, terms.time_to_maturity, hedge_intervals);
        const auto revalue = [&](double time, double fund, const Eigen::Vector4d& state,
                                 contracts::PetieiPortfolioView living) {
            return hedging::price_petiei_at<Stepper>(terms, time, fund, state,
                carbon, allocation, living, mortality, pricing_paths, pricing_rng);
        };
        simulation::MarketPath4 market;
        std::vector<hedging::PetieiHedgeOutcome> outcomes;
        outcomes.reserve(scenarios);
        for (std::size_t j = 0; j < scenarios; ++j) {
            Generator::generate(prepared, carbon_rng, financial_rng, market);
            outcomes.push_back(hedging::run_petiei_hedge(j, market, terms,
                allocation.risk_free_rate, population, mortality,
                initial_value, mortality_rng, revalue));
        }
        io::write_hedging_csv(prefix + ".csv", outcomes);

        std::ofstream meta;
        meta.exceptions(std::ios::badbit | std::ios::failbit);
        meta.open(prefix + "_parameters.json");
        meta.imbue(std::locale::classic());
        meta << std::setprecision(17);
        const auto vector_json = [&](const Eigen::Vector4d& v) {
            meta << '[';
            for (int i = 0; i < 4; ++i) meta << (i ? "," : "") << v[i];
            meta << ']';
        };
        meta << "{\n\"schema_version\":1,\n\"model\":\"CIRNVStepper4\",\n"
             << "\"cost_valuation_time\":0,\n\"positive_cost\":\"additional funding required\",\n"
             << "\"static_strategy\":\"initial aggregate deltas scaled by surviving-notional ratios per type\",\n"
             << "\"outer_measure\":\"P\",\n\"inner_measure\":\"minimal martingale\",\n"
             << "\"death_settlement\":\"interval right endpoint\",\n"
             << "\"rng\":\"Boost xoshiro256pp + Ziggurat\",\n"
             << "\"seeds\":{\"carbon\":2026101601,\"financial\":2026101602,\"mortality\":2026101603,"
                "\"pricing\":2026101604,\"initial_pricing\":2026101605},\n"
             << "\"pricing_estimator\":\"conditioned, antithetic\",\n"
             << "\"outer_scenarios\":" << scenarios << ",\n\"requested_pricing_paths\":" << pricing_paths
             << ",\n\"actual_pricing_paths\":" << (pricing_paths / 2 + pricing_paths % 2) * 2
             << ",\n\"initial_pricing_paths\":" << initial_pricing_paths
             << ",\n\"hedge_intervals\":" << hedge_intervals
             << ",\n\"initial_pricing_intervals\":" << terms.time_intervals
             << ",\n\"maturity\":" << terms.time_to_maturity
             << ",\n\"initial_fund\":" << terms.fund_value << ",\n\"initial_price\":" << initial_value.price
             << ",\n\"initial_deltas\": [" << initial_value.deltas[0] << ',' << initial_value.deltas[1]
             << ',' << initial_value.deltas[2] << "],\n\"minimum_guarantee\":" << terms.minimum_guarantee
             << ",\n\"maximum_guarantee\":" << terms.maximum_guarantee
             << ",\n\"minimum_growth\":" << terms.minimum_guarantee_growth_rate
             << ",\n\"maximum_growth\":" << terms.maximum_guarantee_growth_rate
             << ",\n\"ei_death_fraction\":" << terms.endowment_death_benefit_fraction
             << ",\n\"mortality\":{\"xi\":0.0041959,\"m\":79.6921211,\"growth_rate\":"
             << 1.0 / 11.5818911 << "},\n\"risk_free_rate\":" << allocation.risk_free_rate
             << ",\n\"risk_aversion\":" << allocation.risk_aversion
             << ",\n\"carbon_penalty\":" << allocation.carbon_penalty << ",\n\"initial_carbon\":";
        vector_json(initial_carbon); meta << ",\n\"long_run_carbon\":";
        vector_json(carbon.long_run_mean); meta << ",\n\"mean_reversion\":";
        vector_json(carbon.mean_reversion); meta << ",\n\"carbon_volatility\":";
        vector_json(carbon.volatility); meta << ",\n\"asset_returns\":";
        vector_json(allocation.expected_returns); meta << ",\n\"asset_covariance\":[";
        for (int i = 0; i < 4; ++i) {
            if (i) meta << ',';
            meta << '[';
            for (int j = 0; j < 4; ++j) meta << (j ? "," : "") << allocation.covariance(i, j);
            meta << ']';
        }
        meta << "],\n\"policy_type_order\":[\"PE\",\"TI\",\"EI\"],\n\"policies\":[";
        for (std::size_t i = 0; i < population.size(); ++i) {
            const auto& p = population.policyholders()[i];
            if (i) meta << ',';
            meta << "{\"type\":" << static_cast<int>(p.type) << ",\"age\":" << p.initial_age
                 << ",\"notional\":" << p.notional << '}';
        }
        meta << "],\n\"compiler\":\"" << __VERSION__ << "\"\n}\n";
        meta.close();

        std::vector<double> means, deviations, errors;
        for (auto member : {&hedging::PetieiHedgeOutcome::no_hedge,
                            &hedging::PetieiHedgeOutcome::static_hedge,
                            &hedging::PetieiHedgeOutcome::dynamic_hedge}) {
            double mean = 0.0, m2 = 0.0;
            std::size_t n = 0;
            for (const auto& row : outcomes) {
                ++n;
                const double difference = row.*member - mean;
                mean += difference / static_cast<double>(n);
                m2 += difference * (row.*member - mean);
            }
            const double deviation = std::sqrt(m2 / static_cast<double>(scenarios - 1));
            means.push_back(mean); deviations.push_back(deviation);
            errors.push_back(deviation / std::sqrt(static_cast<double>(scenarios)));
        }
        std::cout << "Initial premium: " << std::setprecision(12) << initial_value.price
                  << "\nDiscounted costs at time zero; positive means additional funding.\n";
        utils::printVectors(utils::TableConfig{utils::FloatFormat::Default, 8},
            {"Strategy", "Mean cost", "Std", "Mean SE"},
            std::vector<std::string>{"No hedge", "Static hedge", "Dynamic hedge"}, means, deviations, errors);
        std::cout << "Saved " << prefix << ".csv and " << prefix << "_parameters.json\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "Hedging example error: " << error.what() << '\n';
        return 1;
    }
}

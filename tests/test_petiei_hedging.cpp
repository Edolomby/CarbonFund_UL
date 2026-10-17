#include <carbonfund/hedging/petiei_hedging.hpp>
#include <carbonfund/io/hedging_csv.hpp>
#include <carbonfund/random/simulation_rng.hpp>

#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <sstream>
#include <vector>

namespace {
using namespace carbonfund;
using contracts::PolicyType;

void require(bool test, const char* message)
{ if (!test) throw std::runtime_error(message); }
void close(double actual, double expected, const char* message)
{ require(std::abs(actual - expected) <= 2e-11 * std::max({1.0, std::abs(actual), std::abs(expected)}), message); }
template<class Function> void invalid(Function&& f)
{
    try { f(); } catch (const std::invalid_argument&) { return; }
    throw std::runtime_error("Invalid hedging input was accepted.");
}
struct Tape {
    std::vector<double> values;
    std::size_t used = 0;
    double uniform01() { return values.at(used++); }
};
struct ZeroNormals {
    void fill_std_normals(Eigen::Vector4d& z) { z.setZero(); }
    double std_normal() { return 0.0; }
};
const mortality::GompertzMakeham gm{{0.0041959, 79.6921211, 1.0 / 11.5818911}};

pricing::PetieiMonteCarloParameters terms()
{
    pricing::PetieiMonteCarloParameters p;
    p.fund_value = 2.0; p.minimum_guarantee = 0.0; p.maximum_guarantee = 10.0;
    p.endowment_death_benefit_fraction = 0.5; p.time_to_maturity = 1.0; p.time_intervals = 8;
    return p;
}

hedging::PetieiHedgeOutcome prescribed(double scale, double terminal = 4.0)
{
    contracts::PolicyholderPopulation population{{
        {60, 100 * scale, PolicyType::pure_endowment},
        {60, 300 * scale, PolicyType::pure_endowment},
        {60, 200 * scale, PolicyType::term_insurance},
        {60, 400 * scale, PolicyType::endowment_insurance}}};
    simulation::MarketPath4 path{{0, .4, 1}, {2, 3, terminal},
        {Eigen::Vector4d::Ones(), Eigen::Vector4d::Constant(2), Eigen::Vector4d::Constant(3)}};
    Tape mortality{{0, .999999, .999999, 0, .999999, 0}};
    int calls = 0;
    auto value_at = [&](double t, double x, const Eigen::Vector4d& c,
                        contracts::PetieiPortfolioView living) {
        ++calls;
        close(t, .4, "Valuator was called at maturity or at the wrong time.");
        close(x, 3, "Observed fund was not injected.");
        close(c[0], 2, "Observed carbon was not injected.");
        close(living.pure_endowment[0].current_age, 60.4, "Current ages were not injected.");
        close(living.pure_endowment[0].total_notional, 300 * scale, "Wrong surviving PE notional.");
        close(living.endowment_insurance[0].total_notional, 0, "Dead EI was repriced.");
        return hedging::PetieiHedgeValuation{800 * scale, {60 * scale, 10 * scale, 0}};
    };
    const auto outcome = hedging::run_petiei_hedge(7, path, terms(), .07, population, gm,
        hedging::PetieiHedgeValuation{1000 * scale, {40 * scale, 20 * scale, 40 * scale}},
        mortality, value_at);
    require(calls == 1 && mortality.used == 6, "Repeated mortality or unexpected valuation calls.");
    return outcome;
}

void check_cash_accounting()
{
    const auto out = prescribed(1);
    const double r = .07, t = .4, T = 1.0;
    const double df1 = std::exp(-r * t), dfT = std::exp(-r * T);
    const double claims = 600 * df1 + 2000 * dfT;
    close(out.claims_pv, claims, "Claims include a wrong death/maturity benefit.");
    close(out.no_hedge, claims - 1000, "No-hedge cost has the wrong valuation date/sign.");
    close(out.static_trading_gains_pv, 100 * (3 * df1 - 2) + 50 * (4 * dfT - 3 * df1),
          "Static position must be 50 units after mortality, not notional times aggregate delta.");
    close(out.dynamic_trading_gains_pv, 100 * (3 * df1 - 2) + 70 * (4 * dfT - 3 * df1),
          "The hedge used future information or the wrong delta.");

    // Independently reproduce NOMINAL cash accounts and discount each addition.
    const double first_wealth = 800 * std::exp(r * t) + 100 * 3 - 600;
    const double first_injection = 800 - first_wealth;
    const double last_wealth = (800 - 70 * 3) * std::exp(r * (T - t)) + 70 * 4 - 2000;
    const double last_injection = -last_wealth;
    close(out.dynamic_hedge, first_injection * df1 + last_injection * dfT,
          "Dynamic funding flows were not discounted at their individual dates.");
    const double static_cash = 800 * std::exp(r * t) - 600 + (100 - 50) * 3;
    close(out.static_hedge, -(static_cash * std::exp(r * (T - t)) + 50 * 4 - 2000) * dfT,
          "Static cost disagrees with its independently evolved cash account.");
    close(out.dynamic_hedge, claims - 1000 - out.dynamic_trading_gains_pv,
          "Discounted dynamic accounting identity failed.");
    close(out.static_hedge, claims - 1000 - out.static_trading_gains_pv,
          "Discounted static accounting identity failed.");
}

void check_notional_scaling()
{
    std::vector<double> original, scaled;
    for (double terminal : {2.5, 4.0, 6.0}) {
        const auto a = prescribed(1, terminal), b = prescribed(7, terminal);
        for (auto member : {&hedging::PetieiHedgeOutcome::no_hedge,
                            &hedging::PetieiHedgeOutcome::static_hedge,
                            &hedging::PetieiHedgeOutcome::dynamic_hedge,
                            &hedging::PetieiHedgeOutcome::claims_pv})
            close(b.*member, 7 * (a.*member), "Costs must scale linearly with notionals.");
        original.push_back(a.static_hedge); scaled.push_back(b.static_hedge);
    }
    const auto variance = [](const std::vector<double>& x) {
        const double mean = (x[0] + x[1] + x[2]) / 3;
        double total = 0; for (double v : x) total += (v - mean) * (v - mean);
        return total / 2;
    };
    close(variance(scaled), 49 * variance(original), "Notional scaling did not square the variance.");
    close(hedging::mortality_adjusted_static_delta({}, {}, {}), 0, "Empty types divide by zero.");
}

template<class Stepper> void check_market_generator()
{
    typename Stepper::Parameters carbon;
    carbon.long_run_mean.setOnes(); carbon.mean_reversion.setConstant(.5);
    carbon.volatility.setZero();
    portfolio::CarbonAllocationParameters4 allocation;
    allocation.risk_free_rate = .07; allocation.risk_aversion = 2;
    allocation.expected_returns.setConstant(.17);
    allocation.covariance = Eigen::Matrix4d::Identity();
    using Generator = simulation::MarketScenarioGenerator4<Stepper>;
    const auto prepared = Generator::prepare(2.5, Eigen::Vector4d::Ones(), carbon, allocation, 2, 5);
    ZeroNormals a, b;
    simulation::MarketPath4 path;
    Generator::generate(prepared, a, b, path);
    path.validate();
    // pi_i=.05, pi'(mu-r)=.02, pi'Sigma*pi=.01.
    close(path.fund.back(), 2.5 * std::exp((.07 + .02 - .005) * 2),
          "Physical fund drift or integrated variance is incorrect.");
    close(path.fund.front(), 2.5, "Initial fund was implicitly reset to one.");
    random::SimulationRng ca{51}, fa{52}, cb{51}, fb{52};
    simulation::MarketPath4 first, second;
    Generator::generate(prepared, ca, fa, first);
    Generator::generate(prepared, cb, fb, second);
    require(first.fund == second.fund, "Market seeds are not reproducible.");
}

void check_maturity_and_revaluation()
{
    const mortality::GompertzMakeham zero_mortality{{0, 100000, 1}};
    contracts::PolicyholderPopulation population{{{60, 100, PolicyType::pure_endowment}}};
    simulation::MarketPath4 path{{0, 1}, {2, 3},
        {Eigen::Vector4d::Ones(), Eigen::Vector4d::Ones()}};
    Tape no_draws;
    auto forbidden = [](double, double, const Eigen::Vector4d&, contracts::PetieiPortfolioView)
        -> hedging::PetieiHedgeValuation { throw std::runtime_error("Maturity was repriced."); };
    for (int repeat = 0; repeat < 2; ++repeat) {
        const auto out = hedging::run_petiei_hedge(repeat, path, terms(), .1, population,
            zero_mortality, hedging::PetieiHedgeValuation{210, {100, 0, 0}}, no_draws, forbidden);
        close(out.dynamic_hedge, -10, "Linear claim hedge mishandled arbitrary X0 or cash growth.");
        close(out.static_hedge, -10, "Static hedge multiplies the notional twice.");
        close(out.claims_pv, 300 * std::exp(-.1), "Population was not reset between scenarios.");
    }

    using Stepper = models::CIRNVStepper4;
    models::CIRParameters4 carbon;
    carbon.long_run_mean.setOnes(); carbon.mean_reversion.setOnes(); carbon.volatility.setConstant(.2);
    portfolio::CarbonAllocationParameters4 allocation;
    allocation.risk_free_rate = .03; allocation.expected_returns.setConstant(.1);
    auto original = terms(); original.minimum_guarantee = 1; original.maximum_guarantee = 3;
    original.minimum_guarantee_growth_rate = .02; original.maximum_guarantee_growth_rate = .04;
    auto manual = original;
    manual.fund_value = 2.3; manual.time_to_maturity = .5; manual.time_intervals = 4;
    manual.minimum_guarantee *= std::exp(.02 * .5);
    manual.maximum_guarantee *= std::exp(.04 * .5);
    population.reset(); (void)population.advance_to(.5, zero_mortality, no_draws);
    random::SimulationRng rng_a{91}, rng_b{91};
    const auto actual = hedging::price_petiei_at<Stepper>(original, .5, 2.3,
        Eigen::Vector4d::Constant(1.1), carbon, allocation, population.living_portfolio(), gm, 32, rng_a);
    using Pricer = pricing::PetieiMonteCarloPricer4<Stepper>;
    const auto prepared = Pricer::prepare(manual, Eigen::Vector4d::Constant(1.1), carbon,
        allocation, population.living_portfolio(), gm);
    const auto expected = Pricer::evaluate(prepared, 32, rng_b);
    close(actual.price, expected.pure_endowment.price.mean, "Revaluation changed the contractual guarantee anchor.");
    close(actual.deltas[0], expected.pure_endowment.delta.mean, "Revaluation delta disagrees with direct pricing.");
    invalid([&] { (void)hedging::price_petiei_at<Stepper>(original, 1, 2.3,
        Eigen::Vector4d::Ones(), carbon, allocation, population.living_portfolio(), gm, 32, rng_a); });
    auto bad = path; bad.fund[0] = 1;
    invalid([&] { (void)hedging::run_petiei_hedge(0, bad, terms(), .1, population,
        gm, {}, no_draws, forbidden); });
    bad = path; bad.times[1] = 0;
    invalid([&] { bad.validate(); });
    bad = path; bad.carbon.clear();
    invalid([&] { bad.validate(); });
}

void check_exhausted_population()
{
    const mortality::GompertzMakeham certain{{0, -100000, 1}};
    contracts::PolicyholderPopulation population{{
        {60, 100, PolicyType::pure_endowment},
        {60, 200, PolicyType::term_insurance},
        {60, 400, PolicyType::endowment_insurance}}};
    simulation::MarketPath4 path{{0, .4, 1}, {2, 3, 4},
        {Eigen::Vector4d::Ones(), Eigen::Vector4d::Ones(), Eigen::Vector4d::Ones()}};
    Tape no_draws;
    auto forbidden = [](double, double, const Eigen::Vector4d&, contracts::PetieiPortfolioView)
        -> hedging::PetieiHedgeValuation { throw std::runtime_error("An exhausted population was repriced."); };
    const auto out = hedging::run_petiei_hedge(0, path, terms(), .07, population, certain,
        hedging::PetieiHedgeValuation{1000, {40, 20, 40}}, no_draws, forbidden);
    close(out.claims_pv, (200 + .5 * 400) * 3 * std::exp(-.07 * .4),
          "Dead PE received a claim or dead EI was paid again at maturity.");
    close(out.static_trading_gains_pv, 100 * (3 * std::exp(-.07 * .4) - 2),
          "The static hedge remained open after all deaths.");
    close(out.dynamic_trading_gains_pv, out.static_trading_gains_pv,
          "The dynamic hedge remained open after all deaths.");
    require(no_draws.used == 0, "Certain deaths consumed random draws.");
}

void check_csv()
{
    const auto root = std::filesystem::temp_directory_path() /
        ("carbonfund-hedging-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    const auto out = prescribed(1);
    io::write_hedging_csv(root / "nested/results.csv", std::vector{out});
    std::ifstream input(root / "nested/results.csv");
    std::string header, row, cell;
    std::getline(input, header); std::getline(input, row);
    require(header.find("dynamic_hedge") != std::string::npos, "CSV header is incompatible with Python.");
    std::istringstream cells(row);
    std::vector<std::string> values;
    while (std::getline(cells, cell, ',')) values.push_back(cell);
    require(values.size() == 9 && std::stod(values[3]) == out.dynamic_hedge,
            "CSV lost double precision or fields.");
    auto nonfinite = out;
    nonfinite.dynamic_hedge = std::numeric_limits<double>::infinity();
    invalid([&] { io::write_hedging_csv(root / "invalid.csv", std::vector{nonfinite}); });
    require(!std::filesystem::exists(root / "invalid.csv"), "Invalid results were exported.");
    std::filesystem::remove_all(root);
}
}

int main()
{
    try {
        check_cash_accounting(); check_notional_scaling();
        check_market_generator<models::OUExactStepper4>();
        check_market_generator<models::CIRNVStepper4>();
        check_maturity_and_revaluation(); check_exhausted_population(); check_csv();
        std::cout << "All PETIEI hedging checks passed.\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "Test failure: " << error.what() << '\n'; return 1;
    }
}

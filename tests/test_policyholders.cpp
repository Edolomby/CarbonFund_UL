#include <carbonfund/contracts/policyholders.hpp>
#include <carbonfund/pricing/petiei_mortality_weights.hpp>
#include <carbonfund/random/simulation_rng.hpp>

#include <cmath>
#include <exception>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

namespace {

using carbonfund::contracts::Policyholder;
using carbonfund::contracts::PolicyholderPopulation;
using carbonfund::contracts::PolicyType;
using carbonfund::mortality::GompertzMakeham;

const GompertzMakeham mortality{{0.0041959, 79.6921211, 1.0 / 11.5818911}};

void require(bool condition, const char* message)
{
    if (!condition) {
        throw std::runtime_error(message);
    }
}

template<class Function>
void require_invalid(Function&& function)
{
    try {
        function();
    } catch (const std::invalid_argument&) {
        return;
    }
    throw std::runtime_error("Invalid population input was accepted.");
}

struct TapeRng {
    std::vector<double> values;
    std::size_t used = 0;

    double uniform01()
    {
        if (used == values.size()) {
            throw std::runtime_error("An unexpected mortality draw was requested.");
        }
        return values[used++];
    }
};

PolicyholderPopulation make_population()
{
    return PolicyholderPopulation{{
        {60.0, 1000.0, PolicyType::pure_endowment},
        {60.0, 2500.0, PolicyType::pure_endowment},
        {65.0,  700.0, PolicyType::term_insurance},
        {60.0,  300.0, PolicyType::endowment_insurance},
        {70.0,    0.0, PolicyType::endowment_insurance}
    }};
}

void check_individual_deaths_and_reset()
{
    auto population = make_population();
    const auto initial = population.initial_totals();
    const auto view = population.living_portfolio();
    const auto* pe_storage = view.pure_endowment.data();
    require(population.size() == 5 && population.elapsed_time() == 0.0 &&
            initial.pure_endowment.count == 2 &&
            initial.pure_endowment.notional == 3500.0 &&
            initial.term_insurance.notional == 700.0 &&
            initial.endowment_insurance.count == 2 &&
            view.pure_endowment.size() == 1 &&
            view.pure_endowment[0].total_notional == 3500.0 &&
            view.endowment_insurance.size() == 2,
            "Initial grouping or population totals are incorrect.");

    TapeRng first{{0.0, 0.999999, 0.0, 0.999999, 0.999999}};
    const auto deaths = population.advance_to(1.0, mortality, first);
    const auto living = population.living_totals();
    require(first.used == 5 &&
            deaths.pure_endowment.count == 1 &&
            deaths.pure_endowment.notional == 1000.0 &&
            deaths.term_insurance.count == 1 &&
            deaths.term_insurance.notional == 700.0 &&
            deaths.endowment_insurance.count == 0 &&
            living.pure_endowment.count == 1 &&
            living.pure_endowment.notional == 2500.0 &&
            living.term_insurance.count == 0 &&
            living.endowment_insurance.count == 2 &&
            !population.is_alive(0) && population.is_alive(1) &&
            !population.is_alive(2) && population.is_alive(3),
            "Deaths must remove the specific person's notional exactly once.");
    require(view.pure_endowment[0].current_age == 61.0 &&
            view.pure_endowment[0].total_notional == 2500.0 &&
            view.term_insurance[0].current_age == 66.0 &&
            view.term_insurance[0].total_notional == 0.0,
            "Borrowed pricing cohorts were not updated in place.");

    auto independent_copy = population;
    independent_copy.reset();
    require(!population.is_alive(0) && independent_copy.is_alive(0) &&
            population.elapsed_time() == 1.0 &&
            independent_copy.elapsed_time() == 0.0,
            "Scenario copies must own independent mutable state.");

    TapeRng second{{0.0, 0.0, 0.0}};
    const auto later_deaths = population.advance_to(3.0, mortality, second);
    require(second.used == 3 &&
            later_deaths.pure_endowment.count == 1 &&
            later_deaths.pure_endowment.notional == 2500.0 &&
            later_deaths.term_insurance.count == 0 &&
            later_deaths.endowment_insurance.count == 2 &&
            later_deaths.endowment_insurance.notional == 300.0 &&
            population.living_totals().pure_endowment.count == 0 &&
            population.living_totals().endowment_insurance.count == 0,
            "Already dead people were sampled again or zero notionals were lost.");
    TapeRng no_draws;
    const auto no_deaths = population.advance_to(4.0, mortality, no_draws);
    require(no_deaths.pure_endowment.count == 0 &&
            no_deaths.term_insurance.count == 0 &&
            no_deaths.endowment_insurance.count == 0 && no_draws.used == 0,
            "An exhausted population must not produce further deaths or draws.");

    population.reset();
    require(population.elapsed_time() == 0.0 && population.is_alive(0) &&
            population.is_alive(4) &&
            population.living_totals().pure_endowment.notional == 3500.0 &&
            view.pure_endowment[0].current_age == 60.0 &&
            view.pure_endowment[0].total_notional == 3500.0 &&
            population.living_portfolio().pure_endowment.data() == pe_storage,
            "Reset must restore the original population without reallocating.");
    TapeRng repeated{{0.0, 0.999999, 0.0, 0.999999, 0.999999}};
    const auto repeated_deaths = population.advance_to(1.0, mortality, repeated);
    require(repeated_deaths.pure_endowment.notional == deaths.pure_endowment.notional &&
            repeated_deaths.term_insurance.notional == deaths.term_insurance.notional,
            "Reset followed by the same mortality tape must reproduce the deaths.");
}

void check_conditional_probability_and_boundary()
{
    PolicyholderPopulation population{{
        {60.0, 10.0, PolicyType::pure_endowment},
        {60.0, 30.0, PolicyType::pure_endowment}
    }};
    const double first_probability = mortality.death_probability(60.0, 5.0);
    TapeRng first{{std::nextafter(first_probability, 0.0), first_probability}};
    const auto deaths = population.advance_to(5.0, mortality, first);
    require(deaths.pure_endowment.notional == 10.0 &&
            !population.is_alive(0) && population.is_alive(1),
            "Death uses the strict uniform < probability convention.");

    const double next_probability = mortality.death_probability(65.0, 5.0);
    require(next_probability > first_probability,
            "The test requires mortality to increase with age.");
    TapeRng second{{0.5 * (first_probability + next_probability)}};
    const auto next_deaths = population.advance_to(10.0, mortality, second);
    require(second.used == 1 && next_deaths.pure_endowment.notional == 30.0,
            "Later mortality must be conditional on the survivor's current age.");
}

void check_small_surviving_notional()
{
    PolicyholderPopulation population{{
        {60.0, 1e16, PolicyType::term_insurance},
        {60.0,  1.0, PolicyType::term_insurance}
    }};
    TapeRng rng{{0.0, 0.999999}};
    const auto deaths = population.advance_to(1.0, mortality, rng);
    require(deaths.term_insurance.notional == 1e16 &&
            population.living_totals().term_insurance.notional == 1.0 &&
            population.living_portfolio().term_insurance[0].total_notional == 1.0,
            "A small survivor must not disappear when a large policyholder dies.");
}

void check_endpoints_and_time()
{
    // Gompertz terms underflow/saturate: exact p=0 and p=1 in double precision.
    const GompertzMakeham zero_mortality{{0.0, 100000.0, 1.0}};
    const GompertzMakeham certain_mortality{{0.0, -100000.0, 1.0}};
    auto population = make_population();
    TapeRng no_draws;
    const auto at_zero = population.advance_to(0.0, mortality, no_draws);
    const auto no_deaths = population.advance_to(0.1, zero_mortality, no_draws);
    (void)population.advance_to(0.3, zero_mortality, no_draws);
    require(at_zero.pure_endowment.count == 0 &&
            no_deaths.term_insurance.count == 0 &&
            population.living_totals().pure_endowment.count == 2 &&
            population.living_portfolio().pure_endowment[0].current_age == 60.0 + 0.3,
            "Zero mortality/time or absolute-age updates are incorrect.");
    const auto all_deaths = population.advance_to(0.4, certain_mortality, no_draws);
    require(no_draws.used == 0 && all_deaths.pure_endowment.count == 2 &&
            all_deaths.term_insurance.count == 1 &&
            all_deaths.endowment_insurance.count == 2,
            "Probability zero/one must be deterministic and consume no RNG draws.");

    PolicyholderPopulation empty;
    const auto empty_deaths = empty.advance_to(1.0, mortality, no_draws);
    require(empty_deaths.pure_endowment.count == 0 && empty.size() == 0 &&
            empty.elapsed_time() == 1.0 &&
            empty.living_portfolio().term_insurance.empty(),
            "Empty populations must remain valid pricing inputs.");
}

void check_pricing_view()
{
    auto population = make_population();
    TapeRng rng{{0.0, 0.999999, 0.0, 0.999999, 0.999999}};
    (void)population.advance_to(1.0, mortality, rng);
    const auto weights = carbonfund::pricing::prepare_petiei_mortality_weights(
        population.living_portfolio(), mortality, 0.03, 4.0, 48);
    const double expected_pe = 2500.0 * std::exp(-0.03 * 4.0) *
        mortality.survival_probability(61.0, 4.0);
    require(std::abs(weights.pure_endowment_maturity - expected_pe) < 1e-10 &&
            weights.term_insurance_death.size() == 49,
            "The existing pricer must accept the updated living cohorts directly.");
    for (double weight : weights.term_insurance_death) {
        require(weight == 0.0, "Dead term policies still contributed pricing weight.");
    }
}

void check_statistical_survival()
{
    constexpr std::size_t lives = 50000;
    std::vector<Policyholder> policies(
        lives, Policyholder{60.0, 1.0, PolicyType::pure_endowment});
    PolicyholderPopulation population{std::move(policies)};
    const double probability = mortality.death_probability(60.0, 10.0);
    const double expected = static_cast<double>(lives) * probability;
    const double standard_deviation = std::sqrt(expected * (1.0 - probability));
    carbonfund::random::SimulationRng direct_rng{20261015};
    const auto direct = population.advance_to(10.0, mortality, direct_rng);
    require(std::abs(static_cast<double>(direct.pure_endowment.count) - expected) <
                6.0 * standard_deviation,
            "Simulated deaths disagree with the analytic survival probability.");

    population.reset();
    carbonfund::random::SimulationRng split_rng{20261016};
    const auto first = population.advance_to(5.0, mortality, split_rng);
    const auto second = population.advance_to(10.0, mortality, split_rng);
    const auto split_deaths = first.pure_endowment.count + second.pure_endowment.count;
    require(std::abs(static_cast<double>(split_deaths) - expected) <
                6.0 * standard_deviation &&
            split_deaths + population.living_totals().pure_endowment.count == lives &&
            population.living_totals().pure_endowment.notional ==
                static_cast<double>(lives - split_deaths),
            "Splitting the mortality grid changed survival or population accounting.");
}

void check_validation()
{
    const double nan = std::numeric_limits<double>::quiet_NaN();
    const double infinity = std::numeric_limits<double>::infinity();
    const double largest = std::numeric_limits<double>::max();
    const auto bad_policy = [](Policyholder policy) {
        require_invalid([&] { const PolicyholderPopulation population{{policy}}; });
    };
    bad_policy({-1.0, 1.0});
    bad_policy({60.0, -1.0});
    bad_policy({nan, 1.0});
    bad_policy({60.0, nan});
    bad_policy({infinity, 1.0});
    bad_policy({60.0, infinity});
    bad_policy({60.0, 1.0, static_cast<PolicyType>(3)});
    require_invalid([&] {
        const PolicyholderPopulation population{{
            {60.0, largest, PolicyType::pure_endowment},
            {60.0, largest, PolicyType::endowment_insurance}
        }};
    });

    auto population = make_population();
    TapeRng no_draws;
    require_invalid([&] { (void)population.advance_to(-1.0, mortality, no_draws); });
    require_invalid([&] { (void)population.advance_to(nan, mortality, no_draws); });
    require_invalid([&] { (void)population.advance_to(infinity, mortality, no_draws); });
    const GompertzMakeham zero_mortality{{0.0, 100000.0, 1.0}};
    (void)population.advance_to(1.0, zero_mortality, no_draws);
    require_invalid([&] { (void)population.advance_to(0.5, mortality, no_draws); });
    require(population.elapsed_time() == 1.0 && no_draws.used == 0 &&
            population.living_totals().pure_endowment.count == 2,
            "Invalid times must leave the population unchanged.");
    PolicyholderPopulation extreme{{{largest, 1.0}}};
    require_invalid([&] { (void)extreme.advance_to(largest, mortality, no_draws); });
    require(extreme.elapsed_time() == 0.0 && extreme.is_alive(0),
            "Overflowing ages must be rejected before changing lives.");
}

} // namespace

int main()
{
    try {
        check_individual_deaths_and_reset();
        check_conditional_probability_and_boundary();
        check_small_surviving_notional();
        check_endpoints_and_time();
        check_pricing_view();
        check_statistical_survival();
        check_validation();
        std::cout << "All policyholder checks passed.\n";
        return 0;
    } catch (const std::exception& exception) {
        std::cerr << "Test failure: " << exception.what() << '\n';
        return 1;
    }
}

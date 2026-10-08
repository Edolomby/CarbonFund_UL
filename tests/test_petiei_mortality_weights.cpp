#include <carbonfund/pricing/petiei_mortality_weights.hpp>

#include <algorithm>
#include <cmath>
#include <exception>
#include <iostream>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <vector>

namespace {

using carbonfund::contracts::CohortBook;
using carbonfund::contracts::PetieiPortfolioView;
using carbonfund::mortality::GompertzMakeham;
using carbonfund::mortality::GompertzMakehamParameters;
using carbonfund::pricing::prepare_petiei_mortality_weights;

void require(bool condition, const char* message)
{
    if (!condition) {
        throw std::runtime_error(message);
    }
}

void require_close(double obtained, double expected, const char* message)
{
    require(
        std::isfinite(obtained) &&
        std::abs(obtained - expected) <=
            2e-12 * std::max(1.0, std::abs(expected)),
        message);
}

template<class Exception, class Function>
void require_throws(Function function)
{
    try {
        function();
    } catch (const Exception&) {
        return;
    }
    throw std::runtime_error(
        "Expected input or overflow error was not raised.");
}

void check_original_pricer_weights()
{
    const GompertzMakehamParameters p{
        0.0027900, 79.6921211, 1.0 / 11.5818911};
    const GompertzMakeham model{p};

    const CohortBook pe{{{60.0, 1200.0}, {65.0, 3500.0}}};
    const CohortBook ti{{{55.5, 2700.0}, {64.5, 9100.0}}};
    const CohortBook ei{{{62.5, 4500.0}}};

    const PetieiPortfolioView portfolio{
        pe.cohorts(), ti.cohorts(), ei.cohorts()};

    const double rate = 0.03;
    const double maturity = 7.0;

    // Reference formulas from the original pricer.
    const auto survival = [&](double age, double time) {
        return std::exp(
            -p.makeham_intensity * time -
            std::exp(
                (age - p.gompertz_location_age) *
                p.gompertz_growth_rate) *
            std::expm1(p.gompertz_growth_rate * time));
    };

    const auto death_density = [&](double age, double time) {
        return (
            p.makeham_intensity +
            p.gompertz_growth_rate *
                std::exp(
                    (age + time - p.gompertz_location_age) *
                    p.gompertz_growth_rate)) *
            survival(age, time);
    };

    for (std::size_t intervals : {std::size_t{1}, std::size_t{8}}) {
        const auto w = prepare_petiei_mortality_weights(
            portfolio, model, rate, maturity, intervals);

        require(
            w.term_insurance_death.size() == intervals + 1 &&
            w.endowment_insurance_death.size() == intervals + 1,
            "Death arrays must contain both endpoints.");

        require_close(
            w.step_size, maturity / intervals, "Incorrect time step.");

        require_close(
            w.pure_endowment_maturity,
            std::exp(-rate * maturity) *
                (1200.0 * survival(60.0, maturity) +
                 3500.0 * survival(65.0, maturity)),
            "PE maturity weight differs from the original.");

        require_close(
            w.endowment_insurance_maturity,
            std::exp(-rate * maturity) *
                4500.0 * survival(62.5, maturity),
            "EI maturity weight differs from the original.");

        for (std::size_t i = 0; i <= intervals; ++i) {
            const double t =
                i == intervals ? maturity : i * w.step_size;
            const double quadrature =
                w.step_size * ((i == 0 || i == intervals) ? 0.5 : 1.0);

            require_close(
                w.term_insurance_death[i],
                quadrature * std::exp(-rate * t) *
                    (2700.0 * death_density(55.5, t) +
                     9100.0 * death_density(64.5, t)),
                "TI weight differs from the original trapezoidal formula.");

            require_close(
                w.endowment_insurance_death[i],
                quadrature * std::exp(-rate * t) *
                    4500.0 * death_density(62.5, t),
                "EI weight differs from the original trapezoidal formula.");
        }
    }
}

void check_aggregation_and_scaling()
{
    const GompertzMakeham model{{0.01, 80.0, 0.1}};
    const CohortBook split{{{60.0, 40000.0}, {60.0, 60000.0}}};
    const CohortBook merged{{{60.0, 100000.0}}};
    const CohortBook scaled{{{60.0, 250000.0}}};

    const auto prepare = [&](const CohortBook& book) {
        return prepare_petiei_mortality_weights(
            {book.cohorts(), book.cohorts(), book.cohorts()},
            model, -0.02, 5.0, 16);
    };

    const auto a = prepare(split);
    const auto b = prepare(merged);
    const auto c = prepare(scaled);

    require_close(
        a.pure_endowment_maturity, b.pure_endowment_maturity,
        "Splitting a cohort changed its maturity weight.");

    require_close(
        c.pure_endowment_maturity, 2.5 * b.pure_endowment_maturity,
        "Maturity weight is not linear in notional.");

    require_close(
        a.endowment_insurance_maturity, b.endowment_insurance_maturity,
        "Splitting a cohort changed its EI maturity weight.");

    for (std::size_t i = 0; i < a.term_insurance_death.size(); ++i) {
        require_close(
            a.term_insurance_death[i], b.term_insurance_death[i],
            "Splitting a cohort changed its death weight.");

        require_close(
            c.term_insurance_death[i], 2.5 * b.term_insurance_death[i],
            "Death weight is not linear in notional.");

        require_close(
            a.endowment_insurance_death[i], a.term_insurance_death[i],
            "TI and EI must share death weights for the same population.");
    }
}

void check_quadrature_convergence()
{
    const GompertzMakeham model{{0.01, 80.0, 0.1}};
    const CohortBook book{{{80.0, 1.0}}};
    const PetieiPortfolioView portfolio{
        book.cohorts(), book.cohorts(), {}};

    const double exact_death = model.death_probability(80.0, 2.0);

    const auto error = [&](std::size_t intervals) {
        const auto w = prepare_petiei_mortality_weights(
            portfolio, model, 0.0, 2.0, intervals);

        return std::abs(
            std::accumulate(
                w.term_insurance_death.begin(),
                w.term_insurance_death.end(), 0.0) -
            exact_death);
    };

    require(
        error(64) < error(16) / 10.0 && error(512) < 1e-8,
        "Death quadrature failed to converge to the exact death probability.");
}

void check_boundaries_and_errors()
{
    const GompertzMakeham model{{0.01, 80.0, 0.1}};
    const CohortBook book{{{60.0, 1000.0}, {65.0, 2000.0}}};
    const PetieiPortfolioView portfolio{
        book.cohorts(), book.cohorts(), book.cohorts()};

    const auto at_maturity = prepare_petiei_mortality_weights(
        portfolio, model, 0.03, 0.0, 1);

    require(
        at_maturity.step_size == 0.0 &&
        at_maturity.pure_endowment_maturity == 3000.0 &&
        at_maturity.endowment_insurance_maturity == 3000.0 &&
        at_maturity.term_insurance_death == std::vector<double>(2, 0.0) &&
        at_maturity.endowment_insurance_death == std::vector<double>(2, 0.0),
        "Zero maturity weights are incorrect.");

    const auto empty = prepare_petiei_mortality_weights(
        {}, model, -1000.0, 1.0, 4);

    require(
        empty.pure_endowment_maturity == 0.0 &&
        empty.endowment_insurance_maturity == 0.0 &&
        empty.term_insurance_death == std::vector<double>(5, 0.0) &&
        empty.endowment_insurance_death == std::vector<double>(5, 0.0),
        "An empty portfolio must have zero weights.");

    const GompertzMakeham extreme{{0.0, -710.0, 1.0}};
    const CohortBook unit{{{0.0, 1.0}}};
    const double tiny_maturity = 1000.0 * std::exp(-710.0);
    const double density =
        std::exp(extreme.log_death_density(0.0, tiny_maturity));

    require(
        std::isinf(extreme.hazard(0.0)) &&
        extreme.survival_probability(0.0, tiny_maturity) == 0.0 &&
        std::isfinite(density) && density > 0.0,
        "A representable death density was lost to intermediate overflow.");

    require(
        extreme.log_death_density(0.0, 1.0) ==
            -std::numeric_limits<double>::infinity(),
        "Large cumulative hazard produced an invalid log density.");

    const auto stable = prepare_petiei_mortality_weights(
        {{}, unit.cohorts(), {}}, extreme, 0.0, tiny_maturity, 1);

    require_close(
        stable.term_insurance_death[0],
        0.5 * std::exp(710.0 + std::log(tiny_maturity)),
        "A finite quadrature weight was lost to hazard overflow.");

    const double infinity = std::numeric_limits<double>::infinity();

    require_throws<std::invalid_argument>([&] {
        static_cast<void>(prepare_petiei_mortality_weights(
            portfolio, model, infinity, 1.0, 4));
    });

    require_throws<std::invalid_argument>([&] {
        static_cast<void>(model.log_death_density(-1.0, 1.0));
    });

    require_throws<std::invalid_argument>([&] {
        static_cast<void>(prepare_petiei_mortality_weights(
            portfolio, model, 0.03, -1.0, 4));
    });

    require_throws<std::invalid_argument>([&] {
        static_cast<void>(prepare_petiei_mortality_weights(
            portfolio, model, 0.03, 1.0, 0));
    });

    require_throws<std::invalid_argument>([&] {
        static_cast<void>(prepare_petiei_mortality_weights(
            portfolio, model, 0.03,
            std::numeric_limits<double>::denorm_min(), 2));
    });

    const std::vector<carbonfund::contracts::Cohort> invalid{
        {60.0, -1.0}};

    require_throws<std::invalid_argument>([&] {
        static_cast<void>(prepare_petiei_mortality_weights(
            {invalid, {}, {}}, model, 0.03, 1.0, 4));
    });

    require_throws<std::overflow_error>([&] {
        static_cast<void>(prepare_petiei_mortality_weights(
            portfolio, model, -1000.0, 1.0, 4));
    });
}

} // namespace

int main()
{
    try {
        check_original_pricer_weights();
        check_aggregation_and_scaling();
        check_quadrature_convergence();
        check_boundaries_and_errors();
        std::cout << "All PETIEI mortality-weight checks passed.\n";
        return 0;
    } catch (const std::exception& exception) {
        std::cerr << "Test failure: " << exception.what() << '\n';
        return 1;
    }
}
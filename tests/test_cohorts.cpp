#include <carbonfund/contracts/cohorts.hpp>

#include <exception>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

namespace {

using carbonfund::contracts::Cohort;
using carbonfund::contracts::CohortBook;
using carbonfund::contracts::PetieiPortfolioView;

void require(bool condition, const char* message)
{
    if (!condition) {
        throw std::runtime_error(message);
    }
}

void require_invalid(std::vector<Cohort> rows)
{
    try {
        const CohortBook book{std::move(rows)};
    } catch (const std::invalid_argument&) {
        return;
    }

    throw std::runtime_error("Invalid cohort book was accepted.");
}

void check_notional_convention()
{
    // Same per-person notional, with different numbers of people at each age.
    const double per_person_notional = 1000.0;

    const CohortBook book{{
        {60.0, 100.0 * per_person_notional},
        {65.0,  80.0 * per_person_notional}
    }};

    const auto rows = book.cohorts();

    require(
        rows.size() == 2 &&
        rows[0].current_age == 60.0 &&
        rows[1].current_age == 65.0,
        "Cohort ages were not preserved.");

    require(
        rows[0].total_notional == 100000.0 &&
        rows[1].total_notional == 80000.0 &&
        book.total_notional() == 180000.0,
        "Aggregate cohort notional is incorrect.");

    // Splitting a row does not change the total notional.
    const CohortBook split{{
        {60.0, 40000.0},
        {60.0, 60000.0}
    }};

    const CohortBook merged{{
        {60.0, 100000.0}
    }};

    require(
        split.total_notional() == merged.total_notional(),
        "Splitting a cohort changed total notional.");
}

void check_empty_and_zero_notional()
{
    const CohortBook empty;

    require(
        empty.cohorts().empty() &&
        empty.total_notional() == 0.0,
        "An empty contract book must have zero notional.");

    const CohortBook zero{{
        {60.0, 0.0},
        {65.0, 0.0}
    }};

    require(
        zero.total_notional() == 0.0,
        "Zero-notional rows must be allowed.");
}

void check_pricing_input()
{
    const CohortBook pe{{
        {60.0, 100000.0},
        {65.0, 80000.0}
    }};

    const CohortBook ti;

    const CohortBook ei{{
        {62.5, 35000.0}
    }};

    const PetieiPortfolioView portfolio{
        .pure_endowment = pe.cohorts(),
        .term_insurance = ti.cohorts(),
        .endowment_insurance = ei.cohorts()
    };

    require(
        portfolio.pure_endowment.size() == 2 &&
        portfolio.pure_endowment[1].total_notional == 80000.0 &&
        portfolio.term_insurance.empty() &&
        portfolio.endowment_insurance.size() == 1 &&
        portfolio.endowment_insurance[0].current_age == 62.5 &&
        portfolio.endowment_insurance[0].total_notional == 35000.0,
        "PE, TI and EI pricing inputs are inconsistent.");
}

void check_validation()
{
    const double nan = std::numeric_limits<double>::quiet_NaN();
    const double infinity = std::numeric_limits<double>::infinity();
    const double largest = std::numeric_limits<double>::max();

    require_invalid({{-1.0, 1000.0}});
    require_invalid({{60.0, -1000.0}});
    require_invalid({{nan, 1000.0}});
    require_invalid({{60.0, nan}});
    require_invalid({{infinity, 1000.0}});
    require_invalid({{60.0, infinity}});
    require_invalid({{60.0, largest}, {65.0, largest}});
}

} // namespace

int main()
{
    try {
        check_notional_convention();
        check_empty_and_zero_notional();
        check_pricing_input();
        check_validation();

        std::cout << "All cohort checks passed.\n";
        return 0;
    } catch (const std::exception& exception) {
        std::cerr << "Test failure: " << exception.what() << '\n';
        return 1;
    }
}
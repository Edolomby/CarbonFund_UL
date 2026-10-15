#include <carbonfund/pricing/min_max_payoff.hpp>

#include <algorithm>
#include <cmath>
#include <exception>
#include <iostream>
#include <limits>
#include <numbers>
#include <stdexcept>
#include <vector>

namespace {

using Kernel = carbonfund::pricing::MinMaxPayoff;
using Parameters = carbonfund::pricing::MinMaxPayoffParameters;

void require(bool condition, const char* message)
{
    if (!condition) {
        throw std::runtime_error(message);
    }
}

void require_close(
    double obtained,
    long double expected,
    const char* message,
    long double relative_tolerance = 1e-10L,
    long double absolute_tolerance = 1e-12L)
{
    require(
        std::isfinite(obtained) &&
        std::abs(static_cast<long double>(obtained) - expected) <=
            absolute_tolerance +
            relative_tolerance * std::abs(expected),
        message);
}

template<class Function>
long double integrate(
    Function function,
    long double lower,
    long double upper)
{
    if (lower >= upper) {
        return 0.0L;
    }

    constexpr int intervals = 4096;
    const long double h = (upper - lower) / intervals;
    long double total = function(lower) + function(upper);

    for (int i = 1; i < intervals; ++i) {
        total += (i % 2 == 0 ? 2.0L : 4.0L) *
                 function(lower + i * h);
    }

    return total * h / 3.0L;
}

long double normal_density(long double z)
{
    return std::exp(-0.5L * z * z) /
           std::sqrt(2.0L * std::numbers::pi_v<long double>);
}

void check_against_direct_integration()
{
    // Independent reference: integrate the clipped payoff over normal draws.
    // Splitting at the kinks avoids quadrature error from nonsmooth payoffs.
    for (const Parameters& p : {
             Parameters{0.7, 1.03, 0.8, 1.4},
             Parameters{1.0, 1.03, 0.8, 1.4},
             Parameters{1.4, 1.03, 0.8, 1.4},
             Parameters{1.0, 1.06, 0.0, 2.5}}) {
        const auto prepared = Kernel::prepare(p);

        for (double variance : {0.01, 0.3, 2.0}) {
            const long double v = variance;
            const long double s = std::sqrt(v);

            const long double forward =
                static_cast<long double>(p.fund_value) *
                p.risk_free_growth;

            const long double minimum = p.minimum_guarantee;
            const long double maximum = p.maximum_guarantee;

            const long double lower_kink = minimum == 0.0L
                ? -std::numeric_limits<long double>::infinity()
                : (std::log(minimum / forward) + 0.5L * v) / s;

            const long double upper_kink =
                (std::log(maximum / forward) + 0.5L * v) / s;

            const auto payoff_integrand = [&](long double z) {
                const long double fund =
                    forward * std::exp(-0.5L * v + s * z);

                return std::clamp(fund, minimum, maximum) *
                       normal_density(z);
            };

            std::vector<long double> cuts{
                -12.0L,
                12.0L,
                std::clamp(lower_kink, -12.0L, 12.0L),
                std::clamp(upper_kink, -12.0L, 12.0L)};

            std::sort(cuts.begin(), cuts.end());

            long double expected_payoff = 0.0L;

            for (std::size_t i = 1; i < cuts.size(); ++i) {
                expected_payoff += integrate(
                    payoff_integrand, cuts[i - 1], cuts[i]);
            }

            const long double expected_delta = integrate(
                [&](long double z) {
                    return p.risk_free_growth *
                           std::exp(-0.5L * v + s * z) *
                           normal_density(z);
                },
                std::max(-12.0L, lower_kink),
                std::min(12.0L, upper_kink));

            const auto result = Kernel::evaluate(prepared, variance);

            require_close(
                result.expected_payoff, expected_payoff,
                "Payoff differs from direct integration.");

            require_close(
                result.delta, expected_delta,
                "Delta differs from direct integration.");

            const double payoff_tolerance =
                1e-12 * std::max({
                    1.0, p.minimum_guarantee, p.maximum_guarantee});

            const double delta_tolerance =
                1e-12 * std::max(1.0, p.risk_free_growth);

            require(
                result.expected_payoff >=
                    p.minimum_guarantee - payoff_tolerance &&
                result.expected_payoff <=
                    p.maximum_guarantee + payoff_tolerance &&
                result.delta >= -delta_tolerance &&
                result.delta <= p.risk_free_growth + delta_tolerance,
                "Payoff or delta violates its mathematical bounds.");

            // Hold guarantees and variance fixed while bumping the fund.
            const double bump = p.fund_value * 1e-5;

            Parameters down = p;
            Parameters up = p;
            down.fund_value -= bump;
            up.fund_value += bump;

            const double difference =
                (Kernel::evaluate(
                     Kernel::prepare(up), variance).expected_payoff -
                 Kernel::evaluate(
                     Kernel::prepare(down), variance).expected_payoff) /
                (2.0 * bump);

            require_close(
                result.delta, difference,
                "Delta differs from a central finite difference.",
                2e-8L, 2e-9L);
        }
    }
}

void check_deterministic_payoffs()
{
    for (const Parameters& p : {
             Parameters{0.5, 2.0, 2.0, 3.0},
             Parameters{1.0, 2.0, 2.0, 3.0},
             Parameters{1.25, 2.0, 2.0, 3.0},
             Parameters{1.5, 2.0, 2.0, 3.0},
             Parameters{2.0, 2.0, 2.0, 3.0}}) {
        const double forward = p.fund_value * p.risk_free_growth;

        const double expected_payoff =
            std::clamp(
                forward, p.minimum_guarantee, p.maximum_guarantee);

        const double expected_delta =
            forward < 2.0 || forward > 3.0 ? 0.0
            : forward == 2.0 || forward == 3.0 ? 1.0
            : 2.0;

        const auto prepared = Kernel::prepare(p);

        for (double variance : {0.0, 1e-320}) {
            const auto result = Kernel::evaluate(prepared, variance);

            require(
                result.expected_payoff == expected_payoff &&
                result.delta == expected_delta,
                "Deterministic payoff or boundary delta is incorrect.");
        }
    }

    const auto limit = Kernel::evaluate(
        Kernel::prepare({1.0, 1.1, 0.8, 1.4}),
        std::numeric_limits<double>::max());

    require(
        limit.expected_payoff == 0.8 && limit.delta == 0.0,
        "Large variance should approach the floor payoff.");
}

void check_tails_and_narrow_guarantees()
{
    const double variance = 0.04;
    const double s = std::sqrt(variance);

    for (double lower : {9.0, -10.0}) {
        const double upper = lower + 1.0;

        const auto prepared = Kernel::prepare({
            1.0,
            1.0,
            std::exp(s * (lower + 0.5 * s)),
            std::exp(s * (upper + 0.5 * s))});

        const auto result = Kernel::evaluate(prepared, variance);

        require_close(
            result.delta,
            integrate(normal_density, lower, upper),
            "A small tail delta was lost to CDF subtraction.",
            1e-11L, 0.0L);

        require(
            result.delta > 0.0,
            "A representable tail delta became zero.");
    }

    const double cap = std::nextafter(1.0, 2.0);

    const auto narrow = Kernel::evaluate(
        Kernel::prepare({1.0, 1.0, 1.0, cap}), variance);

    const long double expected_delta =
        normal_density(-0.5L * s) *
        std::log1p(static_cast<long double>(cap) - 1.0L) / s;

    require_close(
        narrow.delta, expected_delta,
        "Adjacent-guarantee delta exceeds the absolute roundoff tolerance.",
        0.0L,
        static_cast<long double>(std::numeric_limits<double>::epsilon()));

    // Check both sides of the transition to the local density expansion.
    for (double midpoint : {-3.0, 0.0, 2.0}) {
        for (double factor : {0.9, 1.1}) {
            const double width =
                factor * 1e-2 / (1.0 + std::abs(midpoint));

            const Parameters p{
                1.0,
                1.0,
                std::exp(
                    s * (midpoint - 0.5 * width + 0.5 * s)),
                std::exp(
                    s * (midpoint + 0.5 * width + 0.5 * s))};

            const long double reference_s =
                std::sqrt(static_cast<long double>(variance));

            const long double lower =
                std::log(
                    static_cast<long double>(p.minimum_guarantee)) /
                    reference_s -
                0.5L * reference_s;

            const long double upper =
                std::log(
                    static_cast<long double>(p.maximum_guarantee)) /
                    reference_s -
                0.5L * reference_s;

            const auto result =
                Kernel::evaluate(Kernel::prepare(p), variance);

            require_close(
                result.delta,
                integrate(normal_density, lower, upper),
                "Narrow-interval transition changed delta.",
                2e-12L, 0.0L);
        }
    }

    const Parameters p{2.3, 1.02, 1.5, 3.1};

    const auto original =
        Kernel::evaluate(Kernel::prepare(p), 0.7);

    const auto scaled = Kernel::evaluate(
        Kernel::prepare({
            37.0 * p.fund_value,
            p.risk_free_growth,
            37.0 * p.minimum_guarantee,
            37.0 * p.maximum_guarantee}),
        0.7);

    require_close(
        scaled.expected_payoff, 37.0 * original.expected_payoff,
        "Payoff scaling is incorrect.");

    require_close(
        scaled.delta, original.delta,
        "Scaling the fund and guarantees changed delta.");
}

void check_parameter_validation()
{
    const double infinity = std::numeric_limits<double>::infinity();
    const double nan = std::numeric_limits<double>::quiet_NaN();
    const double largest = std::numeric_limits<double>::max();

    for (const Parameters& p : {
             Parameters{0.0, 1.0, 0.0, 1.0},
             Parameters{-1.0, 1.0, 0.0, 1.0},
             Parameters{1.0, 0.0, 0.0, 1.0},
             Parameters{1.0, 1.0, 0.0, 0.0},
             Parameters{1.0, 1.0, 2.0, 2.0},
             Parameters{1.0, 1.0, -0.1, 1.0},
             Parameters{1.0, 1.0, 2.0, 1.0},
             Parameters{nan, 1.0, 0.0, 1.0},
             Parameters{1.0, infinity, 0.0, 1.0},
             Parameters{1.0, 1.0, 0.0, infinity},
             Parameters{largest, 2.0, 0.0, 1.0},
             Parameters{
                 std::numeric_limits<double>::denorm_min(),
                 0.5, 0.0, 1.0}}) {
        bool rejected = false;

        try {
            static_cast<void>(Kernel::prepare(p));
        } catch (const std::invalid_argument&) {
            rejected = true;
        }

        require(rejected, "Invalid payoff parameters were accepted.");
    }
}

} // namespace

int main()
{
    try {
        check_against_direct_integration();
        check_deterministic_payoffs();
        check_tails_and_narrow_guarantees();
        check_parameter_validation();

        std::cout << "All min-max payoff checks passed.\n";
        return 0;
    } catch (const std::exception& exception) {
        std::cerr << "Test failure: " << exception.what() << '\n';
        return 1;
    }
}
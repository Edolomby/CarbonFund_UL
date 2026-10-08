#include <carbonfund/mortality/gompertz_makeham.hpp>

#include <cmath>
#include <exception>
#include <iostream>
#include <limits>
#include <stdexcept>

namespace {

using Model = carbonfund::mortality::GompertzMakeham;
using Parameters = carbonfund::mortality::GompertzMakehamParameters;

void require(bool condition, const char* message)
{
    if (!condition) {
        throw std::runtime_error(message);
    }
}

template<class Function>
void require_invalid(Function function)
{
    try {
        function();
    } catch (const std::invalid_argument&) {
        return;
    }

    throw std::runtime_error("Invalid mortality input was accepted.");
}

void check_analytic_case()
{
    const Model model{Parameters{0.01, 80.0, 0.1}};

    // At age m, exp(b*(a-m))=1.
    // Choose the horizon so that exp(b*t)=2.
    const double horizon = std::log(2.0) / 0.1;
    const double expected_hazard = 1.0 + 0.01 * horizon;

    require(std::abs(model.hazard(80.0) - 0.11) < 1e-15,
            "Hazard at the location age is incorrect.");

    require(
        std::abs(model.cumulative_hazard(80.0, horizon) -
                 expected_hazard) < 1e-14,
        "Analytic cumulative hazard is incorrect.");

    require(
        std::abs(model.survival_probability(80.0, horizon) -
                 std::exp(-expected_hazard)) < 1e-14,
        "Analytic survival probability is incorrect.");

    require(
        std::abs(model.death_probability(80.0, horizon) -
                 (1.0 - std::exp(-expected_hazard))) < 1e-14,
        "Analytic death probability is incorrect.");
}

void check_original_formula_and_conditioning()
{
    const Parameters parameters{
        0.0027900,
        79.6921211,
        1.0 / 11.5818911
    };

    const Model model{parameters};

    for (double age : {30.0, 60.0, 90.0}) {
        for (double horizon : {0.1, 1.0, 20.0}) {
            const double original = std::exp(
                -parameters.makeham_intensity * horizon -
                std::exp(
                    (age - parameters.gompertz_location_age) *
                    parameters.gompertz_growth_rate) *
                (std::exp(
                    horizon * parameters.gompertz_growth_rate) - 1.0));

            const double survival =
                model.survival_probability(age, horizon);

            const double death =
                model.death_probability(age, horizon);

            require(std::abs(survival - original) < 1e-14,
                    "Survival differs from the original formula.");

            require(
                survival >= 0.0 && survival <= 1.0 &&
                death >= 0.0 && death <= 1.0 &&
                std::abs(survival + death - 1.0) < 1e-15,
                "Mortality probabilities are inconsistent.");
        }
    }

    // After surviving five years, the current age becomes 65.
    const double combined =
        model.survival_probability(60.0, 12.0);

    const double successive =
        model.survival_probability(60.0, 5.0) *
        model.survival_probability(65.0, 7.0);

    require(std::abs(combined - successive) < 1e-14,
            "Conditional survival over successive intervals is incorrect.");

    require(
        model.death_probability(70.0, 1.0) >
        model.death_probability(60.0, 1.0),
        "Death probability should increase with age.");
}

void check_boundaries()
{
    const Model model{Parameters{0.0, 80.0, 1.0}};

    require(
        model.cumulative_hazard(1000.0, 0.0) == 0.0 &&
        model.survival_probability(1000.0, 0.0) == 1.0 &&
        model.death_probability(1000.0, 0.0) == 0.0,
        "Zero horizon must give survival=1 and death=0.");

    const double tiny_time = 1e-18;
    const double tiny_death =
        model.death_probability(80.0, tiny_time);

    require(
        tiny_death > 0.0 &&
        std::abs(tiny_death / tiny_time - 1.0) < 1e-14,
        "Short-interval death probability lost precision.");

    // The original exponential product gives 0 * infinity.
    // Its mathematical cumulative hazard is approximately one.
    const Model long_interval{Parameters{0.0, 1000.0, 1.0}};

    require(
        std::abs(
            long_interval.survival_probability(0.0, 1000.0) -
            std::exp(-1.0)) < 1e-14,
        "Avoidable intermediate underflow/overflow changed survival.");

    require(
        model.survival_probability(1000.0, 1.0) == 0.0 &&
        model.death_probability(1000.0, 1.0) == 1.0,
        "Large cumulative hazard must saturate probabilities.");
}

void check_validation()
{
    const double nan = std::numeric_limits<double>::quiet_NaN();
    const double infinity = std::numeric_limits<double>::infinity();

    for (const Parameters& parameters : {
             Parameters{-0.01, 80.0, 0.1},
             Parameters{0.01, 80.0, 0.0},
             Parameters{0.01, 80.0, -0.1},
             Parameters{nan, 80.0, 0.1},
             Parameters{0.01, infinity, 0.1},
             Parameters{0.01, 80.0, infinity},
             Parameters{}}) {
        require_invalid([&] {
            const Model invalid{parameters};
        });
    }

    const Model model{Parameters{0.01, 80.0, 0.1}};

    require_invalid([&] {
        static_cast<void>(model.hazard(-1.0));
    });

    require_invalid([&] {
        static_cast<void>(model.hazard(nan));
    });

    require_invalid([&] {
        static_cast<void>(model.survival_probability(60.0, -1.0));
    });

    require_invalid([&] {
        static_cast<void>(model.death_probability(60.0, infinity));
    });
}

} // namespace

int main()
{
    try {
        check_analytic_case();
        check_original_formula_and_conditioning();
        check_boundaries();
        check_validation();

        std::cout << "All Gompertz-Makeham checks passed.\n";
        return 0;
    } catch (const std::exception& exception) {
        std::cerr << "Test failure: " << exception.what() << '\n';
        return 1;
    }
}
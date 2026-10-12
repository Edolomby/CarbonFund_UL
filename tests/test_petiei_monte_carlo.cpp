#include <carbonfund/pricing/petiei_monte_carlo.hpp>
#include <carbonfund/random/simulation_rng.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <exception>
#include <iostream>
#include <limits>
#include <numeric>
#include <span>
#include <stdexcept>
#include <vector>

namespace {

using carbonfund::contracts::CohortBook;
using carbonfund::contracts::PetieiPortfolioView;
using carbonfund::models::CIRNVStepper4;
using carbonfund::models::GeoOUExactStepper4;
using carbonfund::models::OUExactStepper4;
using carbonfund::mortality::GompertzMakeham;
using carbonfund::portfolio::CarbonAllocationParameters4;
using carbonfund::pricing::MonteCarloEstimate;
using carbonfund::pricing::PayoffDelta;
using carbonfund::pricing::PetieiContractEstimate;
using carbonfund::pricing::PetieiMonteCarloParameters;
using carbonfund::pricing::PetieiMonteCarloPricer4;
using carbonfund::pricing::PetieiMonteCarloResult;
using carbonfund::random::SimulationRng;

using Pricer = PetieiMonteCarloPricer4<OUExactStepper4>;

void require(bool condition, const char* message)
{
    if (!condition) {
        throw std::runtime_error(message);
    }
}

void require_close(
    double obtained,
    double expected,
    const char* message,
    double tolerance = 2e-11)
{
    require(
        std::isfinite(obtained) &&
        std::isfinite(expected) &&
        std::abs(obtained - expected) <=
            tolerance * std::max(1.0, std::abs(expected)),
        message
    );
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
        "Expected pricing input rejection was not raised."
    );
}

struct Fixture {
    PetieiMonteCarloParameters pricing;
    Eigen::Vector4d state{0.8, 1.1, 1.4, 0.6};
    OUExactStepper4::Parameters carbon;
    CarbonAllocationParameters4 allocation;

    GompertzMakeham mortality{
        {0.00279, 79.6921211, 1.0 / 11.5818911}
    };

    CohortBook pe{
        {{55.0, 1200.0}, {65.0, 1800.0}}
    };

    CohortBook ti{
        {{60.0, 900.0}, {70.0, 1100.0}}
    };

    CohortBook ei{
        {{62.0, 2500.0}}
    };

    Fixture()
    {
        pricing.fund_value = 1.1;
        pricing.minimum_guarantee = 0.85;
        pricing.maximum_guarantee = 1.5;
        pricing.minimum_guarantee_growth_rate = 0.01;
        pricing.maximum_guarantee_growth_rate = 0.025;
        pricing.endowment_death_benefit_fraction = 0.35;
        pricing.time_to_maturity = 2.0;
        pricing.time_intervals = 3;

        carbon.long_run_mean << 0.9, 1.2, 1.0, 0.7;
        carbon.mean_reversion << 0.4, 0.7, 0.5, 0.3;
        carbon.volatility << 0.6, 0.7, 0.5, 0.4;

        allocation.risk_free_rate = 0.02;
        allocation.risk_aversion = 1.6;
        allocation.carbon_penalty = 1.1;
        allocation.expected_returns << 0.06, 0.08, 0.1, 0.07;

        allocation.covariance <<
            0.04,  0.01,  0.002, 0.006,
            0.01,  0.06,  0.004, 0.005,
            0.002, 0.004, 0.08,  0.01,
            0.006, 0.005, 0.01,  0.09;
    }

    [[nodiscard]] PetieiPortfolioView portfolio() const
    {
        return {
            pe.cohorts(),
            ti.cohorts(),
            ei.cohorts()
        };
    }

    [[nodiscard]] Pricer::Prepared prepare() const
    {
        return Pricer::prepare(
            pricing,
            state,
            carbon,
            allocation,
            portfolio(),
            mortality
        );
    }
};

// Fixed vectors make path accounting and variance checks deterministic.
struct ScriptedNormals {
    std::span<const Eigen::Vector4d> draws;
    std::size_t position = 0;

    void fill_std_normals(Eigen::Vector4d& destination)
    {
        require(
            position < draws.size(),
            "Pricer consumed too many normal vectors."
        );

        destination = draws[position++];
    }
};

std::array<Eigen::Vector4d, 12> normal_vectors()
{
    return {
        Eigen::Vector4d{0.2, -0.4, 0.6, 0.1},
        Eigen::Vector4d{0.7, 0.3, -0.2, -0.5},
        Eigen::Vector4d{-0.1, 0.9, 0.4, 0.2},
        Eigen::Vector4d{-1.2, -0.7, -0.9, -1.4},
        Eigen::Vector4d{-0.5, -0.3, -0.8, -0.2},
        Eigen::Vector4d{-0.9, -0.6, -0.4, -0.7},
        Eigen::Vector4d{1.0, 1.4, 0.8, 1.2},
        Eigen::Vector4d{0.7, 1.1, 1.3, 0.9},
        Eigen::Vector4d{1.2, 0.6, 1.0, 1.5},
        Eigen::Vector4d{-0.4, 0.8, -0.2, 0.6},
        Eigen::Vector4d{0.3, -0.9, 0.5, -0.7},
        Eigen::Vector4d{-0.6, 0.2, -0.8, 0.4}
    };
}

// Reference uses a general Eigen solve instead of the production fixed solver.
double reference_variance(
    const Fixture& fixture,
    const Eigen::Vector4d& state)
{
    const auto& p = fixture.allocation;

    Eigen::Matrix4d matrix =
        p.risk_aversion * p.covariance;

    matrix.diagonal().array() +=
        p.carbon_penalty *
        p.covariance.diagonal().array() *
        state.array().max(0.0);

    const Eigen::Vector4d excess =
        (p.expected_returns.array() -
         p.risk_free_rate).matrix();

    const Eigen::Vector4d weights =
        matrix.llt().solve(excess);

    return weights.dot(p.covariance * weights);
}

// Independent closed-form lognormal reference,
// with long-double arithmetic.
PayoffDelta reference_payoff(
    const Fixture& fixture,
    double time,
    double variance)
{
    const auto& p = fixture.pricing;

    const long double growth = std::exp(
        static_cast<long double>(
            fixture.allocation.risk_free_rate
        ) * time
    );

    const long double forward =
        p.fund_value * growth;

    const long double minimum =
        p.minimum_guarantee == 0.0
            ? 0.0L
            : p.minimum_guarantee * std::exp(
                  static_cast<long double>(
                      p.minimum_guarantee_growth_rate
                  ) * time
              );

    const long double maximum =
        p.maximum_guarantee * std::exp(
            static_cast<long double>(
                p.maximum_guarantee_growth_rate
            ) * time
        );

    if (variance == 0.0) {
        const long double payoff =
            std::clamp(forward, minimum, maximum);

        const double delta =
            forward < minimum || forward > maximum
                ? 0.0
                : static_cast<double>(
                      growth *
                      (
                          forward == minimum ||
                          forward == maximum
                              ? 0.5L
                              : 1.0L
                      )
                  );

        return {
            static_cast<double>(payoff),
            delta
        };
    }

    const long double v = variance;
    const long double s = std::sqrt(v);

    const long double b_lower =
        minimum == 0.0L
            ? -std::numeric_limits<long double>::infinity()
            : (
                  std::log(minimum / forward) + 0.5L * v
              ) / s;

    const long double b_upper =
        (
            std::log(maximum / forward) + 0.5L * v
        ) / s;

    const auto cdf = [](long double z) {
        return 0.5L * std::erfc(
            -z / std::sqrt(2.0L)
        );
    };

    const long double probability =
        cdf(b_upper - s) - cdf(b_lower - s);

    return {
        static_cast<double>(
            minimum * cdf(b_lower) +
            maximum * (1.0L - cdf(b_upper)) +
            forward * probability
        ),
        static_cast<double>(
            growth * probability
        )
    };
}

double reference_maturity_weight(
    const Fixture& fixture,
    std::span<const carbonfund::contracts::Cohort> cohorts)
{
    double result = 0.0;
    const double time = fixture.pricing.time_to_maturity;

    for (const auto& cohort : cohorts) {
        result +=
            cohort.total_notional *
            fixture.mortality.survival_probability(
                cohort.current_age,
                time
            ) *
            std::exp(
                -fixture.allocation.risk_free_rate * time
            );
    }

    return result;
}

double reference_death_weight(
    const Fixture& fixture,
    std::span<const carbonfund::contracts::Cohort> cohorts,
    double time,
    double step,
    bool endpoint)
{
    double result = 0.0;

    for (const auto& cohort : cohorts) {
        result +=
            cohort.total_notional *
            std::exp(
                fixture.mortality.log_death_density(
                    cohort.current_age,
                    time
                )
            ) *
            std::exp(
                -fixture.allocation.risk_free_rate * time
            );
    }

    return result * step * (endpoint ? 0.5 : 1.0);
}

struct PathSample {
    PayoffDelta pe;
    PayoffDelta ti;
    PayoffDelta ei;
};

// Reconstruct the original pricing equations from fixed normal draws.
std::vector<PathSample> reference_paths(
    const Fixture& fixture,
    std::span<const Eigen::Vector4d> draws)
{
    const auto& p = fixture.pricing;
    const std::size_t intervals = p.time_intervals;

    require(
        draws.size() % intervals == 0,
        "Reference paths require complete draw sequences."
    );

    const double step =
        p.time_to_maturity /
        static_cast<double>(intervals);

    PayoffDelta initial{
        p.fund_value,
        1.0
    };

    if (p.fund_value < p.minimum_guarantee) {
        initial = {
            p.minimum_guarantee,
            0.0
        };
    } else if (p.fund_value > p.maximum_guarantee) {
        initial = {
            p.maximum_guarantee,
            0.0
        };
    }

    const double initial_ti = reference_death_weight(
        fixture,
        fixture.ti.cohorts(),
        0.0,
        step,
        true
    );

    const double initial_ei = reference_death_weight(
        fixture,
        fixture.ei.cohorts(),
        0.0,
        step,
        true
    );

    const double pe_maturity = reference_maturity_weight(
        fixture,
        fixture.pe.cohorts()
    );

    const double ei_maturity = reference_maturity_weight(
        fixture,
        fixture.ei.cohorts()
    );

    std::vector<PathSample> result;

    for (
        std::size_t path = 0;
        path < draws.size() / intervals;
        ++path
    ) {
        Eigen::Vector4d state = fixture.state;

        double previous_variance =
            reference_variance(fixture, state);

        double variance = 0.0;

        PayoffDelta ti{
            initial.expected_payoff * initial_ti,
            initial.delta * initial_ti
        };

        PayoffDelta ei_death{
            initial.expected_payoff * initial_ei,
            initial.delta * initial_ei
        };

        PayoffDelta payoff;

        for (
            std::size_t i = 0;
            i < intervals;
            ++i
        ) {
            for (int factor = 0; factor < 4; ++factor) {
                const double kappa =
                    fixture.carbon.mean_reversion[factor];

                const double decay =
                    std::exp(-kappa * step);

                const double noise_time =
                    kappa == 0.0
                        ? step
                        : -std::expm1(-2.0 * kappa * step) /
                          (2.0 * kappa);

                state[factor] =
                    fixture.carbon.long_run_mean[factor] *
                        (1.0 - decay) +
                    decay * state[factor] +
                    fixture.carbon.volatility[factor] *
                        std::sqrt(noise_time) *
                        draws[path * intervals + i][factor];
            }

            const double current_variance =
                reference_variance(fixture, state);

            variance +=
                0.5 * step *
                (previous_variance + current_variance);

            previous_variance = current_variance;

            const double time =
                i + 1 == intervals
                    ? p.time_to_maturity
                    : static_cast<double>(i + 1) * step;

            payoff = reference_payoff(
                fixture,
                time,
                variance
            );

            const double ti_weight = reference_death_weight(
                fixture,
                fixture.ti.cohorts(),
                time,
                step,
                i + 1 == intervals
            );

            const double ei_weight = reference_death_weight(
                fixture,
                fixture.ei.cohorts(),
                time,
                step,
                i + 1 == intervals
            );

            ti.expected_payoff +=
                payoff.expected_payoff * ti_weight;
            ti.delta +=
                payoff.delta * ti_weight;

            ei_death.expected_payoff +=
                payoff.expected_payoff * ei_weight;
            ei_death.delta +=
                payoff.delta * ei_weight;
        }

        result.push_back({
            {
                payoff.expected_payoff * pe_maturity,
                payoff.delta * pe_maturity
            },
            ti,
            {
                p.endowment_death_benefit_fraction *
                    ei_death.expected_payoff +
                    payoff.expected_payoff * ei_maturity,

                p.endowment_death_benefit_fraction *
                    ei_death.delta +
                    payoff.delta * ei_maturity
            }
        });
    }

    return result;
}

void check_statistic(
    const MonteCarloEstimate& actual,
    const std::vector<double>& samples)
{
    const double mean =
        std::accumulate(
            samples.begin(),
            samples.end(),
            0.0
        ) / samples.size();

    double squared_deviations = 0.0;

    for (double sample : samples) {
        squared_deviations +=
            (sample - mean) * (sample - mean);
    }

    const double variance =
        squared_deviations / (samples.size() - 1);

    require_close(
        actual.mean,
        mean,
        "Incorrect mean path contribution."
    );

    require_close(
        actual.sample_variance,
        variance,
        "Incorrect sample variance.",
        2e-10
    );

    require_close(
        actual.standard_error,
        std::sqrt(variance / samples.size()),
        "Incorrect standard error.",
        2e-10
    );
}

void check_against_reference(
    const PetieiMonteCarloResult& actual,
    const std::vector<PathSample>& samples)
{
    require(
        actual.simulations == samples.size(),
        "Incorrect reported simulation count."
    );

    const auto check = [&](
        const PetieiContractEstimate& contract,
        PayoffDelta PathSample::* member
    ) {
        std::vector<double> prices;
        std::vector<double> deltas;

        for (const auto& sample : samples) {
            prices.push_back(
                (sample.*member).expected_payoff
            );
            deltas.push_back(
                (sample.*member).delta
            );
        }

        check_statistic(contract.price, prices);
        check_statistic(contract.delta, deltas);
    };

    check(
        actual.pure_endowment,
        &PathSample::pe
    );

    check(
        actual.term_insurance,
        &PathSample::ti
    );

    check(
        actual.endowment_insurance,
        &PathSample::ei
    );
}

void check_fixed_paths_and_delta()
{
    const Fixture fixture;
    const auto draws = normal_vectors();
    const auto prepared = fixture.prepare();

    ScriptedNormals stream{draws};

    const auto actual =
        Pricer::evaluate(prepared, 4, stream);

    require(
        stream.position == draws.size(),
        "Incorrect number of carbon transitions."
    );

    check_against_reference(
        actual,
        reference_paths(fixture, draws)
    );

    require(
        actual.endowment_insurance.price.sample_variance > 0.0,
        "Changing carbon paths must produce a nonzero sample variance."
    );

    // Common draws isolate the fund derivative.
    // Guarantees remain fixed.
    Fixture down = fixture;
    Fixture up = fixture;

    const double bump =
        1e-5 * fixture.pricing.fund_value;

    down.pricing.fund_value -= bump;
    up.pricing.fund_value += bump;

    ScriptedNormals down_stream{draws};
    ScriptedNormals up_stream{draws};

    const auto down_value =
        Pricer::evaluate(down.prepare(), 4, down_stream);

    const auto up_value =
        Pricer::evaluate(up.prepare(), 4, up_stream);

    require_close(
        actual.pure_endowment.delta.mean,
        (
            up_value.pure_endowment.price.mean -
            down_value.pure_endowment.price.mean
        ) / (2.0 * bump),
        "PE delta differs from a common-path finite difference.",
        2e-7
    );

    require_close(
        actual.term_insurance.delta.mean,
        (
            up_value.term_insurance.price.mean -
            down_value.term_insurance.price.mean
        ) / (2.0 * bump),
        "TI delta differs from a common-path finite difference.",
        2e-7
    );

    require_close(
        actual.endowment_insurance.delta.mean,
        (
            up_value.endowment_insurance.price.mean -
            down_value.endowment_insurance.price.mean
        ) / (2.0 * bump),
        "EI delta differs from a common-path finite difference.",
        2e-7
    );
}

void check_deterministic_paths_and_initial_endpoint()
{
    Fixture fixture;

    fixture.carbon.mean_reversion.setZero();
    fixture.carbon.volatility.setZero();

    fixture.pricing.fund_value = 1.0;
    fixture.pricing.minimum_guarantee = 1.0;
    fixture.pricing.maximum_guarantee = 1.0;
    fixture.pricing.minimum_guarantee_growth_rate = 0.0;
    fixture.pricing.maximum_guarantee_growth_rate = 0.08;

    const auto draws = normal_vectors();
    ScriptedNormals stream{draws};

    const auto actual =
        Pricer::evaluate(fixture.prepare(), 4, stream);

    check_against_reference(
        actual,
        reference_paths(fixture, draws)
    );

    require(
        actual.pure_endowment.price.sample_variance == 0.0 &&
        actual.term_insurance.delta.standard_error == 0.0 &&
        actual.endowment_insurance.price.standard_error == 0.0,
        "Deterministic carbon must have zero Monte Carlo error."
    );

    // No risky investment:
    // the zero-integrated-variance kernel branch is used.
    fixture.pricing.minimum_guarantee = 0.5;
    fixture.pricing.maximum_guarantee = 2.0;

    fixture.allocation.expected_returns.setConstant(
        fixture.allocation.risk_free_rate
    );

    ScriptedNormals zero_stream{draws};

    check_against_reference(
        Pricer::evaluate(
            fixture.prepare(),
            4,
            zero_stream
        ),
        reference_paths(fixture, draws)
    );
}

void check_rng_and_all_models()
{
    const Fixture fixture;
    const auto prepared = fixture.prepare();

    SimulationRng stream{20261011};
    SimulationRng repeat{20261011};
    SimulationRng control{20261011};

    const auto first =
        Pricer::evaluate(prepared, 16, stream);

    const auto second =
        Pricer::evaluate(prepared, 16, repeat);

    require(
        first.pure_endowment.price.mean ==
            second.pure_endowment.price.mean &&
        first.term_insurance.delta.sample_variance ==
            second.term_insurance.delta.sample_variance &&
        first.endowment_insurance.price.standard_error ==
            second.endowment_insurance.price.standard_error,
        "The same seed and prepared inputs must reproduce results."
    );

    Eigen::Vector4d draws;

    for (
        std::size_t i = 0;
        i < 16 * fixture.pricing.time_intervals;
        ++i
    ) {
        control.fill_std_normals(draws);
    }

    require(
        stream.std_normal() == control.std_normal(),
        "Pricer changed the expected random-stream advancement."
    );

    GeoOUExactStepper4::Parameters geo;

    geo.long_run_log_mean =
        fixture.state.array().log().matrix();
    geo.mean_reversion.setConstant(0.5);
    geo.volatility.setConstant(0.2);

    using GeoPricer =
        PetieiMonteCarloPricer4<GeoOUExactStepper4>;

    const auto geo_prepared = GeoPricer::prepare(
        fixture.pricing,
        fixture.state,
        geo,
        fixture.allocation,
        fixture.portfolio(),
        fixture.mortality
    );

    const auto geo_result =
        GeoPricer::evaluate(geo_prepared, 8, stream);

    require(
        std::isfinite(geo_result.pure_endowment.price.mean),
        "GeoOU pricer failed."
    );

    CIRNVStepper4::Parameters cir;

    cir.long_run_mean = fixture.state;
    cir.mean_reversion.setConstant(0.5);
    cir.volatility.setConstant(0.2);

    using CirPricer =
        PetieiMonteCarloPricer4<CIRNVStepper4>;

    const auto cir_prepared = CirPricer::prepare(
        fixture.pricing,
        fixture.state,
        cir,
        fixture.allocation,
        fixture.portfolio(),
        fixture.mortality
    );

    const auto cir_result =
        CirPricer::evaluate(cir_prepared, 8, stream);

    require(
        std::isfinite(cir_result.endowment_insurance.delta.mean),
        "CIR pricer failed."
    );

    const auto empty_prepared = Pricer::prepare(
        fixture.pricing,
        fixture.state,
        fixture.carbon,
        fixture.allocation,
        {},
        fixture.mortality
    );

    const auto empty =
        Pricer::evaluate(empty_prepared, 2, stream);

    require(
        empty.pure_endowment.price.mean == 0.0 &&
        empty.term_insurance.delta.mean == 0.0 &&
        empty.endowment_insurance.price.standard_error == 0.0,
        "Empty portfolios must have zero values."
    );
}

void check_input_validation()
{
    Fixture fixture;
    const auto prepared = fixture.prepare();
    SimulationRng stream{1};

    for (
        std::size_t simulations :
        {std::size_t{0}, std::size_t{1}}
    ) {
        require_throws<std::invalid_argument>([&] {
            static_cast<void>(
                Pricer::evaluate(
                    prepared,
                    simulations,
                    stream
                )
            );
        });
    }

    const auto rejects = [](Fixture invalid) {
        require_throws<std::invalid_argument>([&] {
            static_cast<void>(invalid.prepare());
        });
    };

    fixture.pricing.time_to_maturity = 0.0;
    rejects(fixture);

    fixture = Fixture{};
    fixture.pricing.time_intervals = 0;
    rejects(fixture);

    fixture = Fixture{};
    fixture.pricing.fund_value = 0.0;
    rejects(fixture);

    fixture = Fixture{};
    fixture.pricing.endowment_death_benefit_fraction = -0.1;
    rejects(fixture);

    fixture = Fixture{};
    fixture.pricing.maximum_guarantee_growth_rate =
        std::numeric_limits<double>::quiet_NaN();
    rejects(fixture);

    fixture = Fixture{};
    fixture.pricing.minimum_guarantee = 1.0;
    fixture.pricing.maximum_guarantee = 1.0;
    fixture.pricing.minimum_guarantee_growth_rate = 0.0;
    fixture.pricing.maximum_guarantee_growth_rate = 0.0;
    rejects(fixture);

    fixture = Fixture{};
    fixture.pricing.minimum_guarantee_growth_rate = 0.5;
    rejects(fixture);

    fixture = Fixture{};
    fixture.state[0] =
        std::numeric_limits<double>::infinity();
    rejects(fixture);

    fixture = Fixture{};
    fixture.state[0] = 0.0;

    GeoOUExactStepper4::Parameters geo;

    require_throws<std::invalid_argument>([&] {
        static_cast<void>(
            PetieiMonteCarloPricer4<GeoOUExactStepper4>::prepare(
                fixture.pricing,
                fixture.state,
                geo,
                fixture.allocation,
                fixture.portfolio(),
                fixture.mortality
            )
        );
    });

    fixture.state[0] = -1.0;

    CIRNVStepper4::Parameters cir;

    require_throws<std::invalid_argument>([&] {
        static_cast<void>(
            PetieiMonteCarloPricer4<CIRNVStepper4>::prepare(
                fixture.pricing,
                fixture.state,
                cir,
                fixture.allocation,
                fixture.portfolio(),
                fixture.mortality
            )
        );
    });
}

} // namespace

int main()
{
    try {
        check_fixed_paths_and_delta();
        check_deterministic_paths_and_initial_endpoint();
        check_rng_and_all_models();
        check_input_validation();

        std::cout <<
            "All serial PETIEI Monte Carlo checks passed.\n";

        return 0;
    } catch (const std::exception& exception) {
        std::cerr <<
            "Test failure: " << exception.what() << '\n';

        return 1;
    }
}
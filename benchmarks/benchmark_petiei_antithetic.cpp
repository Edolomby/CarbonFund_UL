#include <carbonfund/pricing/petiei_monte_carlo.hpp>
#include <carbonfund/random/simulation_rng.hpp>

#include <algorithm>
#include <charconv>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <ctime>
#include <exception>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {

using carbonfund::pricing::MonteCarloEstimate;
using carbonfund::pricing::PetieiMonteCarloResult;
using carbonfund::pricing::PetieiSampling;

// Keep every valuation observable, including the discarded warm-up runs.
volatile double benchmark_checksum = 0.0;

struct Measurement {
    PetieiMonteCarloResult result;
    double cpu_seconds;
};

template<class Integer>
Integer parse_integer(const char* argument, const char* name)
{
    const std::string_view text{argument};
    Integer value{};
    const auto conversion = std::from_chars(text.data(), text.data() + text.size(), value);
    if (conversion.ec != std::errc{} || conversion.ptr != text.data() + text.size()) {
        throw std::invalid_argument(std::string{"Invalid "} + name + ": " + argument);
    }
    return value;
}

double median(std::vector<double> values)
{
    std::sort(values.begin(), values.end());
    const std::size_t middle = values.size() / 2;
    return values.size() % 2 == 0
        ? 0.5 * values[middle - 1] + 0.5 * values[middle]
        : values[middle];
}

template<class Pricer>
Measurement measure(
    const typename Pricer::Prepared& prepared,
    std::size_t simulations,
    std::uint64_t seed,
    PetieiSampling sampling)
{
    // Each method starts from its own freshly seeded stream. Preparation and
    // RNG construction are outside the measured simulation interval.
    carbonfund::random::SimulationRng rng{seed};
    const std::clock_t begin = std::clock();
    const auto result = Pricer::evaluate(prepared, simulations, rng, sampling);
    benchmark_checksum = result.pure_endowment.price.mean + result.pure_endowment.delta.mean +
        result.term_insurance.price.mean + result.term_insurance.delta.mean +
        result.endowment_insurance.price.mean + result.endowment_insurance.delta.mean;
    const std::clock_t end = std::clock();
    if (begin == static_cast<std::clock_t>(-1) || end == static_cast<std::clock_t>(-1)) {
        throw std::runtime_error("Process CPU timing is unavailable.");
    }
    return {result, static_cast<double>(end - begin) / CLOCKS_PER_SEC};
}

void print_estimate(
    const char* quantity,
    const MonteCarloEstimate& plain,
    const MonteCarloEstimate& antithetic,
    double plain_cpu,
    double antithetic_cpu)
{
    std::cout << std::left << std::setw(10) << quantity << std::right
              << std::setw(16) << plain.mean
              << std::setw(16) << antithetic.mean
              << std::setw(14) << plain.standard_error
              << std::setw(14) << antithetic.standard_error;
    const double denominator = antithetic_cpu *
        antithetic.standard_error * antithetic.standard_error;
    if (denominator > 0.0 && plain_cpu > 0.0) {
        const double efficiency =
            plain_cpu * plain.standard_error * plain.standard_error / denominator;
        std::cout << std::setw(13) << efficiency;
    } else {
        std::cout << std::setw(13) << "n/a";
    }
    std::cout << '\n';
}

template<class Stepper>
void benchmark_model(
    const char* model_name,
    const typename Stepper::Parameters& carbon,
    std::size_t simulations,
    std::size_t intervals,
    std::size_t repetitions,
    std::uint64_t seed)
{
    using namespace carbonfund;
    using Pricer = pricing::PetieiMonteCarloPricer4<Stepper>;
    const contracts::CohortBook pe{{{60.0, 100000.0}}};
    const contracts::CohortBook ti{{{60.0, 100000.0}}};
    const contracts::CohortBook ei{{{60.0, 100000.0}}};
    const contracts::PetieiPortfolioView portfolio{pe.cohorts(), ti.cohorts(), ei.cohorts()};
    const mortality::GompertzMakeham mortality{{0.00279, 79.6921211, 1.0 / 11.5818911}};

    pricing::PetieiMonteCarloParameters contract;
    contract.fund_value = 1.0;
    contract.minimum_guarantee = 1.0;
    contract.maximum_guarantee = 1.0;
    contract.minimum_guarantee_growth_rate = 0.0;
    contract.maximum_guarantee_growth_rate = 0.05;
    contract.endowment_death_benefit_fraction = 1.0;
    contract.time_to_maturity = 5.0;
    contract.time_intervals = intervals;
    const Eigen::Vector4d current_carbon{1.0, 1.2, 0.8, 1.1};

    portfolio::CarbonAllocationParameters4 allocation;
    allocation.risk_free_rate = 0.03;
    allocation.risk_aversion = 2.0;
    allocation.carbon_penalty = 1.0;
    allocation.expected_returns << 0.06, 0.08, 0.07, 0.09;
    allocation.covariance = Eigen::Vector4d{0.04, 0.06, 0.05, 0.08}.asDiagonal();

    const auto prepared = Pricer::prepare(
        contract, current_carbon, carbon, allocation, portfolio, mortality);

    // Warm both implementations before measuring. These samples are discarded.
    const std::size_t warmup = std::min(simulations, std::size_t{1000});
    static_cast<void>(measure<Pricer>(prepared, warmup, seed, PetieiSampling::independent));
    static_cast<void>(measure<Pricer>(prepared, warmup, seed, PetieiSampling::antithetic));

    std::vector<double> plain_times;
    std::vector<double> antithetic_times;
    plain_times.reserve(repetitions);
    antithetic_times.reserve(repetitions);
    PetieiMonteCarloResult plain_result;
    PetieiMonteCarloResult antithetic_result;

    for (std::size_t repetition = 0; repetition < repetitions; ++repetition) {
        const std::uint64_t run_seed = seed + static_cast<std::uint64_t>(repetition);
        Measurement plain;
        Measurement antithetic;
        // Alternating the order limits systematic cache/CPU-frequency effects.
        if (repetition % 2 == 0) {
            plain = measure<Pricer>(prepared, simulations, run_seed, PetieiSampling::independent);
            antithetic = measure<Pricer>(prepared, simulations, run_seed, PetieiSampling::antithetic);
        } else {
            antithetic = measure<Pricer>(prepared, simulations, run_seed, PetieiSampling::antithetic);
            plain = measure<Pricer>(prepared, simulations, run_seed, PetieiSampling::independent);
        }
        if (plain.result.simulations != simulations || antithetic.result.simulations != simulations ||
            plain.result.independent_samples != simulations ||
            antithetic.result.independent_samples != simulations / 2) {
            throw std::runtime_error("Benchmark methods evaluated different trajectory counts.");
        }
        plain_times.push_back(plain.cpu_seconds);
        antithetic_times.push_back(antithetic.cpu_seconds);
        // Report statistics from the first seed; timing uses all repetitions.
        if (repetition == 0) {
            plain_result = plain.result;
            antithetic_result = antithetic.result;
        }
    }

    const double plain_cpu = median(plain_times);
    const double antithetic_cpu = median(antithetic_times);
    std::cout << '\n' << model_name
              << ": trajectories=" << simulations << " for both methods"
              << ", independent observations=" << simulations
              << " (plain), " << simulations / 2 << " (antithetic)\n"
              << "Median process CPU seconds: plain=" << plain_cpu
              << ", antithetic=" << antithetic_cpu;
    if (antithetic_cpu > 0.0) {
        std::cout << ", CPU speedup=" << plain_cpu / antithetic_cpu << "x";
    }
    std::cout << '\n' << std::left << std::setw(10) << "Quantity" << std::right
              << std::setw(16) << "Plain mean" << std::setw(16) << "Anti mean"
              << std::setw(14) << "Plain SE" << std::setw(14) << "Anti SE"
              << std::setw(13) << "Efficiency" << '\n';
    const auto print = [&](const char* price_name, const char* delta_name,
                           const pricing::PetieiContractEstimate& plain,
                           const pricing::PetieiContractEstimate& antithetic) {
        print_estimate(price_name, plain.price, antithetic.price, plain_cpu, antithetic_cpu);
        print_estimate(delta_name, plain.delta, antithetic.delta, plain_cpu, antithetic_cpu);
    };
    print("PE price", "PE delta", plain_result.pure_endowment, antithetic_result.pure_endowment);
    print("TI price", "TI delta", plain_result.term_insurance, antithetic_result.term_insurance);
    print("EI price", "EI delta", plain_result.endowment_insurance, antithetic_result.endowment_insurance);
}

} // namespace

int main(int argc, char** argv)
{
    try {
        if (argc > 5) {
            throw std::invalid_argument(
                "Usage: benchmark_petiei_antithetic [trajectories [intervals [repetitions [seed]]]]");
        }
        const std::size_t requested_simulations = argc > 1
            ? parse_integer<std::size_t>(argv[1], "trajectory count") : 20000;
        const std::size_t intervals = argc > 2
            ? parse_integer<std::size_t>(argv[2], "interval count") : 60;
        const std::size_t repetitions = argc > 3
            ? parse_integer<std::size_t>(argv[3], "repetition count") : 5;
        const std::uint64_t seed = argc > 4
            ? parse_integer<std::uint64_t>(argv[4], "seed") : 20261013;
        if (requested_simulations < 3) {
            throw std::invalid_argument(
                "Request at least three trajectories, rounded to at least two complete pairs.");
        }
        if (requested_simulations == std::numeric_limits<std::size_t>::max()) {
            throw std::overflow_error("Rounded benchmark trajectory count overflows.");
        }
        // For an odd request, give BOTH methods the same rounded even budget.
        const std::size_t simulations =
            2 * (requested_simulations / 2 + requested_simulations % 2);
        if (intervals == 0 || repetitions == 0) {
            throw std::invalid_argument("Intervals and repetitions must be positive.");
        }
        if (repetitions - 1 > std::numeric_limits<std::uint64_t>::max() - seed) {
            throw std::overflow_error("Timing repetition seeds would overflow.");
        }

        std::cout << std::fixed << std::setprecision(6);
#ifdef NDEBUG
        std::cout << "Build: NDEBUG defined; use a Release build for CPU comparisons.\n";
#else
        std::cout << "Build: Debug; recompile in Release before interpreting CPU times.\n";
#endif
        std::cout << "Requested trajectories=" << requested_simulations
                  << ", common actual trajectory budget=" << simulations << '\n'
                  << "Grid intervals=" << intervals << ", timing repetitions=" << repetitions
                  << ", reported price/delta seed=" << seed << '\n'
                  << "Plain = independent sampling. Each timing repetition uses seed + repetition.\n"
                  << "Efficiency = (plain CPU * plain SE^2) / (anti CPU * anti SE^2); >1 favours antithetics.\n"
                  << "Means need not match exactly. No assumption of independent errors between methods is made.\n";

        const Eigen::Vector4d state{1.0, 1.2, 0.8, 1.1};
        carbonfund::models::OUParameters4 ou;
        ou.long_run_mean = state;
        ou.mean_reversion.setConstant(0.5);
        ou.volatility.setConstant(0.2);
        benchmark_model<carbonfund::models::OUExactStepper4>(
            "OU", ou, simulations, intervals, repetitions, seed);

        carbonfund::models::GeoOUParameters4 geo;
        geo.long_run_log_mean = state.array().log().matrix();
        geo.mean_reversion.setConstant(0.5);
        geo.volatility.setConstant(0.2);
        benchmark_model<carbonfund::models::GeoOUExactStepper4>(
            "GeoOU", geo, simulations, intervals, repetitions, seed);

        carbonfund::models::CIRParameters4 cir;
        cir.long_run_mean = state;
        cir.mean_reversion.setConstant(0.5);
        cir.volatility.setConstant(0.2);
        benchmark_model<carbonfund::models::CIRNVStepper4>(
            "CIR NV", cir, simulations, intervals, repetitions, seed);
        return 0;
    } catch (const std::exception& exception) {
        std::cerr << "Benchmark failure: " << exception.what() << '\n';
        return 1;
    }
}

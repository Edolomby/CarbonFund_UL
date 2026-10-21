// Compile Boost.JSON's implementation in this translation unit only.
#include <boost/json/src.hpp>
#include <carbonfund/hedging/petiei_hedging.hpp>
#include <carbonfund/io/hedging_csv.hpp>
#include <carbonfund/random/simulation_rng.hpp>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <ctime>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <limits>
#include <mutex>
#include <set>
#include <string>
#include <string_view>
#include <type_traits>
#include <unordered_set>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace {
namespace cf = carbonfund;
namespace js = boost::json;
using Object = js::object;
using Value = js::value;
using Prm = cf::pricing::PetieiMonteCarloParameters;
using Result = cf::pricing::PetieiMonteCarloResult;

[[noreturn]] void fail(const std::string& message)
{ throw std::invalid_argument(message); }

const Object& object(const Value& v, std::string_view name)
{
    if (!v.is_object()) fail(std::string(name) + " must be an object.");
    return v.as_object();
}

void keys(const Object& o, std::initializer_list<std::string_view> allowed,
          std::string_view name)
{
    for (const auto& field : o) {
        const std::string_view key(field.key().data(), field.key().size());
        if (std::find(allowed.begin(), allowed.end(), key) == allowed.end())
            fail(std::string(name) + ": unknown field '" + std::string(key) + "'.");
    }
}

const Value& field(const Object& o, std::string_view name)
{
    auto p = o.if_contains(js::string_view(name.data(), name.size()));
    if (!p) fail("Missing input: " + std::string(name));
    return *p;
}

std::string string(const Value& v)
{
    if (!v.is_string()) fail("Expected a JSON string.");
    return std::string(v.as_string().data(), v.as_string().size());
}

double number(const Value& v)
{
    if (!v.is_number()) fail("Expected a finite JSON number.");
    const double x = v.to_number<double>();
    if (!std::isfinite(x)) fail("Expected a finite JSON number.");
    return x;
}

double number(const Object& o, std::string_view name)
{ return number(field(o, name)); }

std::uint64_t integer(const Value& v)
{
    if (v.is_uint64()) return v.as_uint64();
    if (v.is_int64() && v.as_int64() >= 0)
        return static_cast<std::uint64_t>(v.as_int64());
    fail("Counts and seeds must be nonnegative JSON integers.");
}

std::size_t count(const Object& o, std::string_view name, std::size_t minimum)
{
    const auto n = integer(field(o, name));
    if (n < minimum || n >= std::numeric_limits<std::size_t>::max())
        fail(std::string(name) + " is outside its supported range.");
    return static_cast<std::size_t>(n);
}

Eigen::Vector4d vector4(const Value& v)
{
    if (!v.is_array() || v.as_array().size() != 4)
        fail("A parameter vector must contain exactly four numbers.");
    Eigen::Vector4d x;
    for (int i = 0; i < 4; ++i) x[i] = number(v.as_array()[i]);
    return x;
}

Eigen::Matrix4d matrix4(const Value& v)
{
    if (!v.is_array() || v.as_array().size() != 4)
        fail("The correlation matrix must have four rows.");
    Eigen::Matrix4d x;
    for (int i = 0; i < 4; ++i)
        x.row(i) = vector4(v.as_array()[i]).transpose();
    return x;
}

int type_index(const std::string& type)
{
    if (type == "PE") return 0;
    if (type == "TI") return 1;
    if (type == "EI") return 2;
    fail("Contract type must be PE, TI or EI.");
}

std::uint64_t mix(std::uint64_t x)
{
    x += UINT64_C(0x9e3779b97f4a7c15);
    x = (x ^ (x >> 30)) * UINT64_C(0xbf58476d1ce4e5b9);
    x = (x ^ (x >> 27)) * UINT64_C(0x94d049bb133111eb);
    return x ^ (x >> 31);
}

std::uint64_t repeat_seed(std::uint64_t seed, std::size_t repetition)
{ return repetition == 0 ? seed : mix(seed ^ static_cast<std::uint64_t>(repetition)); }

struct ScenarioSeeds {
    std::uint64_t carbon, financial, mortality, valuation_carbon, valuation_financial;
};

// Assign streams in scenario order before workers start. This makes CSV rows
// independent of worker scheduling, thread count and the total scenario count.
std::vector<ScenarioSeeds> scenario_seeds(
    std::size_t scenarios, const std::array<std::uint64_t, 5>& bases,
    std::uint64_t initial_carbon, std::uint64_t initial_financial)
{
    if (scenarios > (std::numeric_limits<std::size_t>::max() - 2) / 5)
        fail("Scenario count is too large for the random-stream schedule.");
    constexpr std::array<std::uint64_t, 5> tags{
        UINT64_C(0x434152424f4e0001), UINT64_C(0x46494e414e430002),
        UINT64_C(0x4d4f5254414c0003), UINT64_C(0x56414c4341520004),
        UINT64_C(0x56414c46494e0005)};
    std::unordered_set<std::uint64_t> used{initial_carbon, initial_financial};
    used.reserve(5 * scenarios + 2);
    std::vector<ScenarioSeeds> result;
    result.reserve(scenarios);
    for (std::size_t j = 0; j < scenarios; ++j) {
        std::array<std::uint64_t, 5> seeds;
        for (std::size_t role = 0; role < seeds.size(); ++role) {
            auto seed = mix(bases[role] ^ mix(static_cast<std::uint64_t>(j) ^ tags[role]));
            while (!used.insert(seed).second) ++seed;
            seeds[role] = seed;
        }
        result.push_back({seeds[0], seeds[1], seeds[2], seeds[3], seeds[4]});
    }
    return result;
}

int hedging_threads(const Object& settings, std::size_t scenarios, int& requested)
{
    const auto* value = settings.if_contains("threads");
    const auto input = value ? integer(*value) : UINT64_C(1);
    if (input > static_cast<std::uint64_t>(std::numeric_limits<int>::max()))
        fail("hedging.threads must be between 0 and INT_MAX.");
    requested = static_cast<int>(input);
#ifdef _OPENMP
    const int limit = requested == 0 ? std::max(1, omp_get_max_threads()) : requested;
#else
    if (requested > 1)
        fail("Multiple hedging threads require OpenMP. Rebuild with "
             "-DCARBONFUND_ENABLE_OPENMP=ON or select threads=1.");
    const int limit = 1;
#endif
    return static_cast<int>(std::min(scenarios, static_cast<std::size_t>(limit)));
}

struct Mode {
    cf::pricing::PetieiEstimator estimator;
    cf::pricing::PetieiSampling sampling;
};

Mode mode(const std::string& name)
{
    using E = cf::pricing::PetieiEstimator;
    using S = cf::pricing::PetieiSampling;
    if (name == "standard") return {E::standard, S::independent};
    if (name == "standard_antithetic") return {E::standard, S::antithetic};
    if (name == "conditioned") return {E::conditioned, S::independent};
    if (name == "conditioned_antithetic") return {E::conditioned, S::antithetic};
    fail("Unknown estimator configuration: " + name);
}

struct Valuation {
    std::size_t paths, intervals;
    std::string configuration;
    std::uint64_t carbon_seed, financial_seed;
    std::size_t minimum_intervals = 1;
};

Valuation valuation(const Value& value, bool revaluation = false)
{
    const auto& o = object(value, "valuation");
    if (revaluation)
        keys(o, {"paths", "intervals", "minimum_intervals", "configuration",
                 "carbon_seed", "financial_seed"}, "revaluation");
    else
        keys(o, {"paths", "intervals", "configuration",
                 "carbon_seed", "financial_seed"}, "valuation");
    Valuation s{count(o, "paths", 2), count(o, "intervals", 1),
        string(field(o, "configuration")),
        integer(field(o, "carbon_seed")), integer(field(o, "financial_seed"))};
    if (revaluation && o.if_contains("minimum_intervals"))
        s.minimum_intervals = count(o, "minimum_intervals", 1);
    if (s.minimum_intervals > s.intervals)
        fail("revaluation.minimum_intervals must not exceed revaluation.intervals.");
    if (mode(s.configuration).sampling == cf::pricing::PetieiSampling::antithetic &&
        s.paths < 3) fail("Antithetic sampling requires at least two complete pairs.");
    if (s.carbon_seed == s.financial_seed) fail("Carbon and financial seeds must differ.");
    return s;
}

struct Inputs {
    Value config;
    std::string experiment, carbon_model;
    Prm contract;
    Eigen::Vector4d initial_carbon;
    cf::portfolio::CarbonAllocationParameters4 allocation;
    cf::mortality::GompertzMakeham mortality;
    std::vector<cf::contracts::Policyholder> people;

    static cf::mortality::GompertzMakeham parse_mortality(const Object& root)
    {
        const auto& o = object(field(root, "mortality"), "mortality");
        keys(o, {"xi", "m", "b"}, "mortality");
        const double b = number(o, "b");
        if (b <= 0) fail("Mortality scale b must be positive.");
        return cf::mortality::GompertzMakeham{{number(o, "xi"), number(o, "m"), 1.0 / b}};
    }

    explicit Inputs(Value raw)
        : config(std::move(raw)), mortality(parse_mortality(object(config, "configuration")))
    {
        const auto& root = config.as_object();
        keys(root, {"schema_version", "experiment", "market", "carbon", "fund",
                    "contract", "mortality", "population", "pricing", "hedging",
                    "analysis", "notes"}, "configuration");
        if (integer(field(root, "schema_version")) != 1) fail("Unsupported schema_version.");
        experiment = string(field(root, "experiment"));
        if (experiment != "pricing" && experiment != "hedging")
            fail("experiment must be pricing or hedging.");
        const auto& m = object(field(root, "market"), "market");
        keys(m, {"risk_free_rate", "expected_returns", "volatilities", "correlations"}, "market");
        allocation.risk_free_rate = number(m, "risk_free_rate");
        allocation.expected_returns = vector4(field(m, "expected_returns"));
        const auto vol = vector4(field(m, "volatilities"));
        const auto corr = matrix4(field(m, "correlations"));
        if ((vol.array() <= 0).any() ||
            !corr.isApprox(corr.transpose(), 1e-12) ||
            (corr.diagonal().array() - 1.0).abs().maxCoeff() > 1e-12 ||
            (corr.array().abs() > 1.0 + 1e-12).any())
            fail("Need positive volatilities and a symmetric, unit-diagonal correlation matrix.");
        allocation.covariance = vol.asDiagonal() * corr * vol.asDiagonal();
        const auto& f = object(field(root, "fund"), "fund");
        keys(f, {"risk_aversion", "carbon_penalty"}, "fund");
        allocation.risk_aversion = number(f, "risk_aversion");
        allocation.carbon_penalty = number(f, "carbon_penalty");
        static_cast<void>(cf::portfolio::CarbonAllocation4::prepare(allocation));
        const auto& c = object(field(root, "carbon"), "carbon");
        carbon_model = string(field(c, "model"));
        initial_carbon = vector4(field(c, "initial"));
        const auto& t = object(field(root, "contract"), "contract");
        keys(t, {"fund_value", "maturity", "minimum_guarantee", "maximum_guarantee",
                 "minimum_guarantee_growth_rate", "maximum_guarantee_growth_rate",
                 "endowment_death_benefit_fraction"}, "contract");
        contract.fund_value = number(t, "fund_value");
        contract.time_to_maturity = number(t, "maturity");
        contract.minimum_guarantee = number(t, "minimum_guarantee");
        contract.maximum_guarantee = number(t, "maximum_guarantee");
        contract.minimum_guarantee_growth_rate = number(t, "minimum_guarantee_growth_rate");
        contract.maximum_guarantee_growth_rate = number(t, "maximum_guarantee_growth_rate");
        contract.endowment_death_benefit_fraction = number(t, "endowment_death_benefit_fraction");
        const auto& population = field(root, "population");
        if (!population.is_array()) fail("population must be an array of individuals.");
        for (const auto& item : population.as_array()) {
            const auto& p = object(item, "policyholder");
            keys(p, {"type", "age", "notional"}, "policyholder");
            people.push_back({number(p, "age"), number(p, "notional"),
                static_cast<cf::contracts::PolicyType>(type_index(string(field(p, "type"))))});
        }
        static_cast<void>(cf::contracts::PolicyholderPopulation{people});
    }
};

template<class Stepper>
typename Stepper::Parameters carbon_parameters(const Inputs& input)
{
    const auto& c = field(input.config.as_object(), "carbon").as_object();
    typename Stepper::Parameters p;
    if constexpr (std::is_same_v<Stepper, cf::models::GeoOUExactStepper4>) {
        keys(c, {"model", "initial", "long_run_log_mean", "mean_reversion", "volatility"}, "carbon");
        p.long_run_log_mean = vector4(field(c, "long_run_log_mean"));
    } else {
        keys(c, {"model", "initial", "long_run_mean", "mean_reversion", "volatility"}, "carbon");
        p.long_run_mean = vector4(field(c, "long_run_mean"));
    }
    p.mean_reversion = vector4(field(c, "mean_reversion"));
    p.volatility = vector4(field(c, "volatility"));
    return p;
}

void write_json(const std::filesystem::path& path, const Value& value)
{
    std::ofstream file;
    file.exceptions(std::ios::badbit | std::ios::failbit);
    file.open(path);
    file << js::serialize(value) << '\n';
}

double cpu_since(std::clock_t start)
{
    const auto end = std::clock();
    if (start == static_cast<std::clock_t>(-1) || end == static_cast<std::clock_t>(-1))
        throw std::runtime_error("CPU clock unavailable.");
    return static_cast<double>(end - start) / CLOCKS_PER_SEC;
}

const cf::pricing::PetieiContractEstimate& estimate(const Result& r, int contract)
{
    if (contract == 0) return r.pure_endowment;
    if (contract == 1) return r.term_insurance;
    return r.endowment_insurance;
}

Object moment(const cf::pricing::MonteCarloEstimate& e)
{
    if (!std::isfinite(e.mean) || !std::isfinite(e.sample_variance) ||
        !std::isfinite(e.standard_error) || e.sample_variance < 0)
        fail("Nonfinite simulation estimate or variance.");
    return {{"mean", e.mean}, {"sample_variance", e.sample_variance},
            {"standard_error", e.standard_error}};
}

template<class Stepper>
Object pricing(const Inputs& in, const typename Stepper::Parameters& carbon,
               const std::filesystem::path& output)
{
    const auto& s = object(field(in.config.as_object(), "pricing"), "pricing");
    keys(s, {"contract", "paths", "intervals", "configurations",
             "repetitions", "carbon_seed", "financial_seed"}, "pricing");
    const int selected = type_index(string(field(s, "contract")));
    for (const auto& p : in.people)
        if (static_cast<int>(p.type) != selected)
            fail("Pricing requires one contract type. Hedging also accepts mixed types.");
    auto terms = in.contract;
    terms.time_intervals = count(s, "intervals", 1);
    const auto requested = count(s, "paths", 2);
    const auto repetitions = count(s, "repetitions", 1);
    const auto cs = integer(field(s, "carbon_seed"));
    const auto fs = integer(field(s, "financial_seed"));
    if (cs == fs) fail("Carbon and financial seeds must differ.");
    const auto& names = field(s, "configurations");
    if (!names.is_array() || names.as_array().empty()) fail("Select at least one configuration.");
    std::vector<std::string> configurations;
    std::set<std::string> seen;
    bool antithetic = false;
    for (const auto& name : names.as_array()) {
        auto n = string(name);
        if (!seen.insert(n).second) fail("Duplicate estimator configuration.");
        antithetic |= mode(n).sampling == cf::pricing::PetieiSampling::antithetic;
        configurations.push_back(std::move(n));
    }
    if (antithetic && requested < 3) fail("Use at least three requested paths with antithetics.");
    const auto paths = requested + (antithetic ? requested % 2 : 0);
    cf::contracts::PolicyholderPopulation population{in.people};
    using Pricer = cf::pricing::PetieiMonteCarloPricer4<Stepper>;
    const auto prepared = Pricer::prepare(terms, in.initial_carbon, carbon,
        in.allocation, population.living_portfolio(), in.mortality);
    js::array records, summary;
    std::vector<std::vector<double>> timings(configurations.size());
    std::set<std::uint64_t> used_seeds;
    const auto distinct_seed = [&](std::uint64_t candidate) {
        while (!used_seeds.insert(candidate).second) ++candidate;
        return candidate;
    };
    // Rotate evaluation order across repetitions to reduce fixed-order timing effects.
    for (std::size_t repetition = 0; repetition < repetitions; ++repetition) {
        const auto repetition_carbon_seed = distinct_seed(repeat_seed(cs, repetition));
        const auto repetition_financial_seed = distinct_seed(repeat_seed(fs, repetition));
        for (std::size_t k = 0; k < configurations.size(); ++k) {
            const auto index = (k + repetition) % configurations.size();
            const auto& name = configurations[index];
            const auto m = mode(name);
            cf::random::SimulationRng cr{repetition_carbon_seed};
            cf::random::SimulationRng fr{repetition_financial_seed};
            const auto start = std::clock();
            const auto r = Pricer::evaluate(prepared, paths, cr, fr, m.estimator, m.sampling,
                static_cast<cf::pricing::PetieiContract>(selected + 1));
            const auto cpu = cpu_since(start);
            timings[index].push_back(cpu);
            const auto& e = estimate(r, selected);
            const std::array<std::pair<const char*, const cf::pricing::MonteCarloEstimate*>, 2>
                quantities{{{"price", &e.price}, {"delta", &e.delta}}};
            for (const auto& [quantity, value] : quantities) {
                Object row = moment(*value);
                row["configuration"] = name;
                row["quantity"] = quantity;
                row["repetition"] = repetition;
                row["requested_paths"] = requested;
                row["actual_paths"] = r.simulations;
                row["independent_samples"] = r.independent_samples;
                row["cpu_seconds"] = cpu;
                row["carbon_seed"] = repetition_carbon_seed;
                row["financial_seed"] = repetition_financial_seed;
                records.push_back(row);
                if (repetition == 0) summary.push_back(row);
            }
        }
        std::cout << "Pricing repetitions completed: " << repetition + 1
                  << '/' << repetitions << std::endl;
    }
    for (std::size_t index = 0; index < configurations.size(); ++index) {
        auto t = timings[index];
        std::sort(t.begin(), t.end());
        const auto n = t.size();
        const double median = n % 2 ? t[n / 2] : 0.5 * (t[n / 2 - 1] + t[n / 2]);
        for (auto& v : summary)
            if (string(v.as_object().at("configuration")) == configurations[index])
                v.as_object()["cpu_seconds"] = median;
    }
    write_json(output / "pricing.json",
        Object{{"summary", summary}, {"repetitions", records}});
    return {{"actual_paths", paths}, {"timing_statistic", "median"},
        {"reported_estimates", "first repetition; repetitions are not pooled"},
        {"seed_schedule", "repetition 0: supplied seed; r>0: splitmix64(seed XOR r); increment on collision; actual seeds saved per repetition"}};
}

template<class Stepper>
Object hedging(const Inputs& in, const typename Stepper::Parameters& carbon,
               const std::filesystem::path& output)
{
    const auto& s = object(field(in.config.as_object(), "hedging"), "hedging");
    keys(s, {"outer_scenarios", "intervals", "progress_every", "threads", "carbon_seed",
             "financial_seed", "mortality_seed", "initial", "revaluation"}, "hedging");
    const auto scenarios = count(s, "outer_scenarios", 2);
    const auto intervals = count(s, "intervals", 1);
    const auto progress = count(s, "progress_every", 0);
    int requested_threads = 1;
    const int team_limit = hedging_threads(s, scenarios, requested_threads);
    const auto initial = valuation(field(s, "initial"));
    const auto later = valuation(field(s, "revaluation"), true);
    const auto cs = integer(field(s, "carbon_seed"));
    const auto fs = integer(field(s, "financial_seed"));
    const auto ms = integer(field(s, "mortality_seed"));
    const std::set<std::uint64_t> seeds{cs, fs, ms, initial.carbon_seed,
        initial.financial_seed, later.carbon_seed, later.financial_seed};
    if (seeds.size() != 7) fail("Use distinct seeds for all seven hedging streams.");
    cf::random::SimulationRng ic{initial.carbon_seed}, iff{initial.financial_seed};
    cf::contracts::PolicyholderPopulation population{in.people};
    using Pricer = cf::pricing::PetieiMonteCarloPricer4<Stepper>;
    const auto value_at = [&](double time, double fund, const Eigen::Vector4d& state,
                              const cf::contracts::PetieiPortfolioView& living,
                              const Valuation& v, auto& carbon_rng, auto& financial_rng) {
        auto p = in.contract;
        p.fund_value = fund;
        p.time_to_maturity = in.contract.time_to_maturity - time;
        p.minimum_guarantee = p.minimum_guarantee == 0.0 ? 0.0 :
            p.minimum_guarantee * std::exp(p.minimum_guarantee_growth_rate * time);
        p.maximum_guarantee *= std::exp(p.maximum_guarantee_growth_rate * time);
        p.time_intervals = cf::hedging::petiei_revaluation_intervals(
            v.intervals, p.time_to_maturity, in.contract.time_to_maturity,
            v.minimum_intervals);
        const auto prepared = Pricer::prepare(p, state, carbon, in.allocation, living, in.mortality);
        const auto m = mode(v.configuration);
        return Pricer::evaluate(prepared, v.paths, carbon_rng, financial_rng,
                               m.estimator, m.sampling, cf::pricing::PetieiContract::all);
    };
    const auto to_hedge = [](const Result& r) {
        return cf::hedging::PetieiHedgeValuation{
            r.pure_endowment.price.mean + r.term_insurance.price.mean + r.endowment_insurance.price.mean,
            {r.pure_endowment.delta.mean, r.term_insurance.delta.mean, r.endowment_insurance.delta.mean}};
    };
    const auto initial_wall_start = std::chrono::steady_clock::now();
    const auto start = std::clock();
    const auto first = value_at(0.0, in.contract.fund_value, in.initial_carbon,
        population.living_portfolio(), initial, ic, iff);
    const auto initial_cpu = cpu_since(start);
    const double initial_wall = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - initial_wall_start).count();
    const auto initial_value = to_hedge(first);
    // Validate revaluation settings even when there will be no intermediate date.
    auto check_terms = in.contract;
    check_terms.time_intervals = later.intervals;
    static_cast<void>(Pricer::prepare(check_terms, in.initial_carbon, carbon,
        in.allocation, population.living_portfolio(), in.mortality));
    using Generator = cf::simulation::MarketScenarioGenerator4<Stepper>;
    const auto prepared = Generator::prepare(in.contract.fund_value, in.initial_carbon,
        carbon, in.allocation, in.contract.time_to_maturity, intervals);
    const auto simulation_wall_start = std::chrono::steady_clock::now();
    const auto simulation_start = std::clock();
    const auto streams = scenario_seeds(scenarios,
        {cs, fs, ms, later.carbon_seed, later.financial_seed},
        initial.carbon_seed, initial.financial_seed);
    struct Worker {
        cf::contracts::PolicyholderPopulation population;
        cf::simulation::MarketPath4 path;
        explicit Worker(const cf::contracts::PolicyholderPopulation& initial)
            : population(initial) {}
    };
    // Construct worker storage before entering OpenMP, where allocation errors
    // cannot safely escape a parallel region. Each worker reuses its own buffers.
    std::vector<Worker> workers;
    workers.reserve(static_cast<std::size_t>(team_limit));
    for (int i = 0; i < team_limit; ++i) workers.emplace_back(population);
    std::vector<cf::hedging::PetieiHedgeOutcome> outcomes(scenarios);
    std::mutex progress_mutex, failure_mutex;
    std::size_t completed = 0;
    std::atomic<bool> failed{false};
    std::exception_ptr failure;
    const auto run_scenario = [&](std::size_t j, Worker& worker) {
        if (failed.load(std::memory_order_relaxed)) return;
        try {
            const auto& seed = streams[j];
            cf::random::SimulationRng cr{seed.carbon}, fr{seed.financial}, mr{seed.mortality};
            cf::random::SimulationRng rc{seed.valuation_carbon}, rf{seed.valuation_financial};
            const auto revalue = [&](double time, double fund, const Eigen::Vector4d& state,
                                     cf::contracts::PetieiPortfolioView living) {
                return to_hedge(value_at(time, fund, state, living, later, rc, rf));
            };
            Generator::generate(prepared, cr, fr, worker.path);
            outcomes[j] = cf::hedging::run_petiei_hedge(j, worker.path, in.contract,
                in.allocation.risk_free_rate, worker.population, in.mortality,
                initial_value, mr, revalue);
            if (progress) {
                std::lock_guard<std::mutex> lock(progress_mutex);
                ++completed;
                if (completed % progress == 0 || completed == scenarios)
                    std::cout << "Hedging scenarios completed: " << completed
                              << '/' << scenarios << std::endl;
            }
        } catch (...) {
            std::lock_guard<std::mutex> lock(failure_mutex);
            if (!failure) failure = std::current_exception();
            failed.store(true, std::memory_order_relaxed);
        }
    };
    int actual_threads = 1;
#ifdef _OPENMP
#pragma omp parallel num_threads(team_limit) if(team_limit > 1)
    {
#pragma omp single
        actual_threads = omp_get_num_threads();
        auto& worker = workers[static_cast<std::size_t>(omp_get_thread_num())];
#pragma omp for schedule(dynamic, 1)
        for (std::size_t j = 0; j < scenarios; ++j) run_scenario(j, worker);
    }
#else
    for (std::size_t j = 0; j < scenarios; ++j) run_scenario(j, workers.front());
#endif
    if (failure) std::rethrow_exception(failure);
    const auto simulation_cpu = cpu_since(simulation_start);
    const double simulation_wall = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - simulation_wall_start).count();
    cf::io::write_hedging_csv(output / "hedging.csv", outcomes);
    js::array deltas;
    for (double d : initial_value.deltas) deltas.push_back(d);
    return {{"initial_price", initial_value.price}, {"initial_deltas_PE_TI_EI", deltas},
        {"revaluation_minimum_intervals", later.minimum_intervals},
        {"revaluation_grid_rule", "max(minimum_intervals, ceil(intervals * (T-t) / T))"},
        {"initial_actual_paths", first.simulations},
        {"revaluation_actual_paths", later.paths +
            (mode(later.configuration).sampling == cf::pricing::PetieiSampling::antithetic
                ? later.paths % 2 : 0)},
        {"initial_cpu_seconds", initial_cpu}, {"simulation_cpu_seconds", simulation_cpu},
        {"initial_wall_seconds", initial_wall}, {"simulation_wall_seconds", simulation_wall},
        {"threads_requested", requested_threads}, {"threads_used", actual_threads},
#ifdef _OPENMP
        {"openmp_enabled", true},
#else
        {"openmp_enabled", false},
#endif
        {"scenario_seed_schedule", "splitmix64-scenario-role-v1; scenario-major; "
            "five streams; increment on collision; exclude initial seeds"},
        {"cost_valuation_time", 0}, {"positive_cost", "additional funding required"},
        {"outer_measure", "physical"}, {"inner_measure", "minimal martingale"},
        {"death_settlement", "interval right endpoint"},
        {"static_strategy", "initial deltas scaled by living-notional ratios per type"}};
}

template<class Stepper>
Object dispatch(const Inputs& in, const std::filesystem::path& output)
{
    const auto carbon = carbon_parameters<Stepper>(in);
    return in.experiment == "pricing" ? pricing<Stepper>(in, carbon, output)
                                      : hedging<Stepper>(in, carbon, output);
}
} // namespace

int main(int argc, char** argv)
{
    try {
        if (argc == 2 && std::string_view(argv[1]) == "--help") {
            std::cout << "Usage: carbonfund_experiment --config CONFIG.json --output DIRECTORY\n";
            return 0;
        }
        if (argc != 5 || std::string_view(argv[1]) != "--config" ||
            std::string_view(argv[3]) != "--output")
            fail("Usage: carbonfund_experiment --config CONFIG.json --output DIRECTORY");
        std::ifstream file(argv[2]);
        if (!file) fail("Cannot open configuration file.");
        const std::string text{std::istreambuf_iterator<char>(file), {}};
        const Inputs input{js::parse(text)};
        const std::filesystem::path output{argv[4]};
        std::filesystem::create_directories(output);
        const auto start = std::chrono::steady_clock::now();
        Object run;
        if (input.carbon_model == "OU") run = dispatch<cf::models::OUExactStepper4>(input, output);
        else if (input.carbon_model == "GeoOU") run = dispatch<cf::models::GeoOUExactStepper4>(input, output);
        else if (input.carbon_model == "CIR") run = dispatch<cf::models::CIRNVStepper4>(input, output);
        else fail("carbon.model must be OU, GeoOU or CIR.");
        double notional = 0;
        for (const auto& p : input.people) notional += p.notional;
        Object metadata{{"configuration", input.config}, {"run", run},
            {"policy_count", input.people.size()}, {"total_notional", notional},
            {"wall_seconds", std::chrono::duration<double>(
                std::chrono::steady_clock::now() - start).count()},
            {"compiler", __VERSION__}, {"boost_version", BOOST_VERSION},
            {"eigen_version", std::to_string(EIGEN_WORLD_VERSION) + "." +
                std::to_string(EIGEN_MAJOR_VERSION) + "." + std::to_string(EIGEN_MINOR_VERSION)}};
#ifdef NDEBUG
        metadata["build"] = "Release";
#else
        metadata["build"] = "Debug";
#endif
        write_json(output / "metadata.json", metadata);
        std::cout << "Saved experiment to " << output.string() << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "Experiment error: " << error.what() << '\n';
        return 1;
    }
}

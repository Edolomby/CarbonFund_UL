#include <carbonfund/random/simulation_rng.hpp>

#include <cmath>
#include <cstdint>
#include <exception>
#include <iostream>
#include <stdexcept>

namespace {

using carbonfund::random::SimulationRng;
constexpr std::uint64_t seed = 123456789;

void require(bool condition, const char* message)
{
    if (!condition) {
        throw std::runtime_error(message);
    }
}

void check_reproducibility_and_ranges()
{
    SimulationRng first(seed);
    SimulationRng second(seed);

    // Interleave distributions to exercise the shared engine state.
    for (int i = 0; i < 1000; ++i) {
        const double normal = first.std_normal();
        require(std::isfinite(normal), "Nonfinite normal sample.");
        require(normal == second.std_normal(),
                "Normal samples are not reproducible.");

        const double uniform = first.uniform01();
        require(uniform >= 0.0 && uniform < 1.0,
                "Uniform sample is outside [0, 1).");
        require(uniform == second.uniform01(),
                "Uniform samples are not reproducible.");

        const double probability = (i % 11) / 10.0;
        const bool event = first.bernoulli(probability);
        require(event == second.bernoulli(probability),
                "Bernoulli samples are not reproducible.");
    }
}

void check_vector_draws()
{
    SimulationRng vector_rng(seed);
    SimulationRng scalar_rng(seed);
    Eigen::Vector4d std_normals;

    for (int batch = 0; batch < 100; ++batch) {
        vector_rng.fill_std_normals(std_normals);

        require(std_normals.allFinite(),
                "Nonfinite vector normal sample.");

        for (int i = 0; i < 4; ++i) {
            require(std_normals[i] == scalar_rng.std_normal(),
                    "Vector draws differ from consecutive scalar draws.");
        }
    }
}

void check_probability_boundaries()
{
    SimulationRng rng(seed);

    for (int i = 0; i < 1000; ++i) {
        require(!rng.bernoulli(0.0), "An impossible event occurred.");
        require(rng.bernoulli(1.0), "A certain event did not occur.");
    }
}

void check_seed_selection()
{
    SimulationRng first(seed);
    SimulationRng different(seed + 1);
    bool sequences_differ = false;

    for (int i = 0; i < 100; ++i) {
        const double a = first.uniform01();
        const double b = different.uniform01();
        sequences_differ = sequences_differ || (a != b);
    }

    require(sequences_differ,
            "Different seeds produced identical tested sequences.");
}

} // namespace

int main()
{
    try {
        check_reproducibility_and_ranges();
        check_vector_draws();
        check_probability_boundaries();
        check_seed_selection();

        std::cout << "All simulation RNG checks passed.\n";
        return 0;
    } catch (const std::exception& exception) {
        std::cerr << "Test failure: " << exception.what() << '\n';
        return 1;
    }
}
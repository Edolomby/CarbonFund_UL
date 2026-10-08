#pragma once

#include <Eigen/Core>
#include <boost/random/bernoulli_distribution.hpp>
#include <boost/random/normal_distribution.hpp>
#include <boost/random/uniform_01.hpp>
#include <boost/random/xoshiro.hpp>

#include <cstdint>

namespace carbonfund::random {

// Owns one mutable random stream. Pass by reference to simulation routines.
class SimulationRng {
public:
    explicit SimulationRng(std::uint64_t seed)
        : engine_(seed)
    {
    }

    // N(0, 1). The carbon stepper supplies the time-step scaling.
    [[nodiscard]] double std_normal()
    {
        return std_normal_(engine_);
    }

    // Overwrite the vector with four consecutive standard normal samples.
    void fill_std_normals(Eigen::Vector4d& std_normals)
    {
        for (int i = 0; i < 4; ++i) {
            std_normals[i] = std_normal();
        }
    }

    // Uniform on [0, 1): zero is possible; one is excluded.
    [[nodiscard]] double uniform01()
    {
        return uniform_(engine_);
    }

    // Returns true with the supplied probability.
    // Precondition: probability is finite and in [0, 1].
    // Validate probabilities during preparation, outside sampling loops.
    [[nodiscard]] bool bernoulli(double probability)
    {
        const boost::random::bernoulli_distribution<double> event(probability);
        return event(engine_);
    }

private:
    boost::random::xoshiro256pp engine_;
    boost::random::normal_distribution<double> std_normal_{0.0, 1.0};
    boost::random::uniform_01<double> uniform_;
};

} // namespace carbonfund::random
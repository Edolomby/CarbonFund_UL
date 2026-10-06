#include <carbonfund/models/carbon_models_4d.hpp>

#include <exception>
#include <iostream>

namespace models = carbonfund::models;

namespace {

template<class Model>
bool check_equilibrium(
    const char* name,
    const typename Model::Parameters& parameters,
    const Eigen::Vector4d& initial)
{
    // All parameter sets in this test have zero volatility.
    // Therefore, the normal draws must have no effect.
    const Eigen::Vector4d standard_normals{-0.5, 1.0, -2.0, 0.3};

    const auto prepared = Model::prepare(0.25, parameters);

    Eigen::Vector4d state = initial;
    Model::step(state, prepared, standard_normals);

    const bool passed =
        state.allFinite() &&
        (state - initial).cwiseAbs().maxCoeff() < 1e-12;

    if (!passed) {
        std::cerr << name << ": equilibrium test failed\n"
                  << "Expected: " << initial.transpose() << '\n'
                  << "Obtained: " << state.transpose() << '\n';
    }

    return passed;
}

} // namespace

int main()
{
    try {
        const Eigen::Vector4d beta{0.2, 0.4, 0.6, 0.8};
        const Eigen::Vector4d kappa{0.3, 0.7, 1.1, 1.8};
        const Eigen::Vector4d volatility = Eigen::Vector4d::Zero();

        const models::CIRParameters4 cir_parameters{
            beta, kappa, volatility
        };

        const models::OUParameters4 ou_parameters{
            beta, kappa, volatility
        };

        // For geometric OU, beta is the long-run mean of log(Y).
        const models::GeoOUParameters4 geo_parameters{
            beta, kappa, volatility
        };

        const Eigen::Vector4d geometric_initial =
            beta.array().exp().matrix();

        const bool cir =
            check_equilibrium<models::CIRNVStepper4>(
                "CIR", cir_parameters, beta);

        const bool ou =
            check_equilibrium<models::OUExactStepper4>(
                "OU", ou_parameters, beta);

        const bool geometric_ou =
            check_equilibrium<models::GeoOUExactStepper4>(
                "Geometric OU", geo_parameters, geometric_initial);

        if (!(cir && ou && geometric_ou)) {
            return 1;
        }

        std::cout << "All carbon model equilibrium checks passed.\n";
        return 0;
    } catch (const std::exception& exception) {
        std::cerr << "Unexpected test failure: "
                  << exception.what() << '\n';
        return 1;
    }
}
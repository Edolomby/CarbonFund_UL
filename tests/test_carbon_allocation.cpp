#include <carbonfund/portfolio/carbon_allocation_4d.hpp>

#include <algorithm>
#include <cmath>
#include <exception>
#include <iostream>
#include <limits>
#include <stdexcept>

namespace {

using Allocation = carbonfund::portfolio::CarbonAllocation4;
using Parameters = Allocation::Parameters;

void require(bool condition, const char* message)
{
    if (!condition) {
        throw std::runtime_error(message);
    }
}

Parameters make_parameters()
{
    Parameters parameters;
    parameters.risk_free_rate = 0.02;
    parameters.risk_aversion = 2.0;
    parameters.carbon_penalty = 0.5;
    parameters.expected_returns << 0.10, 0.08, 0.14, -0.02;
    parameters.covariance.setZero();
    parameters.covariance.diagonal() << 0.04, 0.09, 0.16, 0.25;
    return parameters;
}

void check_diagonal_case_and_workspace_reuse(double first_asset_variance)
{
    auto parameters = make_parameters();
    parameters.covariance(0, 0) = first_asset_variance;

    const auto prepared = Allocation::prepare(parameters);
    Allocation::Workspace workspace;

    const Eigen::Vector4d states[] = {
        Eigen::Vector4d{-2.0, 0.0, 1.0, 4.0},
        Eigen::Vector4d{3.0, -1.0, 0.0, 2.0}
    };

    for (const auto& state : states) {
        Allocation::evaluate(state, prepared, workspace);

        Eigen::Vector4d expected;
        double expected_variance = 0.0;

        for (int i = 0; i < 4; ++i) {
            const double variance = parameters.covariance(i, i);

            expected[i] =
                (parameters.expected_returns[i] -
                 parameters.risk_free_rate) /
                (variance *
                 (parameters.risk_aversion +
                  parameters.carbon_penalty * std::max(state[i], 0.0)));

            expected_variance += variance * expected[i] * expected[i];
        }

        require(
            (workspace.weights - expected).cwiseAbs().maxCoeff() <
                1e-13 * std::max(1.0, expected.cwiseAbs().maxCoeff()),
            "Diagonal allocation or workspace reuse failed.");

        require(
            std::abs(workspace.instantaneous_variance - expected_variance) <
                1e-13 * std::max(1.0, expected_variance),
            "Diagonal portfolio variance failed.");
    }
}

void check_correlated_case()
{
    auto parameters = make_parameters();
    parameters.covariance(0, 1) = parameters.covariance(1, 0) = 0.01;

    const auto prepared = Allocation::prepare(parameters);
    Allocation::Workspace workspace;
    const Eigen::Vector4d state{2.0, 4.0, -3.0, 0.0};

    Allocation::evaluate(state, prepared, workspace);

    // Analytical solution of the coupled 2x2 block.
    Eigen::Vector4d expected;
    expected << 0.0276 / 0.0428, 0.0056 / 0.0428, 0.375, -0.08;

    const double expected_variance =
        0.04 * expected[0] * expected[0] +
        0.09 * expected[1] * expected[1] +
        0.02 * expected[0] * expected[1] +
        0.16 * expected[2] * expected[2] +
        0.25 * expected[3] * expected[3];

    require(
        (workspace.weights - expected).cwiseAbs().maxCoeff() < 1e-13,
        "Correlated allocation failed.");

    require(
        std::abs(workspace.instantaneous_variance - expected_variance) < 1e-13,
        "Correlated portfolio variance failed.");

    // Reuse the workspace with different prepared data.
    parameters.carbon_penalty = 0.0;
    const auto no_penalty_prepared = Allocation::prepare(parameters);

    Allocation::evaluate(state, no_penalty_prepared, workspace);
    const Eigen::Vector4d no_penalty_weights = workspace.weights;

    Allocation::evaluate(
        Eigen::Vector4d::Constant(100.0), no_penalty_prepared, workspace);

    require(
        (workspace.weights.array() == no_penalty_weights.array()).all(),
        "Zero penalty must make allocation independent of carbon state.");

    parameters.expected_returns.setConstant(parameters.risk_free_rate);
    const auto zero_prepared = Allocation::prepare(parameters);

    Allocation::evaluate(state, zero_prepared, workspace);

    require(
        workspace.weights.isZero(0.0) &&
        workspace.instantaneous_variance == 0.0,
        "Zero excess returns must produce zero weights and variance.");
}

void require_invalid(const Parameters& parameters)
{
    try {
        static_cast<void>(Allocation::prepare(parameters));
    } catch (const std::invalid_argument&) {
        return;
    }

    throw std::runtime_error("Invalid allocation parameters were accepted.");
}

void check_parameter_validation()
{
    auto parameters = make_parameters();
    parameters.risk_aversion = 0.0;
    require_invalid(parameters);

    parameters = make_parameters();
    parameters.carbon_penalty = -1.0;
    require_invalid(parameters);

    parameters = make_parameters();
    parameters.expected_returns[0] =
        std::numeric_limits<double>::quiet_NaN();
    require_invalid(parameters);

    parameters = make_parameters();
    parameters.covariance(0, 1) = 0.01;
    require_invalid(parameters);

    parameters = make_parameters();
    parameters.covariance(0, 0) = 0.0;
    require_invalid(parameters);
}

} // namespace

int main()
{
    try {
        check_diagonal_case_and_workspace_reuse(0.04);

        // Small first pivot exercises the Eigen LLT fallback.
        check_diagonal_case_and_workspace_reuse(1e-12);

        check_correlated_case();
        check_parameter_validation();

        std::cout << "All carbon allocation checks passed.\n";
        return 0;
    } catch (const std::exception& exception) {
        std::cerr << "Test failure: " << exception.what() << '\n';
        return 1;
    }
}
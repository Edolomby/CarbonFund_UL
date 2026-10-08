#pragma once

#include <Eigen/Core>
#include <Eigen/Cholesky>

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace carbonfund::portfolio {

namespace detail {

// Retain the original fixed-size solve.
// Small pivots request an LLT fallback.
inline bool solve_fixed4_ldlt(
    const Eigen::Matrix4d& a,
    const Eigen::Vector4d& b,
    Eigen::Vector4d& x)
{
    const double a0 = a(0, 0);
    const double a1 = a(1, 1);
    const double a2 = a(2, 2);
    const double a3 = a(3, 3);

    const double largest_diagonal = std::max({a0, a1, a2, a3});

    if (!(largest_diagonal > 0.0) ||
        !std::isfinite(largest_diagonal)) {
        return false;
    }

		// heuristic pivot floor: 1e-5 * largest diagonal
    const double pivot_floor = 1e-5 * largest_diagonal;

    const double d0 = a0;
    if (!(d0 > pivot_floor) || !std::isfinite(d0)) {
        return false;
    }

    const double l10 = a(1, 0) / d0;
    const double l20 = a(2, 0) / d0;
    const double l30 = a(3, 0) / d0;

    const double d1 = a1 - l10 * l10 * d0;
    if (!(d1 > pivot_floor) || !std::isfinite(d1)) {
        return false;
    }

    const double l21 = (a(2, 1) - l20 * l10 * d0) / d1;
    const double l31 = (a(3, 1) - l30 * l10 * d0) / d1;

    const double d2 =
        a2 - l20 * l20 * d0 - l21 * l21 * d1;

    if (!(d2 > pivot_floor) || !std::isfinite(d2)) {
        return false;
    }

    const double l32 =
        (a(3, 2) - l30 * l20 * d0 - l31 * l21 * d1) / d2;

    const double d3 =
        a3 - l30 * l30 * d0 -
        l31 * l31 * d1 - l32 * l32 * d2;

    if (!(d3 > pivot_floor) || !std::isfinite(d3)) {
        return false;
    }

    // Forward substitution.
    const double u0 = b[0];
    const double u1 = b[1] - l10 * u0;
    const double u2 = b[2] - l20 * u0 - l21 * u1;
    const double u3 = b[3] - l30 * u0 - l31 * u1 - l32 * u2;

    // Diagonal solve.
    const double z0 = u0 / d0;
    const double z1 = u1 / d1;
    const double z2 = u2 / d2;
    const double z3 = u3 / d3;

    // Back substitution.
    const double x3 = z3;
    const double x2 = z2 - l32 * x3;
    const double x1 = z1 - l21 * x2 - l31 * x3;
    const double x0 = z0 - l10 * x1 - l20 * x2 - l30 * x3;

    x << x0, x1, x2, x3;
    return x.allFinite();
}

} // namespace detail

struct CarbonAllocationParameters4 {
    double risk_free_rate = 0.0;
    double risk_aversion  = 1.0; // 1.0 = log utility
    double carbon_penalty = 0.0; 

    Eigen::Vector4d expected_returns = Eigen::Vector4d::Zero();
    Eigen::Matrix4d covariance = Eigen::Matrix4d::Identity();
};

struct CarbonAllocation4 {
    using Parameters = CarbonAllocationParameters4;

    class Prepared {
        friend struct CarbonAllocation4;

        Eigen::Matrix4d covariance_factor_;
        Eigen::Matrix4d base_covariance_;
        Eigen::Vector4d excess_returns_;
        Eigen::Vector4d penalty_scale_;

        Prepared(
            const Eigen::Matrix4d& covariance_factor,
            const Eigen::Matrix4d& base_covariance,
            const Eigen::Vector4d& excess_returns,
            const Eigen::Vector4d& penalty_scale)
            : covariance_factor_(covariance_factor),
              base_covariance_(base_covariance),
              excess_returns_(excess_returns),
              penalty_scale_(penalty_scale)
        {
        }
    };

    class Workspace {
        friend struct CarbonAllocation4;

        Eigen::Matrix4d penalised_covariance_;
        Eigen::LLT<Eigen::Matrix4d> solver_;

    public:
        Eigen::Vector4d weights = Eigen::Vector4d::Zero();
        double instantaneous_variance = 0.0;
    };

    [[nodiscard]] static Prepared prepare(
        const Parameters& parameters)
    {
        if (!std::isfinite(parameters.risk_free_rate) ||
            !std::isfinite(parameters.risk_aversion) ||
            !std::isfinite(parameters.carbon_penalty) ||
            !parameters.expected_returns.allFinite() ||
            !parameters.covariance.allFinite()) {
            throw std::invalid_argument(
                "Allocation parameters must be finite.");
        }

        if (parameters.risk_aversion <= 0.0 ||
            parameters.carbon_penalty < 0.0) {
            throw std::invalid_argument(
                "Risk aversion must be positive; "
                "carbon penalty must be nonnegative.");
        }

        if (!parameters.covariance.isApprox(
                parameters.covariance.transpose(), 1e-12)) {
            throw std::invalid_argument(
                "Asset covariance must be symmetric.");
        }

        // Remove roundoff-sized asymmetry.
        const Eigen::Matrix4d covariance =
            0.5 * parameters.covariance +
            0.5 * parameters.covariance.transpose();

        const Eigen::LLT<Eigen::Matrix4d> covariance_solver(covariance);

        if (covariance_solver.info() != Eigen::Success) {
            throw std::invalid_argument(
                "Asset covariance must be positive definite.");
        }

        const Eigen::Matrix4d base_covariance =
            parameters.risk_aversion * covariance;

        const Eigen::Vector4d excess_returns =
            (parameters.expected_returns.array() -
             parameters.risk_free_rate).matrix();

        const Eigen::Vector4d penalty_scale =
            parameters.carbon_penalty * covariance.diagonal();

        if (!base_covariance.allFinite() ||
            !excess_returns.allFinite() ||
            !penalty_scale.allFinite()) {
            throw std::invalid_argument(
                "Precomputed allocation coefficients overflow.");
        }

        const Eigen::Matrix4d covariance_factor =
            covariance_solver.matrixU();

        return Prepared{
            covariance_factor,
            base_covariance,
            excess_returns,
            penalty_scale
        };
    }

    // Overwrites workspace outputs.
    // Caller preconditions: finite carbon state and representable arithmetic.
    // Weights are unconstrained: borrowing and short positions are allowed.
    static void evaluate(
        const Eigen::Vector4d& carbon_state,
        const Prepared& prepared,
        Workspace& workspace)
    {
        workspace.penalised_covariance_ = prepared.base_covariance_;

        workspace.penalised_covariance_.diagonal().array() +=
            prepared.penalty_scale_.array() *
            carbon_state.array().max(0.0);

        if (!detail::solve_fixed4_ldlt(
                workspace.penalised_covariance_,
                prepared.excess_returns_,
                workspace.weights)) 
				{
            workspace.solver_.compute(workspace.penalised_covariance_);

            if (workspace.solver_.info() != Eigen::Success) {
              	throw std::runtime_error("Allocation factorization failed.");
            }

            workspace.weights =
              workspace.solver_.solve(prepared.excess_returns_);
        }

        // instantaneous_variance = ||U pi||^2.
        workspace.instantaneous_variance =
    			(prepared.covariance_factor_ * workspace.weights).squaredNorm();
    }
};

} // namespace carbonfund::portfolio
#pragma once

#include <carbonfund/models/parameters.hpp>

#include <cmath>
#include <stdexcept>
#include <utility>

namespace carbonfund::models {

struct GeoOUExactStepper4;

namespace detail {

inline void validate_time_step4(double dt)
{
    if (!std::isfinite(dt) || dt <= 0.0) {
        throw std::invalid_argument(
            "Time step must be finite and strictly positive.");
    }
}

inline void validate_parameters4(
    const Eigen::Vector4d& mean,
    const Eigen::Vector4d& mean_reversion,
    const Eigen::Vector4d& volatility)
{
    if (!mean.allFinite() ||
        !mean_reversion.allFinite() ||
        !volatility.allFinite()) {
        throw std::invalid_argument(
            "Model parameters must be finite.");
    }

    if ((mean_reversion.array() < 0.0).any() ||
        (volatility.array() < 0.0).any()) {
        throw std::invalid_argument(
            "Mean reversion and volatility must be nonnegative.");
    }
}

// Integral from 0 to dt of exp(-2*kappa*u) du.
// The small-product branch also avoids losing subnormal kappa*dt values
// through multiplication by 0.5 before division by kappa.
inline double ou_variance_time4(double kappa, double dt)
{
    const double product = kappa * dt;

    if (kappa == 0.0 || product == 0.0) {
        return dt;
    }

    if (product < 0.5) {
        const double twice_product = 2.0 * product;
        return dt * (-std::expm1(-twice_product) / twice_product);
    }

    return (0.5 / kappa) * (-std::expm1(-2.0 * product));
}

} // namespace detail

// Independent four-factor exact OU transition.
// Each entry of standard_normals is a N(0,1) draw,
// not a Brownian increment.
struct OUExactStepper4 {
    using Parameters = OUParameters4;

    class Prepared {
        friend struct OUExactStepper4;
        friend struct ::carbonfund::models::GeoOUExactStepper4;

        Eigen::Vector4d mean_shift_;
        Eigen::Vector4d decay_;
        Eigen::Vector4d noise_scale_;

        Prepared(
            const Eigen::Vector4d& shift,
            const Eigen::Vector4d& decay,
            const Eigen::Vector4d& noise)
            : mean_shift_(shift),
              decay_(decay),
              noise_scale_(noise)
        {
        }
    };

    [[nodiscard]] static Prepared prepare(
        double dt,
        const Parameters& parameters)
    {
        detail::validate_time_step4(dt);
        detail::validate_parameters4(
            parameters.long_run_mean,
            parameters.mean_reversion,
            parameters.volatility);

        Eigen::Vector4d shift = Eigen::Vector4d::Zero();
        Eigen::Vector4d decay = Eigen::Vector4d::Ones();
        Eigen::Vector4d noise = Eigen::Vector4d::Zero();

        for (int i = 0; i < 4; ++i) {
            const double kappa = parameters.mean_reversion[i];
            const double product = kappa * dt;

            shift[i] =
                parameters.long_run_mean[i] * (-std::expm1(-product));

            decay[i] = std::exp(-product);

            noise[i] =
                parameters.volatility[i] *
                std::sqrt(detail::ou_variance_time4(kappa, dt));
        }

        return Prepared{shift, decay, noise};
    }

    // Unchecked in-place transition.
    // Finite state, finite normal draws, and representability of results
    // are the caller's responsibility.
    // Prepared data may be shared between workers when kept read-only.
    static void step(
        Eigen::Vector4d& state,
        const Prepared& prepared,
        const Eigen::Vector4d& standard_normals) noexcept
    {
        state.array() =
            prepared.mean_shift_.array() +
            prepared.decay_.array() * state.array() +
            prepared.noise_scale_.array() * standard_normals.array();
    }
};

// Exact transition when log(Y) follows an OU process.
// The parameter mean is the long-run mean of log(Y),
// not the long-run expectation of Y.
struct GeoOUExactStepper4 {
    using Parameters = GeoOUParameters4;

    class Prepared {
        friend struct ::carbonfund::models::GeoOUExactStepper4;

        OUExactStepper4::Prepared log_transition_;

        explicit Prepared(OUExactStepper4::Prepared transition)
            : log_transition_(std::move(transition))
        {
        }
    };

    [[nodiscard]] static Prepared prepare(
        double dt,
        const Parameters& parameters)
    {
        const OUParameters4 log_parameters{
            parameters.long_run_log_mean,
            parameters.mean_reversion,
            parameters.volatility
        };

        return Prepared{
            OUExactStepper4::prepare(dt, log_parameters)
        };
    }

    // Unchecked in-place transition.
    // Positive finite state and finite normal draws are caller preconditions.
    // Overflow and underflow are not trapped here.
    static void step(
        Eigen::Vector4d& state,
        const Prepared& prepared,
        const Eigen::Vector4d& standard_normals) noexcept
    {
        const auto& coeff = prepared.log_transition_;

        state.array() =
      (
          coeff.mean_shift_.array() +
          coeff.decay_.array() * state.array().log() +
          coeff.noise_scale_.array() * standard_normals.array()
      ).exp();
    }
};

// CIR Ninomiya-Victoir square-root split, not an exact CIR draw.
// Its admissible regime is kappa*mean >= volatility^2/4 in every factor.
// This differs from the Feller boundary-inaccessibility condition.
struct CIRNVStepper4 {
    using Parameters = CIRParameters4;

    class Prepared {
        friend struct CIRNVStepper4;

        Eigen::Vector4d drift_shift_;
        Eigen::Vector4d radicand_shift_;
        Eigen::Vector4d decay_;
        Eigen::Vector4d noise_scale_;

        Prepared(
            const Eigen::Vector4d& drift,
            const Eigen::Vector4d& radicand,
            const Eigen::Vector4d& decay,
            const Eigen::Vector4d& noise)
            : drift_shift_(drift),
              radicand_shift_(radicand),
              decay_(decay),
              noise_scale_(noise)
        {
        }
    };

    [[nodiscard]] static Prepared prepare(
        double dt,
        const Parameters& parameters)
    {
        detail::validate_time_step4(dt);
        detail::validate_parameters4(
            parameters.long_run_mean,
            parameters.mean_reversion,
            parameters.volatility);

        if ((parameters.long_run_mean.array() < 0.0).any()) {
            throw std::invalid_argument(
                "CIR long-run mean must be nonnegative.");
        }

        Eigen::Vector4d drift = Eigen::Vector4d::Zero();
        Eigen::Vector4d radicand = Eigen::Vector4d::Zero();
        Eigen::Vector4d decay = Eigen::Vector4d::Ones();
        Eigen::Vector4d noise = Eigen::Vector4d::Zero();

        for (int i = 0; i < 4; ++i) {
            const long double kappa = parameters.mean_reversion[i];
            const long double mean = parameters.long_run_mean[i];
            const long double sigma = parameters.volatility[i];

            if (sigma != 0.0L &&
                (kappa == 0.0L || mean == 0.0L)) {
                throw std::invalid_argument(
                    "CIR NV requires zero volatility "
                    "when kappa or mean is zero.");
            }

            const long double split_drift =
                kappa * mean - 0.25L * sigma * sigma;

            if (split_drift < 0.0L) {
                throw std::invalid_argument(
                    "CIR NV requires kappa*mean >= volatility^2/4 "
                    "in every factor.");
            }

            // kappa == 0 forces sigma == 0 in this scheme's domain.
            // The initialized coefficients then represent a constant factor.
            if (kappa == 0.0L) {
                continue;
            }

            const long double product =
                kappa * static_cast<long double>(dt);

            const long double half_integral =
                -std::expm1(-0.5L * product) / kappa;

            drift[i] =
                static_cast<double>(split_drift * half_integral);

            radicand[i] =
                drift[i] *
                static_cast<double>(std::exp(-0.5L * product));

            decay[i] =
                static_cast<double>(std::exp(-product));

            noise[i] = static_cast<double>(
                std::exp(-0.25L * product) *
                0.5L *
                std::sqrt(static_cast<long double>(dt)) *
                sigma);
        }

        return Prepared{drift, radicand, decay, noise};
    }

    // Unchecked in-place transition.
    // Start from a finite nonnegative state and supply finite normal draws.
    // Preparation enforces the split's parameter domain.
    static void step(
        Eigen::Vector4d& state,
        const Prepared& prepared,
        const Eigen::Vector4d& standard_normals) noexcept
    {
        state.array() =
            prepared.drift_shift_.array() +
            ((prepared.radicand_shift_.array() +
              prepared.decay_.array() * state.array()).sqrt() +
             prepared.noise_scale_.array() *
                 standard_normals.array()).square();
    }
};

} // namespace carbonfund::models
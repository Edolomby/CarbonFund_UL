#ifndef CARBON_MODELS_4D_H
#define CARBON_MODELS_4D_H

#include "Eigen/Dense"

#include <cmath>

// Four-factor versions of the carbon-intensity schemes in hedge_cost.cpp.
// They advance the simulated state as-is; a positive-part operation belongs
// only in the separate covariance-penalization calculation.
struct CIRModel4 {
  static void prepare(double h, const Eigen::Vector4d& beta,
                      const Eigen::Vector4d& kappa,
                      const Eigen::Vector4d& s,
                      Eigen::Vector4d& C0, Eigen::Vector4d& C1,
                      Eigen::Vector4d& C2, Eigen::Vector4d& C3) {
    // Same Ninomiya-Victoir coefficients as prm_NV_CIR.
    for (int i = 0; i < 4; ++i) {
      const double ex = std::exp(-kappa[i] * h);
      const double ex2 = std::exp(-0.5 * kappa[i] * h);
      const double ex4 = std::exp(-0.25 * kappa[i] * h);
      const double psi2 = (1.0 - ex2) / kappa[i];
      C0[i] = (kappa[i] * beta[i] - 0.25 * s[i] * s[i]) * psi2;
      C1[i] = C0[i] * ex2;
      C2[i] = ex;
      C3[i] = ex4 * 0.5 * std::sqrt(h) * s[i];
    }
  }

  static void step(Eigen::Vector4d& Y, const Eigen::Vector4d& C0,
                   const Eigen::Vector4d& C1, const Eigen::Vector4d& C2,
                   const Eigen::Vector4d& C3, const Eigen::Vector4d& G) {
    // Same state update as NV_CIR; do not clamp Y here.
    Y.array() = C0.array() +
                ((C1.array() + C2.array() * Y.array()).sqrt() +
                 C3.array() * G.array()).square();
  }
};

struct OUModel4 {
  static void prepare(double h, const Eigen::Vector4d& beta,
                      const Eigen::Vector4d& kappa,
                      const Eigen::Vector4d& s,
                      Eigen::Vector4d& C0, Eigen::Vector4d& C1,
                      Eigen::Vector4d& C2, Eigen::Vector4d& /*C3*/) {
    // Same exact-step coefficients as prm_Exact_OU.
    for (int i = 0; i < 4; ++i) {
      const double ex = std::exp(-kappa[i] * h);
      const double ex2 = std::exp(-2.0 * kappa[i] * h);
      const double psi2 = kappa[i] == 0.0
                              ? h
                              : 0.5 * (1.0 - ex2) / kappa[i];
      C0[i] = beta[i] * (1.0 - ex);
      C1[i] = ex;
      C2[i] = s[i] * std::sqrt(psi2);
    }
  }

  static void step(Eigen::Vector4d& Y, const Eigen::Vector4d& C0,
                   const Eigen::Vector4d& C1, const Eigen::Vector4d& C2,
                   const Eigen::Vector4d& /*C3*/, const Eigen::Vector4d& G) {
    Y.array() = C0.array() + C1.array() * Y.array() + C2.array() * G.array();
  }
};

struct GeoOUModel4 {
  static void prepare(double h, const Eigen::Vector4d& beta,
                      const Eigen::Vector4d& kappa,
                      const Eigen::Vector4d& s,
                      Eigen::Vector4d& C0, Eigen::Vector4d& C1,
                      Eigen::Vector4d& C2, Eigen::Vector4d& C3) {
    // prm_Exact_geoOU uses precisely the same coefficients as prm_Exact_OU.
    OUModel4::prepare(h, beta, kappa, s, C0, C1, C2, C3);
  }

  static void step(Eigen::Vector4d& Y, const Eigen::Vector4d& C0,
                   const Eigen::Vector4d& C1, const Eigen::Vector4d& C2,
                   const Eigen::Vector4d& /*C3*/, const Eigen::Vector4d& G) {
    Y.array() = (C0.array() + C1.array() * Y.array().log() +
                 C2.array() * G.array()).exp();
  }
};

#endif  // CARBON_MODELS_4D_H

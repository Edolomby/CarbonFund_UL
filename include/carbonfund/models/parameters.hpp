#pragma once

#include <Eigen/Core>

namespace carbonfund::models {

// dY = kappa * (mean - Y) dt + volatility dW
struct OUParameters4 {
    Eigen::Vector4d long_run_mean =
        Eigen::Vector4d::Zero();

    Eigen::Vector4d mean_reversion =
        Eigen::Vector4d::Zero();

    Eigen::Vector4d volatility =
        Eigen::Vector4d::Zero();
};

// L = log(Y)
// dL = kappa * (log_mean - L) dt + volatility dW
struct GeoOUParameters4 {
    Eigen::Vector4d long_run_log_mean =
        Eigen::Vector4d::Zero();

    Eigen::Vector4d mean_reversion =
        Eigen::Vector4d::Zero();

    Eigen::Vector4d volatility =
        Eigen::Vector4d::Zero();
};

// dY = kappa * (mean - Y) dt + volatility * sqrt(Y) dW
struct CIRParameters4 {
    Eigen::Vector4d long_run_mean =
        Eigen::Vector4d::Zero();

    Eigen::Vector4d mean_reversion =
        Eigen::Vector4d::Zero();

    Eigen::Vector4d volatility =
        Eigen::Vector4d::Zero();
};

} // namespace carbonfund::models
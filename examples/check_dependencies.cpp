#include <Eigen/Core>
#include <boost/random/xoshiro.hpp>
#include <boost/version.hpp>
#include <carbonfund/utils/TablePrinter.hpp>

#include <string>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

int main()
{
    static_assert(BOOST_VERSION >= 108800);

    const Eigen::Vector4d values{1.0, 2.0, 3.0, 4.0};
    boost::random::xoshiro256pp rng;

    const std::vector<std::string> checks{
        "Boost", "Eigen", "Eigen squared norm", "Xoshiro sample", "OpenMP"};
    std::vector<std::string> results{
        BOOST_LIB_VERSION,
        std::to_string(EIGEN_WORLD_VERSION) + '.' +
            std::to_string(EIGEN_MAJOR_VERSION) + '.' +
            std::to_string(EIGEN_MINOR_VERSION),
        carbonfund::utils::toString(values.squaredNorm()),
        carbonfund::utils::toString(rng())};

#ifdef _OPENMP
    results.push_back(std::to_string(omp_get_max_threads()) + " maximum threads");
#else
    results.emplace_back("disabled");
#endif
    carbonfund::utils::printVectors({"Check", "Result"}, checks, results);

    return values.squaredNorm() == 30.0 ? 0 : 1;
}

#include <Eigen/Core>
#include <boost/random/xoshiro.hpp>
#include <boost/version.hpp>

#include <iostream>

#ifdef _OPENMP
#include <omp.h>
#endif

int main()
{
    static_assert(BOOST_VERSION >= 108800);

    const Eigen::Vector4d values{1.0, 2.0, 3.0, 4.0};
    boost::random::xoshiro256pp rng;

    std::cout << "Boost: " << BOOST_LIB_VERSION << '\n';
    std::cout << "Eigen: "
              << EIGEN_WORLD_VERSION << '.'
              << EIGEN_MAJOR_VERSION << '.'
              << EIGEN_MINOR_VERSION << '\n';

    std::cout << "Eigen squared norm: "
              << values.squaredNorm() << '\n';
    std::cout << "Xoshiro sample: " << rng() << '\n';

#ifdef _OPENMP
    std::cout << "OpenMP maximum threads: "
              << omp_get_max_threads() << '\n';
#else
    std::cout << "OpenMP disabled\n";
#endif

    return values.squaredNorm() == 30.0 ? 0 : 1;
}
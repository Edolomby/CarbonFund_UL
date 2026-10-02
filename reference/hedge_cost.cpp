#include <iostream>
#include <fstream>
#include <chrono>  // for high_resolution_clock
#include <math.h>
#include <omp.h>
#include <boost/random/xoshiro.hpp>
#include <boost/random/random_device.hpp>
#include <boost/random/uniform_real_distribution.hpp>
#include <boost/random/normal_distribution.hpp>
#include <algorithm>
#include <iomanip>
#include <complex>
#include <cmath>
#include <stack>
#include "Eigen/Dense"
#include "Eigen/Eigenvalues"
#include "carbon_models_4d.h"
#include <stdexcept>
//#include <boost/math/distributions/non_central_chi_squared.hpp>
#include <matplot/matplot.h>

// Intel 2026.1.1 crashes when a dynamically initialized thread_local is in an
// OpenMP parallel block. Each Site keeps a separate generator for each thread.
template <int Site>
[[gnu::noinline]] boost::random::xoshiro256pp& parallel_thread_rng() {
  thread_local boost::random::xoshiro256pp gen(boost::random::random_device{}());
  return gen;
}

void writeToCSV(const std::vector<double>& vec1, const std::vector<double>& vec2, 
                const std::vector<double>& vec3, const std::string& filename) 
{
    // Open the file for writing.
    std::ofstream file(filename);

    // Check if the file was opened successfully.
    if (!file.is_open()) {
        std::cerr << "Error: Could not open the file " << filename << std::endl;
        return;
    }

    // Write the header row to the CSV file.
    file << "No_hedge,Static_hedge,Continous_hedge\n";

    // Determine the maximum size of the three vectors.
    size_t maxSize = std::max({vec1.size(), vec2.size(), vec3.size()});

    // Write the data to the file, one row at a time.
    for (size_t i = 0; i < maxSize; ++i) {
        // Write the value from the first vector if it exists.
        if (i < vec1.size()) {
            file << vec1[i];
        }
        file << ",";

        // Write the value from the second vector if it exists.
        if (i < vec2.size()) {
            file << vec2[i];
        }
        file << ",";

        // Write the value from the third vector if it exists.
        if (i < vec3.size()) {
            file << vec3[i];
        }
        file << "\n";
    }

    // Close the file.
    file.close();
    std::cout << "CSV file '" << filename << "' created successfully!" << std::endl;
}




void writeToCSV_CIs_Weights(const std::vector<std::vector<double>> & Weights, 
                           const std::vector<std::vector<double>> & CIs, 
                          const std::string& filename) 
{
  // Open the file for writing.
  std::ofstream file(filename);

  // Determine the maximum size of the three vectors.
  size_t Size_vec_vec = Weights.size();

  size_t Size_vec = Weights[0].size();

  // Check if the file was opened successfully.
  if (!file.is_open()) {
    std::cerr << "Error: Could not open the file " << filename << std::endl;
    return;
  }

  // Write the header row to the CSV file.
  file << "No_hedge, Static_hedge, Continous_hedge\n";
  for(size_t j = 0; j < Size_vec_vec; j++) 
  {
    file << "CI" << std::to_string(j+1)<< ",";
  };

  for(size_t j = 0; j < Size_vec_vec; j++)
  {
    file << "Weight" << std::to_string(j+1);
    if(j<Size_vec_vec-1) file << ",";
    else file << "\n";
  };

  // Write the data to the file, one row at a time.
  for (size_t i = 0; i < Size_vec; i++) 
  { 
    for (size_t j = 0; j < Size_vec_vec; j++)
    {
      if (i < CIs[j].size()) { file << CIs[j][i]; }
      file << ",";
    }
    for (size_t j = 0; j < Size_vec_vec; j++)
    {
      if (i < Weights[j].size()) { file << Weights[j][i]; }
      if(j<Size_vec_vec-1) file << ",";
      else file << "\n";
    }
  }

  // Close the file.
  file.close();
  std::cout << "CSV file '" << filename << "' created successfully!" << std::endl;
}



typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> MatrixXd;
typedef Eigen::Matrix<bool,   Eigen::Dynamic, Eigen::Dynamic> MatrixXb;
typedef Eigen::Matrix<double, Eigen::Dynamic, 1> VectorXd;
typedef Eigen::Array <double, Eigen::Dynamic, 1> ArrayXd;
typedef Eigen::Array <bool,   Eigen::Dynamic, 1> ArrayXb;



struct Output_hedging {
  std::vector<double> no_hedge;
  std::vector<double> static_hedge;
  std::vector<double> trading_gains;
  std::vector<double> hedging_cost;
  std::vector<double> X; //distribution of fund at maturity
  double start_price;
};

struct Scenarios {
  std::vector<std::vector<double>> X;
  std::vector<std::vector<ArrayXd>> Y;

  /**
   * @brief Prints the contents of the Scenarios object to the console.
   *
   * This function iterates through each scenario stored in the 'X' and 'Y'
   * member variables and prints them in a formatted way. It is declared
   * 'const' because it does not modify the state of the object.
   */
  void print() const {
    std::cout << std::fixed << std::setprecision(4); // Set precision for floating-point output

    std::cout << "--- Printing Scenarios ---" << std::endl;

    // Determine the number of scenarios, assuming X and Y have the same outer size.
    size_t num_scenarios = X.size();
    if (num_scenarios == 0) {
      std::cout << "No scenarios to print." << std::endl;
      return;
    }

    // Loop through each scenario
    for (size_t i = 0; i < num_scenarios; ++i) {
      std::cout << "\n==========================" << std::endl;
      std::cout << "       Scenario " << i + 1 << std::endl;
      std::cout << "==========================" << std::endl;

      // --- Print X data for the current scenario ---
      std::cout << "\n--- Data X ---" << std::endl;
      if (i < X.size() && !X[i].empty()) {
        for (size_t j = 0; j < X[i].size(); ++j) {
          std::cout << "X[" << i << "][" << j << "]: " << X[i][j] << std::endl;
        }
      } else {
        std::cout << "No X data for this scenario." << std::endl;
      }

      // --- Print Y data for the current scenario ---
      std::cout << "\n--- Data Y ---" << std::endl;
      if (i < Y.size() && !Y[i].empty()) {
        for (size_t j = 0; j < Y[i].size(); ++j) {
          std::cout << "Y[" << i << "][" << j << "]:\n" << Y[i][j].transpose() << std::endl;
        }
      } else {
        std::cout << "No Y data for this scenario." << std::endl;
      }
    }
     std::cout << "\n--- End of Scenarios ---" << std::endl;
  }
};



struct PETIEI_polyciholders{
  std::vector<double> age_PE;
  std::vector<ArrayXd> nominals_PE;
  std::vector<ArrayXb> is_alive_PE;
  std::vector<double> nominals_sum_PE;
  std::vector<double> age_TI;
  std::vector<ArrayXd> nominals_TI;
  std::vector<ArrayXb> is_alive_TI;
  std::vector<double> nominals_sum_TI;
  std::vector<double> age_EI;
  std::vector<ArrayXd> nominals_EI;
  std::vector<ArrayXb> is_alive_EI;
  std::vector<double> nominals_sum_EI;

  // --- Constructors ---

  /**
   * @brief Default constructor.
   * Initializes all member vectors as empty.
   * We must explicitly default it because we have a custom constructor.
   */
  PETIEI_polyciholders() = default;

  /**
   * @brief Custom constructor to initialize data.
   * * @param age_PE_in Input vector for PE ages.
   * @param age_TI_in Input vector for TI ages.
   * @param age_EI_in Input vector for EI ages.
   * @param nom_PE    Total nominal value for the PE group.
   * @param nom_TI    Total nominal value for the TI group.
   * @param nom_EI    Total nominal value for the EI group.
   * @param n_PE      Number of policyholders in the PE group.
   * @param n_TI      Number of policyholders in the TI group.
   * @param n_EI      Number of policyholders in the EI group.
   */
  PETIEI_polyciholders(
    std::vector<double> age_PE_in,
    std::vector<double> age_TI_in,
    std::vector<double> age_EI_in,
    double nom_PE,
    double nom_TI,
    double nom_EI,
    int n_PE,
    int n_TI,
    int n_EI)
    // Use std::move to efficiently transfer ownership of the input vectors
    : age_PE(std::move(age_PE_in)),
      age_TI(std::move(age_TI_in)),
      age_EI(std::move(age_EI_in))
  {
    // --- Initialize PE Group ---
    if (!age_PE.empty()) {
      // Calculate the nominal sum value
      double value_pe = nom_PE / age_PE.size();
      nominals_sum_PE.assign(age_PE.size(), value_pe);
      
      // Initialize is_alive_PE
      ArrayXb alive_template_PE(n_PE);
      alive_template_PE.setConstant(true);
      is_alive_PE.assign(age_PE.size(), alive_template_PE);

      // Initialize nominals_PE
      if (n_PE > 0) {
        double nominal_value = nom_PE / age_PE.size() / n_PE;
        ArrayXd nominals_template_PE(n_PE);
        nominals_template_PE.setConstant(nominal_value);
        nominals_PE.assign(age_PE.size(), nominals_template_PE);
      }
    }

    // --- Initialize TI Group ---
    if (!age_TI.empty()) {
      double value_ti = nom_TI / age_TI.size();
      nominals_sum_TI.assign(age_TI.size(), value_ti);

      // Initialize is_alive_TI
      ArrayXb alive_template_TI(n_TI);
      alive_template_TI.setConstant(true);
      is_alive_TI.assign(age_TI.size(), alive_template_TI);

      // Initialize nominals_TI
      if (n_TI > 0) {
        double nominal_value = nom_TI / age_TI.size() / n_TI;
        ArrayXd nominals_template_TI(n_TI);
        nominals_template_TI.setConstant(nominal_value);
        nominals_TI.assign(age_TI.size(), nominals_template_TI);
      }
    }

    // --- Initialize EI Group ---
    if (!age_EI.empty()) {
      double value_ei = nom_EI / age_EI.size();
      nominals_sum_EI.assign(age_EI.size(), value_ei);

      // Initialize is_alive_EI
      ArrayXb alive_template_EI(n_EI);
      alive_template_EI.setConstant(true);
      is_alive_EI.assign(age_EI.size(), alive_template_EI);

      // Initialize nominals_EI
      if (n_EI > 0) {
        double nominal_value = nom_EI / age_EI.size() / n_EI;
        ArrayXd nominals_template_EI(n_EI);
        nominals_template_EI.setConstant(nominal_value);
        nominals_EI.assign(age_EI.size(), nominals_template_EI);
      }
    }
  }

  // Optional: A helper function to print the state of the struct
  void print_summary() const {
    std::cout << "--- Struct Summary ---" << std::endl;
    std::cout << "PE Group:" << std::endl;
    std::cout << "  Age vector size: " << age_PE.size() << std::endl;
    std::cout << "  Nominals vector size: " << nominals_PE.size() << std::endl;
    if (!nominals_PE.empty()) {
        std::cout << "    -> Each 'Nominals' array has size: " << nominals_PE[0].size() 
                  << " with value: " << nominals_PE[0][0] << std::endl;
    }
    std::cout << "  Is Alive vector size: " << is_alive_PE.size() << std::endl;
    if (!is_alive_PE.empty()) {
        std::cout << "    -> Each 'Is Alive' array has size: " << is_alive_PE[0].size() << std::endl;
    }
    std::cout << "  Nominals Sum vector size: " << nominals_sum_PE.size() << std::endl;
    std::cout << "  Nominals Sum value: " << (nominals_sum_PE.empty() ? "N/A" : std::to_string(nominals_sum_PE[0])) << std::endl;
    
    std::cout << "\nTI Group:" << std::endl;
    std::cout << "  Age vector size: " << age_TI.size() << std::endl;
    std::cout << "  Nominals vector size: " << nominals_TI.size() << std::endl;
    if (!nominals_TI.empty()) {
        std::cout << "    -> Each 'Nominals' array has size: " << nominals_TI[0].size() 
                  << " with value: " << nominals_TI[0][0] << std::endl;
    }
    std::cout << "  Is Alive vector size: " << is_alive_TI.size() << std::endl;
    if (!is_alive_TI.empty()) {
        std::cout << "    -> Each 'Is Alive' array has size: " << is_alive_TI[0].size() << std::endl;
    }
    std::cout << "  Nominals Sum vector size: " << nominals_sum_TI.size() << std::endl;
    std::cout << "  Nominals Sum value: " << (nominals_sum_TI.empty() ? "N/A" : std::to_string(nominals_sum_TI[0])) << std::endl;

    std::cout << "\nEI Group:" << std::endl;
    std::cout << "  Age vector size: " << age_EI.size() << std::endl;
    std::cout << "  Nominals vector size: " << nominals_EI.size() << std::endl;
    if (!nominals_EI.empty()) {
        std::cout << "    -> Each 'Nominals' array has size: " << nominals_EI[0].size() 
                  << " with value: " << nominals_EI[0][0] << std::endl;
    }
    std::cout << "  Is Alive vector size: " << is_alive_EI.size() << std::endl;
    if (!is_alive_EI.empty()) {
        std::cout << "    -> Each 'Is Alive' array has size: " << is_alive_EI[0].size() << std::endl;
    }
    std::cout << "  Nominals Sum vector size: " << nominals_sum_EI.size() << std::endl;
    std::cout << "  Nominals Sum value: " << (nominals_sum_EI.empty() ? "N/A" : std::to_string(nominals_sum_EI[0])) << std::endl;
    std::cout << "----------------------" << std::endl;
  }
};



template<typename T1,typename T2>
std::pair<double,double> lreg(std::vector<T1> X, std::vector<T2> Y, bool loglog = false)
{ 
  std::pair<double,double> Res;
  if(loglog)
  {
		std::transform(X.begin(), X.end(), X.begin(), [&](T1 &n){ return std::log(n); });
		std::transform(Y.begin(), Y.end(), Y.begin(), [&](T2 &n){ return std::log(n); });
  }
	double meanX = double(std::accumulate(X.begin(),X.end(),0.0)) / X.size();
	double meanY = double(std::accumulate(Y.begin(),Y.end(),0.0)) / Y.size();
	double varX  = std::inner_product(X.begin(),X.end(),X.begin(),0.0);
	double covXY = std::inner_product(X.begin(),X.end(),Y.begin(),0.0);

	covXY -= X.size()*meanX*meanY;
	varX  -= X.size()*meanX*meanX; 
	Res.first  = covXY/varX;
	Res.second = meanY-Res.first*meanX;

	return Res;
}

template <typename T>
void print_vector(std::vector<T> V, int i = -999999)
{
  if (i == -999999)
  {
    for (T entry : V)
      std::cout << entry << std::endl;
  }
  else
  {
    if (i > V.size() || i < 0)
    {
      std::cout << "Error: the number of coordinates requested exceed the size of vector or is negative" << std::endl;
    }
    else if (i < V.size())
    {
      for (int j = 0; j < i; j++)
        std::cout << V[j] << std::endl;
    }
  }
}



double min_vec(std::vector<double> const &A)
{
  double min =  A[0];
  for (int i = 1; i < A.size(); i++)
  {
    if(min>A[i]) min=A[i];
  }
  return min;
}

std::vector<double> operator+(std::vector<double> const &A, double const &B)
{
  std::vector<double> C(A.size());
  for (int i = 0; i < A.size(); i++)
  {
    C[i] = A[i] + B;
  }
  return C;
}

std::vector<double> operator+(double const &B, std::vector<double> const &A)
{
  std::vector<double> C(A.size());
  for (int i = 0; i < A.size(); i++)
  {
      C[i] = A[i] + B;
  }
  return C;
}

std::vector<double> operator+(std::vector<double> const &A, std::vector<double> const &B)
{
  std::vector<double> C(A.size());
  for (int i = 0; i < A.size(); i++)
  {
    C[i] = A[i] + B[i];
  }
  return C;
}

std::vector<double> operator*(std::vector<double> const &A, double scalar)
{
  std::vector<double> C(A.size());
  for (int i = 0; i < A.size(); i++)
  {
    C[i] = A[i] * scalar;
  }
  return C;
}

std::vector<double>& operator+=(std::vector<double>& vec, double scalar) {
  // Iterate through each element of the vector
  for (double& element : vec) {
      // Add the scalar value to the current element
      element += scalar;
  }
  // Return a reference to the modified vector to allow chaining (e.g., vec += 1.0 += 2.0;)
  return vec;
}

double sum(std::vector<double> const &A)
{ 
  double sum_vec = 0.0;
  for(int i=0;i<A.size();i++) sum_vec += A[i];
  return sum_vec;
}

double mean(std::vector<double> const &A)
{ 
  double mean = 0.0;
  for(int i=0;i<A.size();i++) mean += A[i];
  mean /= A.size();
  return mean;
}

double stddev(std::vector<double> const &A)
{ 
  double mean = 0.0;
  double var  = 0.0;
  int M = A.size();
  for(int i=0;i<M;i++)
  {
    mean += A[i];
    var  += A[i]*A[i];
  }
  mean /= M;
  if(M>1) return sqrt( (var - M * mean * mean) / (M-1.0) ); // unbiased
  else    return sqrt( (var - M * mean * mean) /  M      );
}

//return empirical var and its empirical 95CI
std::pair<double,double>
empirical_var_CI95(std::vector<double> const &A)
{
  double first=0.0, second=0.0, third=0.0, fourth=0.0;
  int M = A.size();
  for(int i=0;i<M;i++)
  {
    first  += A[i];
    second += A[i]*A[i];
    third  += A[i]*A[i]*A[i];
    fourth += A[i]*A[i]*A[i]*A[i];
  }
  first  /= M;
  second /= M;
  third  /= M;
  fourth /= M;

  double var = second - first*first;
  double varvar = fourth -4.*first*third -second*second 
        +8.*first*first*second -4.*first*first*first*first;

	return std::make_pair(var, 1.96*sqrt(varvar/M));
}


// normcdf
double normcdf(double z) {
  // The constant 1/sqrt(2) is pre-calculated for efficiency
  // erfc algo slower but than erf BUT ESSENTIAL FOR PRECISION
  return 0.5 * (std::erfc(-z * 0.7071067811865475244));
}

double normpdf(double z)
{ 
  // 1./ sqrt(2.*M_PI)  = 0.39894228040143270286
	return 0.39894228040143270286 * exp(-0.5*z*z);
} 

//high quality
double normcdfinv_as241(double p) 
{
  
  double q, r, num, den, res;

  q = p - 0.5;
  if (fabs(q) <= 0.425) {
    r = 0.180625 - q*q;
                  
    num =         2.5090809287301226727e+3;
    num = r*num + 3.3430575583588128105e+4;
    num = r*num + 6.7265770927008700853e+4;
    num = r*num + 4.5921953931549871457e+4;
    num = r*num + 1.3731693765509461125e+4;
    num = r*num + 1.9715909503065514427e+3;
    num = r*num + 1.3314166789178437745e+2;
    num = r*num + 3.3871328727963666080e0;

    den =         5.2264952788528545610e+3;
    den = r*den + 2.8729085735721942674e+4;
    den = r*den + 3.9307895800092710610e+4;
    den = r*den + 2.1213794301586595867e+4;
    den = r*den + 5.3941960214247511077e+3;
    den = r*den + 6.8718700749205790830e+2;
    den = r*den + 4.2313330701600911252e+1;
    den = r*den + 1.0000000000e+00;

    res = q * num / den; 

    return res;
  }

  else {

    if (q < 0.0)
      r = p;
    else
      r = 1.0 - p;

    r = sqrt(-log(r));

    if (r <= 5.0) {
      r = r - 1.6;

      num =         7.74545014278341407640e-4;
      num = r*num + 2.27238449892691845833e-2;
      num = r*num + 2.41780725177450611770e-1;
      num = r*num + 1.27045825245236838258e0;
      num = r*num + 3.64784832476320460504e0;
      num = r*num + 5.76949722146069140550e0;
      num = r*num + 4.63033784615654529590e0;
      num = r*num + 1.42343711074968357734e0;

      den =         1.05075007164441684324e-9;
      den = r*den + 5.47593808499534494600e-4;
      den = r*den + 1.51986665636164571966e-2;
      den = r*den + 1.48103976427480074590e-1;
      den = r*den + 6.89767334985100004550e-1;
      den = r*den + 1.67638483018380384940e0;
      den = r*den + 2.05319162663775882187e0;
      den = r*den + 1.0000000000e+00;

      res = num / den;
    }

    else {
      r = r - 5.0;

      num =         2.01033439929228813265e-7;
      num = r*num + 2.71155556874348757815e-5;
      num = r*num + 1.24266094738807843860e-3;
      num = r*num + 2.65321895265761230930e-2;
      num = r*num + 2.96560571828504891230e-1;
      num = r*num + 1.78482653991729133580e0;
      num = r*num + 5.46378491116411436990e0;
      num = r*num + 6.65790464350110377720e0;

      den =         2.04426310338993978564e-15;
      den = r*den + 1.42151175831644588870e-7;
      den = r*den + 1.84631831751005468180e-5;
      den = r*den + 7.86869131145613259100e-4;
      den = r*den + 1.48753612908506148525e-2;
      den = r*den + 1.36929880922735805310e-1;
      den = r*den + 5.99832206555887937690e-1;
      den = r*den + 1.0000000000e+00;

      res = num / den;
    }

    if (q < 0.0)
      res = - res;

    return res;
  }
}


// Rational inverse-standard-normal approximation due to Peter Acklam; requires 0 < p < 1.
double normcdfinv_acklam(double p)
{
  constexpr double a1=-3.969683028665376e+01, a2=2.209460984245205e+02;
  constexpr double a3=-2.759285104469687e+02, a4=1.383577518672690e+02;
  constexpr double a5=-3.066479806614716e+01, a6=2.506628277459239e+00;
  constexpr double b1=-5.447609879822406e+01, b2=1.615858368580409e+02;
  constexpr double b3=-1.556989798598866e+02, b4=6.680131188771972e+01;
  constexpr double b5=-1.328068155288572e+01;
  constexpr double c1=-7.784894002430293e-03, c2=-3.223964580411365e-01;
  constexpr double c3=-2.400758277161838e+00, c4=-2.549732539343734e+00;
  constexpr double c5=4.374664141464968e+00, c6=2.938163982698783e+00;
  constexpr double d1=7.784695709041462e-03, d2=3.224671290700398e-01;
  constexpr double d3=2.445134137142996e+00, d4=3.754408661907416e+00;

  if (p < 0.02425) {
    const double q = std::sqrt(-2.0*std::log(p));
    return (((((c1*q+c2)*q+c3)*q+c4)*q+c5)*q+c6) /
           ((((d1*q+d2)*q+d3)*q+d4)*q+1.0);
  }
  if (p > 0.97575) {
    const double q = std::sqrt(-2.0*std::log(1.0-p));
    return -(((((c1*q+c2)*q+c3)*q+c4)*q+c5)*q+c6) /
            ((((d1*q+d2)*q+d3)*q+d4)*q+1.0);
  }
  const double q = p-0.5;
  const double z = q*q;
  return (((((a1*z+a2)*z+a3)*z+a4)*z+a5)*z+a6)*q /
         (((((b1*z+b2)*z+b3)*z+b4)*z+b5)*z+1.0);
}


using prm_comp = void (*)(double t, const ArrayXd & beta, const ArrayXd & kappa, const ArrayXd & s,  
						ArrayXd & C0, ArrayXd & C1, ArrayXd & C2, ArrayXd & C3);

using scheme_update = void (*)(ArrayXd & Y, ArrayXd & C0, ArrayXd & C1, 
                                ArrayXd & C2, ArrayXd & C3, ArrayXd & G);



void prm_NV_CIR(double t, const ArrayXd & beta, const ArrayXd & kappa, const ArrayXd & s,  
						ArrayXd & C0, ArrayXd & C1, ArrayXd & C2, ArrayXd & C3)
{
	// parameters for NV scheme only for under "weak feller condition"
	for(int i=0;i<C0.size();i++)
	{	
		double ex   = exp(-     kappa[i]*t);
		double ex2  = exp(-0.5 *kappa[i]*t);
		double ex4  = exp(-0.25*kappa[i]*t);
		double psi2 = (1-ex2)/kappa[i];
		C0[i] = (kappa[i]*beta[i] - 0.25*s[i]*s[i]) * psi2;
		C1[i] = C0[i]*ex2;
		C2[i] = ex;
		C3[i] = ex4 * 0.5*sqrt(t)*s[i];
	}
}

void NV_CIR(ArrayXd & Y, ArrayXd & C0, ArrayXd & C1, ArrayXd & C2, ArrayXd & C3, ArrayXd & G)
{
	// Ninomya-Victoir scheme under "weak feller condition" of multidim CIR in generic dim 
	Y = C0 + Eigen::square( Eigen::sqrt( C1+C2*Y ) + C3*G ) ;
}

void prm_Exact_geoOU(double t, const ArrayXd & beta, const ArrayXd & kappa, const ArrayXd & s,  
						ArrayXd & C0, ArrayXd & C1, ArrayXd & C2, ArrayXd & /*C3*/)
{
	// parameters for exact scheme of OU process
	for(int i=0;i<C0.size();i++)
	{	
		double ex   = exp(-  kappa[i]*t);
		double ex2  = exp(-2*kappa[i]*t);
    double psi2 = (kappa[i]==0.0) ? t : 0.5*(1-ex2)/kappa[i];

		C0[i] = beta[i] * ( 1 - ex);
		C1[i] = ex;
		C2[i] = s[i]*sqrt(psi2);
	}
}

void GeoOU_Exact(ArrayXd & Y, ArrayXd & C0, ArrayXd & C1, ArrayXd & C2, ArrayXd & /*C3*/, ArrayXd & G)
{ 
  // TO BE CHECKED AND OPTIMIZED!
	// Exact scheme of geoOU process in generic dim
	Y = Eigen::exp(C0 + C1*Eigen::log(Y) + C2*G); 
}

void prm_Exact_OU(double t, const ArrayXd & beta, const ArrayXd & kappa, const ArrayXd & s,  
						ArrayXd & C0, ArrayXd & C1, ArrayXd & C2, ArrayXd & /*C3*/)
{
	// parameters for exact scheme of OU process
	for(int i=0;i<C0.size();i++)
	{	
		double ex   = exp(-  kappa[i]*t);
		double ex2  = exp(-2*kappa[i]*t);
    double psi2 = (kappa[i]==0.0) ? t : 0.5*(1-ex2)/kappa[i];

		C0[i] = beta[i] * ( 1 - ex);
		C1[i] = ex;
		C2[i] = s[i]*sqrt(psi2);
	}
}





void OU_Exact(ArrayXd & Y, ArrayXd & C0, ArrayXd & C1, ArrayXd & C2, ArrayXd & /*C3*/, ArrayXd & G)
{
	// Exact scheme of OU process in generic dim
	Y = C0 + C1*Y + C2*G;  
}


double my_cir_mom(int L, double x, double t, double a, double b, double s)
{ 
  double v = 2*a/s/s;
  double ex = exp(-b*t);
  double psi_star = s*s*(1-ex)/b/2;
  double B = 1;
  double prod = 1;
  double A[L+1];
  
  double bin_coeff = 1;
  A[0] = 1;
  for(int l=1;l<L+1;l++)
  {
    A[l] = A[l-1]*ex*x;
  }

  double sum = A[L];
  //std::cout << "sum = " << sum << std::endl;
  for(int l=L-1;l>0;l--)
  {
    B *= psi_star;
    bin_coeff = bin_coeff * (l+1) / (L-l);
    prod *= (l+v);
    //std::cout << "prod = " << prod << std::endl;
    sum += bin_coeff*prod*A[l]*B;
    //std::cout << "sum = " << sum << std::endl;
  }

  prod *= v;
  sum += prod*B*psi_star;
  //std::cout << "sum = " << sum << std::endl;
	return sum;
}


/////////////////////////////////////////////////////////
/////// CIR with long-term mean time dependent /////////
///////////////////////////////////////////////////////
////////////
// this type of dynamic is used to model the evolution of the intensity in the SGM model
// is a 1D model so no need of array but we use std::vector because of time dependency

struct prm_TDCIR {
  std::vector<double> C0;
  std::vector<double> C1;
              double  C2;
              double  C3;

  // --- Constructors ---
  prm_TDCIR() = default;

  prm_TDCIR(unsigned N, 
              double time_window,  
              double k, 
              double sigma,
              double curr_age,
              double phi,
              double m,
              double b)
  {
    // Now, calculate the rest and populate the vectors
    double h = time_window / N;
    double ex  =  std::exp(-       k*h);
    double ex2  = std::exp(-0.5  * k*h);
    double ex4  = std::exp(-0.25 * k*h);

    double psi2 = (1.0-ex2)/k;

    C2 = ex;
    C3 = 0.5*ex4*std::sqrt(h)*sigma;


    double C0_base = (k*phi -0.25*sigma*sigma)*psi2 ;
    double C1_base = C0_base*ex2;

    double zeta = k*b/(k+b) * std::exp(b*(curr_age-m)) * ( std::exp(0.5*b*h) - ex2 );
  
    // Reserve memory for efficiency and then fill the vectors
    C0.reserve(N);
    C1.reserve(N);

    for(int i=0; i < N; i++)
    { 
      C0.push_back(C0_base + std::exp( b*(i+0.5)*h )*zeta);
      C1.push_back(C1_base + std::exp( b* i     *h )*zeta * ex2);
    }
  }
  
};


double TDNV_CIR(double Y, unsigned n_step, prm_TDCIR &prm, double G)
{
	// Ninomya-Victoir timpe dependendt scheme under "generalized weak feller condition" 
  /* std::cout << "ciao " << std::endl;
  std::cout << "n_step =" << n_step << std::endl;
  std::cout << "prm.C1[n_step] =" << prm.C1[n_step] << std::endl;
  std::cout << "prm.C0[n_step] =" << prm.C0[n_step] << std::endl; */
  double fact =  std::sqrt(prm.C1[n_step] + prm.C2*Y) + prm.C3*G;
	return prm.C0[n_step] + fact*fact;
}


/////////////////////////////////////////////
///////// Gompertz - Makeham law ///////////
///////////////////////////////////////////

//Refactored b_GM = 1/b_GM of older versions

double GM_survival_proba(double curr_age, double phi_GM, double m_GM, double b_GM, double time_window)
{
  return exp( -phi_GM*time_window -exp((curr_age-m_GM)*b_GM)*(exp(time_window*b_GM) -1.) );
}

double GM_survival_proba_discounted(double curr_age, double phi_GM, double m_GM, double b_GM, double time_window, double r)
{
  return exp( -(r+phi_GM)*time_window -exp((curr_age-m_GM)*b_GM)*(exp(time_window*b_GM) -1.) );
}

// not used will be deleted in further release
double GM_spXmi(double curr_age, double phi_GM, double m_GM, double b_GM, double time_window)
{ 
  //survival_proba x mortality intensity
  return (phi_GM + exp( (curr_age+time_window-m_GM)*b_GM ) * b_GM) * 
          exp( -phi_GM*time_window -exp((curr_age-m_GM)*b_GM)*(exp(time_window*b_GM) -1.) );
}

double GM_spXmi_discounted(double curr_age, double phi_GM, double m_GM, double b_GM, double time_window, double r)
{ 
  //survival_proba x mortality intensity x discounting
  // r is the risk free rate
  return (phi_GM + exp( (curr_age+time_window-m_GM)*b_GM ) * b_GM) * 
          exp( -(r+phi_GM)*time_window -exp((curr_age-m_GM)*b_GM) * (exp(time_window*b_GM) -1.) );
}

double 
GM_sp_discounted_mc(const std::vector<double>  &nominals, const std::vector<double> &curr_ages, 
                    double phi_GM, double m_GM, double b_GM, double time_window, double r)
{ 
  //multiclasses //several classes different ages
  //survival_proba x discounting
  // r is the risk free rate
  double output = 0.0;
  for(int i=0;i<nominals.size();i++)
    output += nominals[i] * 
            exp( -(r+phi_GM)*time_window -exp((curr_ages[i]-m_GM)*b_GM) * (exp(time_window*b_GM) -1.) );
          
  return output;
}

double 
GM_spXmi_discounted_mc(const std::vector<double> &nominals, const std::vector<double> &curr_ages,
                    double phi_GM, double m_GM, double b_GM, double time_window, double r)
{ 
  //multiclasses //several classes different ages
  //survival_proba x mortality intensity x discounting
  // r is the risk free rate
  double output = 0.0;
  for(int i=0;i<nominals.size();i++)
    output += nominals[i] * (phi_GM + exp( (curr_ages[i]+time_window-m_GM)*b_GM ) * b_GM) * 
          exp( -(r+phi_GM)*time_window -exp((curr_ages[i]-m_GM)*b_GM) * (exp(time_window*b_GM) -1.) );
          
  return output;
}


////////////////////////////////////////////////////////
///////// Stochastic Gompertz - Makeham law ///////////
//////////////////////////////////////////////////////

// no close formula -> integration via simpson 1/3 rule (function "super" regular)
double 
SGM_sp_discounted(double curr_age, double gam0, double k_SGM, double sigma_SGM, 
               double phi_SGM, double m_SGM, double b_SGM, double time_window, double r, unsigned N = 1000)
{ 
  const double d = std::sqrt(k_SGM*k_SGM + 2.*sigma_SGM*sigma_SGM);
  
  // Ensure N (number of intervals) is even for Simpson's rule
  if (N % 2 != 0) N += 1;

  const ArrayXd time_grid = ArrayXd::LinSpaced(N + 1, 0.0, time_window);

  const ArrayXd exp_term_minus_1 = (d*time_grid).exp() - 1.0;
  const ArrayXd C = 2.0 * exp_term_minus_1 / ((d + k_SGM) * exp_term_minus_1 + 2.0 * d);
  const ArrayXd gam_bar = phi_SGM + b_SGM * (b_SGM * (curr_age - m_SGM + time_window-time_grid)).exp();

  const ArrayXd product = C * gam_bar;

  // Simpson with weights
  ArrayXd weights = Eigen::ArrayXd::Constant(N + 1, 2.0); // Start with all 2s

  // Set the alternating 4s for the odd-indexed elements
  weights(Eigen::seq(1, N + 1 - 2, 2)) = 4.0;

  // Set the first and last elements to 1
  weights(0) = 1.0;
  weights(N) = 1.0;

  const double integral = time_window / (3.0* N) * (product * weights).sum();

  return std::exp( -r*time_window -k_SGM * integral - C(C.size()-1) * gam0);
}


// fast_algo partially integrated the C(s,T) function, 
// should be rechecked to guaranty reliability for more extreme values
std::vector<double> 
SGM_spXmi_discounted(double curr_age, double gam0, double k_SGM, double sigma_SGM, double phi_SGM, double m_SGM, double b_SGM, 
                     double time_window, double r,  unsigned N_time, unsigned K = 8)
{ 
  std::vector<double> results(N_time+1);
  const double d = std::sqrt(k_SGM*k_SGM + 2.*sigma_SGM*sigma_SGM);
  // Ensure N (number of intervals) is even for Simpson's rule
  if      (K     == 0) K  = 2;
  else if (K % 2 != 0) K += 1;

  unsigned N = N_time * K;
  
  const unsigned n_points = N + 1; // Total number of points must be odd

  const ArrayXd time_grid = ArrayXd::LinSpaced(n_points, 0.0, time_window);

  const ArrayXd exp_term_minus_1 = (d*time_grid).exp() - 1.0;

  const ArrayXd common_denom = (d + k_SGM) * exp_term_minus_1 + 2.0 * d;

  const ArrayXd C = 2.0 * exp_term_minus_1 / common_denom;

  const ArrayXd E = 4.0 * d*d * (exp_term_minus_1 + 1.0) / common_denom.square();

  const ArrayXd A = (- b_SGM * time_grid).exp();

  // Simpson with weights
  ArrayXd weights = Eigen::ArrayXd::Constant(n_points, 2.0); // Start with all 2s

  // Set the alternating 4s for the odd-indexed elements
  weights(Eigen::seq(1, n_points - 2, 2)) = 4.0;

  // Set the first and last elements to 1
  weights(0) = 1.0;
  weights(N) = 1.0;

  const double c_0 = 0.5*(d-k_SGM) / d;
  const double c_1 = 0.5*(d+k_SGM) / d;

  const double c_2 = -2.0*k_SGM*phi_SGM / (sigma_SGM*sigma_SGM);
  const double c_3 =  2.0*k_SGM*phi_SGM / (d-k_SGM);

  results[0] = gam0;

  double h = time_window / N;

  const double c_4 = b_SGM * std::exp( b_SGM * (curr_age -m_SGM) ) * (h/3.0);  //rescaled with h/3.0

  const ArrayXd integrand_C = weights * C * A;
  const ArrayXd integrand_E = weights * E * A;

  double par_sum_C = integrand_C.head(1+K).sum();
  double par_sum_E = integrand_E.head(1+K).sum();

  double fact = c_4*exp(b_SGM*h*K);

  double simps_int_C = fact * (par_sum_C - C(K) * A(K)); // we correct partial_sums weights are not corrects for partialvectors

  double simps_int_E = fact * (par_sum_E - E(K) * A(K)); // we correct partial_sums weights are not corrects for partialvectors

  results[1] =  std::exp( -k_SGM*simps_int_C - C(K)*gam0 
                      + c_2*std::log( c_0 + c_1*( exp_term_minus_1(K) + 1.0 ) ) + c_3*time_grid(K) ) 
                    * (k_SGM * ( simps_int_E + phi_SGM*C(K) ) + E(K)*gam0);
  
  for(unsigned i=2; i<results.size();i++)
  {

    unsigned idx1 = (i-1) * K + 1;
    unsigned idx2 =  i    * K;

    par_sum_C += integrand_C.segment(idx1, K).sum();
    par_sum_E += integrand_E.segment(idx1, K).sum();

    fact = c_4*exp(b_SGM*h*idx2);

    if(i<results.size()-1)
    {
      simps_int_C = fact * (par_sum_C - C(idx2) * A(idx2)); // we correct partial_sums weights are not corrects for partialvectors
      simps_int_E = fact * (par_sum_E - E(idx2) * A(idx2)); // we correct partial_sums weights are not corrects for partialvectors
    }
    else
    {
      simps_int_C = fact * par_sum_C ; 
      simps_int_E = fact * par_sum_E ;
    }

  
    results[i] =  std::exp( -r*idx2*h -k_SGM*simps_int_C - C(idx2)*gam0 
                      + c_2*std::log( c_0 + c_1*( exp_term_minus_1(idx2) + 1.0 ) ) + c_3*time_grid(idx2) ) 
                    * (k_SGM * ( simps_int_E + phi_SGM*C(idx2) ) + E(idx2)*gam0);

    /* std::cout << "\nidx = " << idx;
    std::cout << "\n-k_SGM*simps_int_C = " << -k_SGM*simps_int_C;
    std::cout << "\n- C(idx-1)*gam0 = " << - C(idx-1)*gam0;
    std::cout << "\nc_2*std::log( c_0 + c_1*( exp_term_minus_1(idx-1) + 1.0 ) ) = " << c_2*std::log( c_0 + c_1*( exp_term_minus_1(idx-1) + 1.0 ) );
    std::cout << "\nc_3*time_grid(idx-1) = " << c_3*time_grid(idx-1) << "\n"; */
  }
  return results;
}


// could be optimized maybe
double 
SGM_sp_discounted_mc(std::vector<double> nominals, std::vector<double> curr_ages,
              double gam0, double k_SGM, double sigma_SGM, double curr_age, double phi_SGM, double m_SGM, double b_SGM, 
              double r, double time_window, unsigned N_time = 1000)
{ 
  //multiclasses //several classes different ages
  //survival_proba x mortality intensity x discounting
  // r is the risk free rate
  double output = 0.0;

  for(unsigned i=0;i<nominals.size();i++)
    output += nominals[i] * SGM_sp_discounted(gam0, k_SGM, sigma_SGM, curr_ages[i], phi_SGM, m_SGM, b_SGM, 
                                              time_window, r,  N_time);
          
  return output;
}


// could be optimized maybe
std::vector<double> 
SGM_spXmi_discounted_mc(std::vector<double> nominals, std::vector<double> curr_ages,
              double gam0, double k_SGM, double sigma_SGM, double curr_age, double phi_SGM, double m_SGM, double b_SGM, 
              double r, double time_window, unsigned N_time, unsigned K = 8)
{ 
  //multiclasses //several classes different ages
  //survival_proba x mortality intensity x discounting
  // r is the risk free rate
  std::vector<double> output(N_time+1,0.0);

  for(unsigned i=0;i<nominals.size();i++)
  { 
    std::vector<double> res = SGM_spXmi_discounted(gam0, k_SGM, sigma_SGM, curr_ages[i], phi_SGM, m_SGM, b_SGM, 
                     time_window, r,  N_time, K);

    for(unsigned j=0; j<res.size(); j++) output[j] += nominals[i] * res[j];
  }
          
  return output;
}


////////////////////////////////////////////
////////// Pricing routines ///////////////
//////////////////////////////////////////


void PE_NOVR_min_max_GM_MMM(double CRRA_delta, double X0, double r, ArrayXd mu, ArrayXd vols, MatrixXd cov_mat, 
									double alpha_penalty, ArrayXd Y0, ArrayXd beta, ArrayXd kappa, ArrayXd s,
                  double current_age, double phi_GM, double m_GM, double b_GM,  
                  double min_gua, double max_gua, double TtoMat, int N, int M,
                  double &price, double &var_price, double &delta, double &var_delta)
{	
  //NO VARIANCE REDUCTION FORMULA
  // Pure Endowment option with min and max guaranteed threshold 
  // option evalueted under the minimal martingale probability
	VectorXd mu_mod = (mu-r).matrix();
	ArrayXd vols_sq_penalised = alpha_penalty*vols.square();
	MatrixXd d_cov_mat = CRRA_delta * cov_mat;
	double h = TtoMat/double(N);
	auto dim = (beta.size()); 
	ArrayXd C0(dim), C1(dim), C2(dim), C3(dim);
	prm_NV_CIR(h, beta, kappa, s, C0, C1, C2, C3); //setting parameters for NV scheme for CIR

	ArrayXd e_CI = Y0 * vols_sq_penalised;

	// for large n this is faster than allocate new matrixes everytime
	d_cov_mat.diagonal().array() += e_CI;
	Eigen::LLT<Eigen::MatrixXd> llt_solver_start(d_cov_mat);
	VectorXd pi_start = llt_solver_start.solve(mu_mod);
	d_cov_mat.diagonal().array() -= e_CI;


	double var_start = 0.5 * pi_start.dot(cov_mat*pi_start);

	thread_local boost::random::xoshiro256pp gen(boost::random::random_device{}());    // thread safe
  boost::random::uniform_real_distribution<double> U(0, 1); // uni dist over (0,1)

	price      = 0.0;
	var_price  = 0.0;
  delta      = 0.0;
	var_delta  = 0.0;
	for(int j=0;j<M;j++)
	{
		double var_val = var_start;
		ArrayXd Y = Y0;
		ArrayXd G(dim);  
		for(int i=0;i<N;i++)
		{
			for(int l=0;l<dim;l++) G(l) = normcdfinv_as241(U(gen));
			NV_CIR(Y, C0, C1, C2, C3, G);

			e_CI = Y * vols_sq_penalised;
			d_cov_mat.diagonal().array() += e_CI;
			Eigen::LLT<Eigen::MatrixXd> llt_solver(d_cov_mat);
			VectorXd pi = llt_solver.solve(mu_mod);
			d_cov_mat.diagonal().array() -= e_CI;
			if(i==N-1) var_val += 0.5 * pi.dot(cov_mat*pi);
			else 			 var_val +=       pi.dot(cov_mat*pi);
		}
		var_val *= h;

    double X = X0*exp(r*TtoMat -0.5*var_val +sqrt(var_val)*normcdfinv_as241(U(gen)));

		double payoff;
    double delta_payoff = 0.0;

    if(X<min_gua)      payoff = min_gua;
    else if(X>max_gua) payoff = max_gua;
    else{
                       payoff = X;
                       delta_payoff = X/X0;
    }

		price      += payoff;
		var_price  += payoff*payoff;
    delta      += delta_payoff;
	  var_delta  += delta_payoff*delta_payoff;
	}

  delta /= double(M);
  var_delta = (var_delta - M * delta * delta) / double(M-1.0);

	price /= double(M);
	var_price = (var_price - M * price * price) / double(M-1.0);
	
  double discount = exp(-r*TtoMat);
  double survival_proba = GM_survival_proba(current_age, phi_GM, m_GM, b_GM, TtoMat);
  //std::cout << "\nsurvival_proba = " << survival_proba << std::endl;
  price      *= discount*survival_proba;
  delta      *= discount*survival_proba;
  var_price  *= discount*discount * survival_proba*survival_proba;
  var_delta  *= discount*discount * survival_proba*survival_proba;
}



void PE_min_max_GM_MMM(double CRRA_delta, double X0, double r, ArrayXd mu, ArrayXd vols, MatrixXd cov_mat, 
									double alpha_penalty, ArrayXd Y0, ArrayXd beta, ArrayXd kappa, ArrayXd s,
                  double current_age, double phi_GM, double m_GM, double b_GM,  
                  double min_gua, double max_gua, double TtoMat, int N, int M,
                  double &price, double &var_price, double &delta, double &var_delta)
{	
  // Pure Endowment option with min and max guaranteed threshold 
  // option evalueted under the minimal martingale probability
	VectorXd mu_mod = (mu-r).matrix();
	ArrayXd vols_sq_penalised = alpha_penalty*vols.square();
	MatrixXd d_cov_mat = CRRA_delta * cov_mat;
	double h = TtoMat/double(N);
	auto dim = (beta.size()); 
	ArrayXd C0(dim), C1(dim), C2(dim), C3(dim);
	prm_NV_CIR(h, beta, kappa, s, C0, C1, C2, C3); //setting parameters for NV scheme for CIR

	ArrayXd e_CI = Y0 * vols_sq_penalised;

	// for large n this is faster than allocate new matrixes everytime
	d_cov_mat.diagonal().array() += e_CI;
	Eigen::LLT<Eigen::MatrixXd> llt_solver_start(d_cov_mat);
	VectorXd pi_start = llt_solver_start.solve(mu_mod);
	d_cov_mat.diagonal().array() -= e_CI;


	double var_start = 0.5 * pi_start.dot(cov_mat*pi_start);

	thread_local boost::random::xoshiro256pp gen(boost::random::random_device{}());    // thread safe
  boost::random::uniform_real_distribution<double> U(0, 1); // uni dist over (0,1)

	price      = 0.0;
	var_price  = 0.0;
  delta      = 0.0;
	var_delta  = 0.0;
	for(int j=0;j<M;j++)
	{
		double var_val = var_start;
		ArrayXd Y = Y0;
		ArrayXd G(dim);  
		for(int i=0;i<N;i++)
		{
			for(int l=0;l<dim;l++) G(l) = normcdfinv_as241(U(gen));
			NV_CIR(Y, C0, C1, C2, C3, G);

			e_CI = Y * vols_sq_penalised;
			d_cov_mat.diagonal().array() += e_CI;
			Eigen::LLT<Eigen::MatrixXd> llt_solver(d_cov_mat);
			VectorXd pi = llt_solver.solve(mu_mod);
			d_cov_mat.diagonal().array() -= e_CI;
			if(i==N-1) var_val += 0.5 * pi.dot(cov_mat*pi);
			else 			 var_val +=       pi.dot(cov_mat*pi);
		}
		var_val *= h;

		double mean_val = -0.5*var_val + r*TtoMat;

    double temp  =  1. / sqrt(var_val);

		double b1_pi = (log(min_gua/X0)-mean_val)         * temp;
    double b2_pi = (log(max_gua/X0)-mean_val)         * temp;
    double a1_pi = (log(min_gua/X0)-mean_val-var_val) * temp;
    double a2_pi = (log(max_gua/X0)-mean_val-var_val) * temp;

		double payoff =   min_gua           * normcdf(b1_pi) 
              			+ max_gua - max_gua * normcdf(b2_pi) 
           					+ X0*exp(r*TtoMat)  *(normcdf(a2_pi)-normcdf(a1_pi));

    double delta_payoff  = (  max_gua          * normpdf(b2_pi) 
                            - min_gua          * normpdf(b1_pi) 
                            - X0*exp(r*TtoMat) *(normpdf(a2_pi)-normpdf(a1_pi)) ) * temp / X0
                            + exp(r*TtoMat)    *(normcdf(a2_pi)-normcdf(a1_pi));

		price      += payoff;
		var_price  += payoff*payoff;
    delta      += delta_payoff;
	  var_delta  += delta_payoff*delta_payoff;
	}

  delta /= double(M);
  var_delta = (var_delta - M * delta * delta) / double(M-1.0);

	price /= double(M);
	var_price = (var_price - M * price * price) / double(M-1.0);
	
  double discount = exp(-r*TtoMat);
  double survival_proba = GM_survival_proba(current_age, phi_GM, m_GM, b_GM, TtoMat);
  //std::cout << "\nsurvival_proba = " << survival_proba << std::endl;
  price      *= discount*survival_proba;
  delta      *= discount*survival_proba;
  var_price  *= discount*discount * survival_proba*survival_proba;
  var_delta  *= discount*discount * survival_proba*survival_proba;
}


std::pair<double,double> 
utility_func_PE(double X0, double r, double TtoMat, double var_val, double min_gua, double max_gua)
{ 
  double mean_val = -0.5*var_val + r*TtoMat;
  double temp  =  1. / sqrt(var_val);

  double fact = exp(r*TtoMat);

  double b1_pi = (log(min_gua/X0)-mean_val)         * temp;
  double b2_pi = (log(max_gua/X0)-mean_val)         * temp;
  double a1_pi = (log(min_gua/X0)-mean_val-var_val) * temp;
  double a2_pi = (log(max_gua/X0)-mean_val-var_val) * temp;

  std::pair<double,double> pricedelta;
  pricedelta.first  =  min_gua           * normcdf(b1_pi) 
                     + max_gua - max_gua * normcdf(b2_pi) 
                     + X0*fact           *(normcdf(a2_pi)-normcdf(a1_pi));

  /* pricedelta.second = (  max_gua         * normpdf(b2_pi) 
                       - min_gua         * normpdf(b1_pi) 
                       - X0*fact         *(normpdf(a2_pi)-normpdf(a1_pi)) ) * temp / X0
                       + fact            *(normcdf(a2_pi)-normcdf(a1_pi)); */
  
  pricedelta.second = fact *(normcdf(a2_pi)-normcdf(a1_pi));

  return pricedelta;
}



void TI_NOVR_min_max_GM_MMM(double CRRA_delta, double X0, double r, ArrayXd mu, ArrayXd vols, MatrixXd cov_mat, 
									double alpha_penalty, ArrayXd Y0, ArrayXd beta, ArrayXd kappa, ArrayXd s,
                  double current_age, double phi_GM, double m_GM, double b_GM,  
                  double min_gua_start, double max_gua_start, double min_rate, double max_rate,
                  double TtoMat, int N, int M,
                  double &price, double &var_price, double &delta, double &var_delta)
{	
  // Term Insurance option with min and max guaranteed threshold 
  // option evalueted under the minimal martingale probability
	VectorXd mu_mod = (mu-r).matrix();
	ArrayXd vols_sq_penalised = alpha_penalty*vols.square();
	MatrixXd d_cov_mat = CRRA_delta * cov_mat;
	double h = TtoMat/double(N);
	auto dim = (beta.size()); 
	ArrayXd C0(dim), C1(dim), C2(dim), C3(dim);
	prm_NV_CIR(h, beta, kappa, s, C0, C1, C2, C3); //setting parameters for NV scheme for CIR

	ArrayXd e_CI = Y0 * vols_sq_penalised;

	// for large n add and subctrat e_CI is faster than allocate new matrixes everytime
	d_cov_mat.diagonal().array() += e_CI;
	Eigen::LLT<Eigen::MatrixXd> llt_solver_start(d_cov_mat);
	VectorXd pi_start = llt_solver_start.solve(mu_mod);
	d_cov_mat.diagonal().array() -= e_CI;


	double var_start = h * 0.5 * pi_start.dot(cov_mat*pi_start); 

	thread_local boost::random::xoshiro256pp gen(boost::random::random_device{}());    // thread safe
  boost::random::uniform_real_distribution<double> U(0, 1); // uni dist over (0,1)

	price      = 0.0;
	var_price  = 0.0;
  delta      = 0.0;
	var_delta  = 0.0;
	for(int j=0; j<M; j++)
	{
    double price_tot = 0.0;
    double delta_tot = 0.0;

    double fact = GM_spXmi_discounted(current_age, phi_GM, m_GM, b_GM, 0.0, r);

    if     (X0<min_gua_start)
    {
      price_tot += 0.5*fact*min_gua_start;
    }
    else if(X0>max_gua_start)
    {
      price_tot += 0.5*fact*max_gua_start;
    }
    else{
      price_tot += 0.5*fact*X0;
      delta_tot += 0.5*fact;
    }
    
    double var_increment = var_start;
    double var_val = 0.0;
		ArrayXd Y = Y0;
		ArrayXd G(dim);
    double TimeWindow = 0.0;

		for(int i=0;i<N;i++)
		{ 
      TimeWindow +=h;
      var_val += var_increment;
			for(int l=0;l<dim;l++) G(l) = normcdfinv_as241(U(gen));
			NV_CIR(Y, C0, C1, C2, C3, G);

			e_CI = Y * vols_sq_penalised;
			d_cov_mat.diagonal().array() += e_CI;
			Eigen::LLT<Eigen::MatrixXd> llt_solver(d_cov_mat);
			VectorXd pi = llt_solver.solve(mu_mod);
			d_cov_mat.diagonal().array() -= e_CI;

      var_increment = h* 0.5 * pi.dot(cov_mat*pi);
			var_val += var_increment;

      double min_gua = min_gua_start*exp(min_rate * TimeWindow);
      double max_gua = max_gua_start*exp(max_rate * TimeWindow);

      double X = X0*exp(r*TimeWindow -0.5*var_val + sqrt(var_val)*normcdfinv_as241(U(gen)));

      double price_PE;
      double delta_PE = 0.0;
      if     (X<min_gua)
      {
        price_PE = min_gua;
      }
      else if(X>max_gua)
      {
        price_PE = max_gua;
      }
      else{
        price_PE = X;
        delta_PE = X/X0;
      }

      fact = GM_spXmi_discounted(current_age, phi_GM, m_GM, b_GM, TimeWindow, r);
      if(i<N-1)
      {
        price_tot +=     price_PE*fact;
        delta_tot +=     delta_PE*fact;
      }
      else
      {
        price_tot += 0.5*price_PE*fact;
        delta_tot += 0.5*delta_PE*fact;
      }
            
		}
    price_tot *= h;
    delta_tot *= h;

		price      += price_tot;
		var_price  += price_tot*price_tot;
    delta      += delta_tot;
	  var_delta  += delta_tot*delta_tot;
	}

  delta /= double(M);
  var_delta = (var_delta - M * delta * delta) / double(M-1.0);

	price /= double(M);
	var_price = (var_price - M * price * price) / double(M-1.0);
	
  //double discount = exp(-r*TtoMat); // discount is smoothed on the integral
  //price      *= discount;
  //delta      *= discount;
  //var_price  *= discount*discount;
  //var_delta  *= discount*discount;
  //std::cout << "price = "     << price     << std::endl;
  //std::cout << "var_price = " << var_price << std::endl;
  //std::cout << "delta = "     << delta     << std::endl;
  //std::cout << "var_delta = " << var_delta << std::endl;
}

 

void TI_min_max_GM_MMM(double CRRA_delta, double X0, double r, ArrayXd mu, ArrayXd vols, MatrixXd cov_mat, 
									double alpha_penalty, ArrayXd Y0, ArrayXd beta, ArrayXd kappa, ArrayXd s,
                  double current_age, double phi_GM, double m_GM, double b_GM,  
                  double min_gua_start, double max_gua_start, double min_rate, double max_rate,
                  double TtoMat, int N, int M,
                  double &price, double &var_price, double &delta, double &var_delta)
{	
  // Term Insurance option with min and max guaranteed threshold 
  // option evalueted under the minimal martingale probability
	VectorXd mu_mod = (mu-r).matrix();
	ArrayXd vols_sq_penalised = alpha_penalty*vols.square();
	MatrixXd d_cov_mat = CRRA_delta * cov_mat;
	double h = TtoMat/double(N);
	auto dim = (beta.size()); 
	ArrayXd C0(dim), C1(dim), C2(dim), C3(dim);
	prm_NV_CIR(h, beta, kappa, s, C0, C1, C2, C3); //setting parameters for NV scheme for CIR

	ArrayXd e_CI = Y0 * vols_sq_penalised;

	// for large n add and subctrat e_CI is faster than allocate new matrixes everytime
	d_cov_mat.diagonal().array() += e_CI;
	Eigen::LLT<Eigen::MatrixXd> llt_solver_start(d_cov_mat);
	VectorXd pi_start = llt_solver_start.solve(mu_mod);
	d_cov_mat.diagonal().array() -= e_CI;


	double var_start = h * 0.5 * pi_start.dot(cov_mat*pi_start); 

	thread_local boost::random::xoshiro256pp gen(boost::random::random_device{}());    // thread safe
  boost::random::uniform_real_distribution<double> U(0, 1); // uni dist over (0,1)

	price      = 0.0;
	var_price  = 0.0;
  delta      = 0.0;
	var_delta  = 0.0;

	for(int j=0; j<M; j++)
	{
    double price_tot = 0.0;
    double delta_tot = 0.0;

    double fact = GM_spXmi_discounted(current_age, phi_GM, m_GM, b_GM, 0.0, r);

    if     (X0<min_gua_start) price_tot += 0.5*fact*min_gua_start;
    else if(X0>max_gua_start) price_tot += 0.5*fact*max_gua_start;
    else{
                              price_tot += 0.5*fact*X0;
                              delta_tot += 0.5*fact;
    }
    
    double var_increment = var_start;
    double var_val = 0.0;
		ArrayXd Y = Y0;
		ArrayXd G(dim);
    double TimeWindow = 0.0;

		for(int i=0;i<N;i++)
		{ 
      TimeWindow += h;
      var_val += var_increment;
			for(int l=0;l<dim;l++) G(l) = normcdfinv_as241(U(gen));
			NV_CIR(Y, C0, C1, C2, C3, G);

			e_CI = Y * vols_sq_penalised;
			d_cov_mat.diagonal().array() += e_CI;
			Eigen::LLT<Eigen::MatrixXd> llt_solver(d_cov_mat);
			VectorXd pi = llt_solver.solve(mu_mod);
			d_cov_mat.diagonal().array() -= e_CI;

      var_increment = h* 0.5 * pi.dot(cov_mat*pi);
			var_val += var_increment;
      double mean_val = -0.5*var_val + r*TimeWindow;
      double temp  =  1. / sqrt(var_val);

      double min_gua = min_gua_start*exp(min_rate * TimeWindow);
      double max_gua = max_gua_start*exp(max_rate * TimeWindow);

      double b1_pi = (log(min_gua/X0)-mean_val)         * temp;
      double b2_pi = (log(max_gua/X0)-mean_val)         * temp;
      double a1_pi = (log(min_gua/X0)-mean_val-var_val) * temp;
      double a2_pi = (log(max_gua/X0)-mean_val-var_val) * temp;
      double price_PE =   min_gua               * normcdf(b1_pi) 
                        + max_gua - max_gua     * normcdf(b2_pi) 
                        + X0*exp(r*TimeWindow)  *(normcdf(a2_pi)-normcdf(a1_pi));

      //double delta_PE = (  max_gua              * normpdf(b2_pi) 
      //                   - min_gua              * normpdf(b1_pi) 
      //                   - X0*exp(r*TimeWindow) *(normpdf(a2_pi)-normpdf(a1_pi)) ) * temp / X0
      //                   + exp(r*TimeWindow)    *(normcdf(a2_pi)-normcdf(a1_pi));

      double delta_PE = exp(r*TimeWindow) * (normcdf(a2_pi)-normcdf(a1_pi));

      fact = GM_spXmi_discounted(current_age, phi_GM, m_GM, b_GM, TimeWindow, r);
      if(i<N-1)
      {
        price_tot +=     price_PE*fact;
        delta_tot +=     delta_PE*fact;
      }
      else
      {
        price_tot += 0.5*price_PE*fact;
        delta_tot += 0.5*delta_PE*fact;
      }
            
		}
    price_tot *= h;
    delta_tot *= h;

		price      += price_tot;
		var_price  += price_tot*price_tot;
    delta      += delta_tot;
	  var_delta  += delta_tot*delta_tot;
	}

  delta /= double(M);
  var_delta = (var_delta - M * delta * delta) / double(M-1.0);

	price /= double(M);
	var_price = (var_price - M * price * price) / double(M-1.0);
}


std::pair<double,double> 
utility_func_TI(double X0, double h, double r,  double TimeWindow, 
                double &var_increment, double & var_val,
                ArrayXd &Y,  ArrayXd &vols_sq_penalised, VectorXd &mu_mod,
                const MatrixXd &cov_mat, MatrixXd &d_cov_mat,
                double min_gua_start, double max_gua_start, 
                double min_rate, double max_rate)
{ 
  ArrayXd e_CI = Y * vols_sq_penalised;
	d_cov_mat.diagonal().array() += e_CI;
	Eigen::LLT<Eigen::MatrixXd> llt_solver(d_cov_mat);
	VectorXd pi = llt_solver.solve(mu_mod);
	d_cov_mat.diagonal().array() -= e_CI;

  var_increment = h * 0.5 * pi.dot(cov_mat*pi);
  var_val += var_increment;
  double mean_val = -0.5*var_val + r*TimeWindow;
  double temp  =  1. / sqrt(var_val);

  double min_gua = min_gua_start*exp(min_rate * TimeWindow);
  double max_gua = max_gua_start*exp(max_rate * TimeWindow);

  double temp_min = log(min_gua/X0);
  double temp_max = log(max_gua/X0);

  double b1_pi = (temp_min-mean_val)         * temp;
  double b2_pi = (temp_max-mean_val)         * temp;
  double a1_pi = (temp_min-mean_val-var_val) * temp;
  double a2_pi = (temp_max-mean_val-var_val) * temp;

  std::pair<double,double> pricedelta;
  pricedelta.second = exp(r*TimeWindow) * (normcdf(a2_pi)-normcdf(a1_pi));
  pricedelta.first  =  min_gua               * normcdf(b1_pi) 
                     + max_gua - max_gua     * normcdf(b2_pi) 
                     + X0*pricedelta.second;

  /* pricedelta.second = (  max_gua           * normpdf(b2_pi) 
                       - min_gua              * normpdf(b1_pi) 
                       - X0*exp(r*TimeWindow) *(normpdf(a2_pi)-normpdf(a1_pi)) ) * temp / X0
                       + exp(r*TimeWindow)    *(normcdf(a2_pi)-normcdf(a1_pi)); */

  return pricedelta;
}


// The four-factor penalty changes only the diagonal of an SPD matrix.
// For this fixed size, an unrolled LDL^T solve avoids a generic factorization
// and triangular-solve dispatch at every Monte Carlo time step. A small pivot
// sends poorly conditioned cases through Eigen's LLT below.
inline bool solve_fixed4_ldlt(const Eigen::Matrix4d &a,
                              const Eigen::Vector4d &b,
                              Eigen::Vector4d &x)
{
  const double a0 = a(0,0), a1 = a(1,1), a2 = a(2,2), a3 = a(3,3);
  const double largest_diagonal = std::max({a0, a1, a2, a3});
  if (!(largest_diagonal > 0.0) || !std::isfinite(largest_diagonal)) return false;
  const double pivot_floor = 1e-5 * largest_diagonal;

  const double d0 = a0;
  if (!(d0 > pivot_floor) || !std::isfinite(d0)) return false;
  const double l10 = a(1,0) / d0;
  const double l20 = a(2,0) / d0;
  const double l30 = a(3,0) / d0;

  const double d1 = a1 - l10*l10*d0;
  if (!(d1 > pivot_floor) || !std::isfinite(d1)) return false;
  const double l21 = (a(2,1) - l20*l10*d0) / d1;
  const double l31 = (a(3,1) - l30*l10*d0) / d1;

  const double d2 = a2 - l20*l20*d0 - l21*l21*d1;
  if (!(d2 > pivot_floor) || !std::isfinite(d2)) return false;
  const double l32 = (a(3,2) - l30*l20*d0 - l31*l21*d1) / d2;

  const double d3 = a3 - l30*l30*d0 - l31*l31*d1 - l32*l32*d2;
  if (!(d3 > pivot_floor) || !std::isfinite(d3)) return false;

  const double u0 = b[0];
  const double u1 = b[1] - l10*u0;
  const double u2 = b[2] - l20*u0 - l21*u1;
  const double u3 = b[3] - l30*u0 - l31*u1 - l32*u2;
  const double z0 = u0/d0, z1 = u1/d1, z2 = u2/d2, z3 = u3/d3;
  const double x3 = z3;
  const double x2 = z2 - l32*x3;
  const double x1 = z1 - l21*x2 - l31*x3;
  const double x0 = z0 - l10*x1 - l20*x2 - l30*x3;
  x << x0, x1, x2, x3;
  return x.allFinite();
}

template <typename CovarianceMatrix, typename PortfolioVector>
inline void solve_penalised_covariance(const CovarianceMatrix &a,
                                       const PortfolioVector &b,
                                       Eigen::LLT<CovarianceMatrix> &llt_solver,
                                       PortfolioVector &x)
{
  llt_solver.compute(a);
  x = llt_solver.solve(b);
}

inline void solve_penalised_covariance(const Eigen::Matrix4d &a,
                                       const Eigen::Vector4d &b,
                                       Eigen::LLT<Eigen::Matrix4d> &llt_solver,
                                       Eigen::Vector4d &x)
{
  if (solve_fixed4_ldlt(a, b, x)) return;
  llt_solver.compute(a);
  x = llt_solver.solve(b);
}

// Same conditional price/delta calculation as utility_func_TI, but with the
// time-dependent payoff quantities prepared by the caller. The Eigen objects
// at the end are reusable workspaces and must not be shared between threads.
template <typename CarbonVector, typename PortfolioVector, typename CovarianceMatrix>
inline std::pair<double,double>
utility_func_TI_precomputed(
                double X0, double half_time_step,
                double min_guarantee, double max_guarantee,
                double log_min_forward_moneyness, double log_max_forward_moneyness,
                double risk_free_growth,
                double &previous_half_variance, double &integrated_variance,
                const CarbonVector &Y, const CarbonVector &vols_sq_penalised,
                const PortfolioVector &mu_mod, const CovarianceMatrix &cov_mat,
                const CarbonVector &base_penalised_diagonal,
                CovarianceMatrix &penalised_cov_mat,
                Eigen::LLT<CovarianceMatrix> &llt_solver,
                PortfolioVector &pi, PortfolioVector &cov_times_pi)
{
  // Add the right half of the preceding interval.
  integrated_variance += previous_half_variance;

  // Only the diagonal changes with the carbon state. The off-diagonal entries
  // of penalised_cov_mat are initialized once by the caller and remain fixed.
  // Equation (2.2): Y * 1_{Y>0} is the componentwise positive part of Y.
  penalised_cov_mat.diagonal().array() =
      base_penalised_diagonal.array()
      + Y.array().max(0.0) * vols_sq_penalised.array();

  solve_penalised_covariance(penalised_cov_mat, mu_mod, llt_solver, pi);
  cov_times_pi.noalias() = cov_mat * pi;

  // Store the left half for the next interval and complete this trapezoid.
  previous_half_variance = half_time_step * pi.dot(cov_times_pi);
  integrated_variance += previous_half_variance;

  const double sqrt_variance = sqrt(integrated_variance);
  const double inverse_sqrt_variance = 1.0 / sqrt_variance;

  // log_#_forward_moneyness = log(#_guarantee / X0) - r*t.
  const double b1_pi = (log_min_forward_moneyness + 0.5*integrated_variance)
                     * inverse_sqrt_variance;
  const double b2_pi = (log_max_forward_moneyness + 0.5*integrated_variance)
                     * inverse_sqrt_variance;
  const double a1_pi = b1_pi - sqrt_variance;
  const double a2_pi = b2_pi - sqrt_variance;

  const double probability_between = normcdf(a2_pi) - normcdf(a1_pi);

  std::pair<double,double> pricedelta;
  pricedelta.second = risk_free_growth * probability_between;
  pricedelta.first  = min_guarantee * normcdf(b1_pi)
                     + max_guarantee * (1.0 - normcdf(b2_pi))
                     + X0 * pricedelta.second;

  return pricedelta;
}



//Scale with simulations for the first level M
void PE_min_max_GM_MMM_MLMC(double CRRA_delta, double X0, double r, ArrayXd mu, ArrayXd vols, MatrixXd cov_mat, 
									double alpha_penalty, ArrayXd Y0, ArrayXd beta, ArrayXd kappa, ArrayXd s,
                  double current_age, double phi_GM, double m_GM, double b_GM,  
                  double min_gua, double max_gua, 
                  double TtoMat, int start_pow, int n_lvl, int M,
                  std::vector<double> &price_lvl, std::vector<double> &price_var_lvl, 
                  std::vector<double> &delta_lvl, std::vector<double> &delta_var_lvl)
{	
  // Pure Endowment option with min and max guaranteed threshold 
  // option evalueted under the minimal martingale probability
  // MULTI LEVEL MONTE CARLO

  // for a second order scheme to 
	VectorXd mu_mod = (mu-r).matrix();
	ArrayXd vols_sq_penalised = alpha_penalty*vols.square();
	MatrixXd d_cov_mat = CRRA_delta * cov_mat;
  int N = 1.0;
  for(int l = 0; l < start_pow; ++l) N *= 2;

	double h = TtoMat/double(N);
  //std::cout << "\nAt level 0,  h = " << h << std::endl;
	auto dim = (beta.size()); 
	ArrayXd C0(dim), C1(dim), C2(dim), C3(dim);
	prm_NV_CIR(h,  beta, kappa, s, C0,   C1,   C2,   C3); //setting parameters for NV scheme for CIR

  

	ArrayXd e_CI = Y0 * vols_sq_penalised;

	// for large n add and subctrat e_CI is faster than allocate new matrixes everytime
	d_cov_mat.diagonal().array() += e_CI;
	Eigen::LLT<Eigen::MatrixXd> llt_solver_start(d_cov_mat);
	VectorXd pi_start = llt_solver_start.solve(mu_mod);
	d_cov_mat.diagonal().array() -= e_CI;


	double var_start   = 0.5 * pi_start.dot(cov_mat*pi_start); // no *h pure enodwment

	thread_local boost::random::xoshiro256pp gen(boost::random::random_device{}());    // thread safe
  boost::random::uniform_real_distribution<double> U(0, 1); // uni dist over (0,1)

	std::fill(price_lvl.begin(), price_lvl.end(), 0.0);
	std::fill(price_var_lvl.begin(), price_var_lvl.end(), 0.0);
  std::fill(delta_lvl.begin(), delta_lvl.end(), 0.0);
  std::fill(delta_var_lvl.begin(), delta_var_lvl.end(), 0.0);

  //double KL = 0.0;
  //for(int l=1; l<n_lvl; l++)  KL += pow(2, l/2.0);
  //KL = sqrt(3/2)*KL +1.0; //adjusting and adding first level contribution


	for(int j=0; j<M; j++)
	{
    double var_val = var_start;
		ArrayXd Y = Y0;
		ArrayXd G(dim);

    for(int i=0;i<N;i++)
		{
			for(int l=0;l<dim;l++) G(l) = normcdfinv_as241(U(gen));
			NV_CIR(Y, C0, C1, C2, C3, G);

      e_CI = Y * vols_sq_penalised;
			d_cov_mat.diagonal().array() += e_CI;
			Eigen::LLT<Eigen::MatrixXd> llt_solver(d_cov_mat);
			VectorXd pi = llt_solver.solve(mu_mod);
			d_cov_mat.diagonal().array() -= e_CI;

			if(i==N-1) var_val += 0.5 * pi.dot(  cov_mat*pi);
			else       var_val +=       pi.dot(  cov_mat*pi);
		}
		var_val   *= h;

    std::pair<double,double> pricedelta =
                        utility_func_PE(X0, r, TtoMat, var_val, min_gua, max_gua);

    price_lvl[0]      += pricedelta.first;
		price_var_lvl[0]  += pricedelta.first *pricedelta.first;
    delta_lvl[0]      += pricedelta.second;
	  delta_var_lvl[0]  += pricedelta.second*pricedelta.second;
	}

  price_lvl[0]     /= double(M);
  price_var_lvl[0] = (price_var_lvl[0] - M*price_lvl[0]*price_lvl[0]) / (M-1.0);
  delta_lvl[0]     /= double(M);
  delta_var_lvl[0] = (delta_var_lvl[0] - M*delta_lvl[0]*delta_lvl[0]) / (M-1.0);

  
  for(int lvl=1;lvl<n_lvl;lvl++)
  { 
    N = 1.0;
    for(int l = 0; l < start_pow+lvl-1; ++l) N *= 2;
    
    h         = TtoMat/double(N);
    double h2 = TtoMat/double(2*N);
    //std::cout << "\nAt level " << lvl << " h2 = " << h2 << std::endl;
    prm_NV_CIR(h,  beta, kappa, s, C0,   C1,   C2,   C3); //resetting parameters for CIR, new h

    ArrayXd C0_r(dim), C1_r(dim), C2_r(dim), C3_r(dim);
	  prm_NV_CIR(h2, beta, kappa, s, C0_r,   C1_r,   C2_r,   C3_r);



    int M_lvl = std::ceil(std::max(sqrt(2.0/3.0)*pow(2, -lvl/2.0)*M, 2.0));

 
    for(int j=0; j<M_lvl; j++)
    {
      double var_val   = var_start;
      double var_val_r = var_start;
      ArrayXd Y   = Y0;
      ArrayXd Y_r = Y0;
      ArrayXd G(dim);
      ArrayXd G_sum(dim); 
      for(int i=0;i<N;i++)
      {
        for(int l=0;l<dim;l++) G(l) = normcdfinv_as241(U(gen));
        NV_CIR(Y_r, C0_r, C1_r, C2_r, C3_r, G);
        G_sum = G;
        
        e_CI = Y_r * vols_sq_penalised;
        d_cov_mat.diagonal().array() += e_CI;
        Eigen::LLT<Eigen::MatrixXd> llt_solver_r1(d_cov_mat);
        VectorXd pi_r = llt_solver_r1.solve(mu_mod);
        d_cov_mat.diagonal().array() -= e_CI;

        var_val_r += pi_r.dot(cov_mat*pi_r);

        for(int l=0;l<dim;l++) G(l) = normcdfinv_as241(U(gen));
        NV_CIR(Y_r, C0_r, C1_r, C2_r, C3_r, G);
        G_sum += G;
        G_sum /= sqrt(2.0);

        NV_CIR(Y, C0, C1, C2, C3, G_sum);

        e_CI = Y_r * vols_sq_penalised;
        d_cov_mat.diagonal().array() += e_CI;
        Eigen::LLT<Eigen::MatrixXd> llt_solver_r2(d_cov_mat);
        pi_r = llt_solver_r2.solve(mu_mod);
        d_cov_mat.diagonal().array() -= e_CI;

        e_CI = Y * vols_sq_penalised;
        d_cov_mat.diagonal().array() += e_CI;
        Eigen::LLT<Eigen::MatrixXd> llt_solver(d_cov_mat);
        VectorXd pi = llt_solver.solve(mu_mod);
        d_cov_mat.diagonal().array() -= e_CI;

        if(i==N-1)
        {
          var_val_r += 0.5 * pi_r.dot(cov_mat*pi_r);
          var_val   += 0.5 * pi.dot(  cov_mat*pi);
        } 
        else
        {
          var_val_r +=       pi_r.dot(cov_mat*pi_r);
          var_val   +=       pi.dot(  cov_mat*pi);
        } 			 
      }
      var_val_r *= h2;
      var_val   *= h;

      std::pair<double,double> pricedelta_r =
                          utility_func_PE(X0, r, TtoMat, var_val_r, min_gua, max_gua);

      std::pair<double,double> pricedelta =
                          utility_func_PE(X0, r, TtoMat, var_val,   min_gua, max_gua);

      double temp       = pricedelta_r.first  - pricedelta.first;

      double delta_temp = pricedelta_r.second - pricedelta.second;


      price_lvl[lvl]      += temp;
      price_var_lvl[lvl]  += temp*temp;
      delta_lvl[lvl]      += delta_temp;
      delta_var_lvl[lvl]  += delta_temp*delta_temp;

    }

    price_lvl[lvl]     /= double(M_lvl);
		price_var_lvl[lvl] = (price_var_lvl[lvl] - M_lvl*price_lvl[lvl]*price_lvl[lvl]) / (M_lvl-1.0);
    delta_lvl[lvl]     /= double(M_lvl);
	  delta_var_lvl[lvl] = (delta_var_lvl[lvl] - M_lvl*delta_lvl[lvl]*delta_lvl[lvl]) / (M_lvl-1.0);

  }

  double disc_surv_proba = exp(-r*TtoMat)*GM_survival_proba(current_age, phi_GM, m_GM, b_GM, TtoMat);
  for(int l=0;l<n_lvl;l++)
  { 
    price_lvl[l]     *= disc_surv_proba;
    price_var_lvl[l] *= disc_surv_proba*disc_surv_proba;
    delta_lvl[l]     *= disc_surv_proba;
    delta_var_lvl[l] *= disc_surv_proba*disc_surv_proba;
  }
}



//Scale with simulations for the first level M
void TI_min_max_GM_MMM_MLMC(double CRRA_delta, double X0, double r, ArrayXd mu, ArrayXd vols, MatrixXd cov_mat, 
									double alpha_penalty, ArrayXd Y0, ArrayXd beta, ArrayXd kappa, ArrayXd s,
                  double current_age, double phi_GM, double m_GM, double b_GM,  
                  double min_gua_start, double max_gua_start, double min_rate, double max_rate,
                  double TtoMat, int start_pow, int n_lvl, int M,
                  std::vector<double> &price_lvl, std::vector<double> &price_var_lvl, 
                  std::vector<double> &delta_lvl, std::vector<double> &delta_var_lvl)
{	
  // Term Insurance option with min and max guaranteed threshold 
  // option evalueted under the minimal martingale probability
  // MULTI LEVEL MONTE CARLO

  // for a second order scheme to 
	VectorXd mu_mod = (mu-r).matrix();
	ArrayXd vols_sq_penalised = alpha_penalty*vols.square();
	MatrixXd d_cov_mat = CRRA_delta * cov_mat;
  int N = 1.0;
  for(int l = 0; l < start_pow; ++l) N *= 2;

	double h = TtoMat/double(N);
  //std::cout << "\nAt level 0,  h = " << h << std::endl;
	auto dim = (beta.size()); 
	ArrayXd C0(dim), C1(dim), C2(dim), C3(dim);
	prm_NV_CIR(h,  beta, kappa, s, C0,   C1,   C2,   C3); //setting parameters for NV scheme for CIR

  

	ArrayXd e_CI = Y0 * vols_sq_penalised;

	// for large n add and subctrat e_CI is faster than allocate new matrixes everytime
	d_cov_mat.diagonal().array() += e_CI;
	Eigen::LLT<Eigen::MatrixXd> llt_solver_start(d_cov_mat);
	VectorXd pi_start = llt_solver_start.solve(mu_mod);
	d_cov_mat.diagonal().array() -= e_CI;


	double var_start   = h  * 0.5 * pi_start.dot(cov_mat*pi_start); 

	thread_local boost::random::xoshiro256pp gen(boost::random::random_device{}());    // thread safe
  boost::random::uniform_real_distribution<double> U(0, 1); // uni dist over (0,1)

	std::fill(price_lvl.begin(), price_lvl.end(), 0.0);
	std::fill(price_var_lvl.begin(), price_var_lvl.end(), 0.0);
  std::fill(delta_lvl.begin(), delta_lvl.end(), 0.0);
  std::fill(delta_var_lvl.begin(), delta_var_lvl.end(), 0.0);

  //double KL = 0.0;
  //for(int l=1; l<n_lvl; l++)  KL += pow(2, l/2.0);
  //KL = sqrt(3/2)*KL +1.0; //adjusting and adding first level contribution


	for(int j=0; j<M; j++)
	{
    double price_tot = 0.0;
    double delta_tot = 0.0;

    double fact = GM_spXmi_discounted(current_age, phi_GM, m_GM, b_GM, 0.0, r);

    double payoff_0;
    double delta_0  = 0.0;
    if     (X0<min_gua_start) payoff_0 = min_gua_start;
    else if(X0>max_gua_start) payoff_0 = max_gua_start;
    else{
                              payoff_0 = X0;
                              delta_0  = 1.0;
    }
    price_tot += 0.5*fact*payoff_0;
    delta_tot += 0.5*fact*delta_0;

    
    double var_increment = var_start;
    double var_val = 0.0;
		ArrayXd Y = Y0;
		ArrayXd G(dim);
    double TimeWindow = 0.0;

		for(int i=0;i<N;i++)
		{ 
      TimeWindow += h;
      var_val += var_increment;
			for(int l=0;l<dim;l++) G(l) = normcdfinv_as241(U(gen));
      NV_CIR(Y, C0, C1, C2, C3, G);

      std::pair<double,double> pricedelta = 
            utility_func_TI(X0, h, r, TimeWindow, var_increment, var_val, 
                      Y, vols_sq_penalised, mu_mod, cov_mat, d_cov_mat,
                      min_gua_start, max_gua_start, min_rate, max_rate);

      fact = GM_spXmi_discounted(current_age, phi_GM, m_GM, b_GM, TimeWindow, r);

      if(i<N-1) {
        price_tot   +=     pricedelta.first    * fact;
        delta_tot   +=     pricedelta.second   * fact;
      }
      else {
        price_tot   += 0.5*pricedelta.first    * fact;
        delta_tot   += 0.5*pricedelta.second   * fact;
      }
            
		}
    price_tot   *= h;
    delta_tot   *= h;

    price_lvl[0]      += price_tot;
		price_var_lvl[0]  += price_tot*price_tot;
    delta_lvl[0]      += delta_tot;
	  delta_var_lvl[0]  += delta_tot*delta_tot;
	}

  price_lvl[0]     /= double(M);
  price_var_lvl[0] = (price_var_lvl[0] - M*price_lvl[0]*price_lvl[0]) / (M-1.0);
  delta_lvl[0]     /= double(M);
  delta_var_lvl[0] = (delta_var_lvl[0] - M*delta_lvl[0]*delta_lvl[0]) / (M-1.0);

  
  for(int lvl=1;lvl<n_lvl;lvl++)
  { 
    N = 1.0;
    for(int l = 0; l < start_pow+lvl-1; ++l) N *= 2;
    
    h         = TtoMat/double(N);
    double h2 = TtoMat/double(2*N);
    //std::cout << "\nAt level " << lvl << " h2 = " << h2 << std::endl;
    prm_NV_CIR(h,  beta, kappa, s, C0,   C1,   C2,   C3); //resetting parameters for CIR, new h

    ArrayXd C0_r(dim), C1_r(dim), C2_r(dim), C3_r(dim);
	  prm_NV_CIR(h2, beta, kappa, s, C0_r,   C1_r,   C2_r,   C3_r);

    var_start          = h  * 0.5 * pi_start.dot(cov_mat*pi_start);  //new h
    double var_start_r = h2 * 0.5 * pi_start.dot(cov_mat*pi_start);

    int M_lvl = std::ceil(std::max(sqrt(2.0/3.0)*pow(2, -3*lvl/2.0)*M, 2.0));

 
    for(int j=0; j<M_lvl; j++)
    {
      double price_tot = 0.0;
      double delta_tot = 0.0;

      double price_tot_r = 0.0;
      double delta_tot_r = 0.0;

      double fact = GM_spXmi_discounted(current_age, phi_GM, m_GM, b_GM, 0.0, r);

      double payoff_0;
      double delta_0  = 0.0;
      if     (X0<min_gua_start) payoff_0 = min_gua_start;
      else if(X0>max_gua_start) payoff_0 = max_gua_start;
      else{
                                payoff_0 = X0;
                                delta_0  = 1.0;
      }
      price_tot   += 0.5*fact*payoff_0;
      delta_tot   += 0.5*fact*delta_0;
      price_tot_r += 0.5*fact*payoff_0;
      delta_tot_r += 0.5*fact*delta_0;
      
      double var_increment   = var_start;
      double var_val = 0.0;
      double var_increment_r = var_start_r;
      double var_val_r = 0.0;
      ArrayXd Y   = Y0;
      ArrayXd Y_r = Y0;
      ArrayXd G(dim);
      ArrayXd G_sum(dim);
      double TimeWindow = 0.0;

      for(int i=0;i<N;i++)
      { 
        TimeWindow += h2;
        var_val_r += var_increment_r;
        for(int l=0;l<dim;l++) G(l) = normcdfinv_as241(U(gen));
        G_sum = G;
        NV_CIR(Y_r, C0_r, C1_r, C2_r, C3_r, G);

        std::pair<double,double> pricedelta_r = 
              utility_func_TI(X0, h2, r, TimeWindow, var_increment_r, var_val_r, 
                        Y_r, vols_sq_penalised, mu_mod, cov_mat, d_cov_mat,
                        min_gua_start, max_gua_start, min_rate, max_rate);

        fact = GM_spXmi_discounted(current_age, phi_GM, m_GM, b_GM, TimeWindow, r);
        price_tot_r +=     pricedelta_r.first *fact;
        delta_tot_r +=     pricedelta_r.second*fact;

        TimeWindow += h2;
        var_val_r += var_increment_r;
        for(int l=0;l<dim;l++) G(l) = normcdfinv_as241(U(gen));
        G_sum += G;
        G_sum /= sqrt(2.0); //normalizing to get std gaussian r.v.
        NV_CIR(Y_r, C0_r, C1_r, C2_r, C3_r, G);

        pricedelta_r = 
              utility_func_TI(X0, h2, r, TimeWindow, var_increment_r, var_val_r, 
                        Y_r, vols_sq_penalised, mu_mod, cov_mat, d_cov_mat,
                        min_gua_start, max_gua_start, min_rate, max_rate);

        fact = GM_spXmi_discounted(current_age, phi_GM, m_GM, b_GM, TimeWindow, r);

        // rough trajectory
        var_val += var_increment;
        NV_CIR(Y, C0, C1, C2, C3, G_sum);

        std::pair<double,double> pricedelta = 
                    utility_func_TI(X0, h, r, TimeWindow, var_increment, var_val, 
                        Y, vols_sq_penalised, mu_mod, cov_mat, d_cov_mat,
                        min_gua_start, max_gua_start, min_rate, max_rate);

        if(i<N-1) {
          price_tot_r +=     pricedelta_r.first  * fact;
          delta_tot_r +=     pricedelta_r.second * fact;
          price_tot   +=     pricedelta.first    * fact;
          delta_tot   +=     pricedelta.second   * fact;
        }
        else {
          price_tot_r += 0.5*pricedelta_r.first  * fact;
          delta_tot_r += 0.5*pricedelta_r.second * fact;
          price_tot   += 0.5*pricedelta.first    * fact;
          delta_tot   += 0.5*pricedelta.second   * fact;
        }
              
      }
      price_tot_r *= h2;
      delta_tot_r *= h2;
      price_tot   *= h;
      delta_tot   *= h;

      double temp         = price_tot_r - price_tot;

      double delta_temp   = delta_tot_r - delta_tot;

      price_lvl[lvl]      += temp;
      price_var_lvl[lvl]  += temp*temp;
      delta_lvl[lvl]      += delta_temp;
      delta_var_lvl[lvl]  += delta_temp*delta_temp;

    }

    price_lvl[lvl]     /= double(M_lvl);
		price_var_lvl[lvl] = (price_var_lvl[lvl] - M_lvl*price_lvl[lvl]*price_lvl[lvl]) / (M_lvl-1.0);
    delta_lvl[lvl]     /= double(M_lvl);
	  delta_var_lvl[lvl] = (delta_var_lvl[lvl] - M_lvl*delta_lvl[lvl]*delta_lvl[lvl]) / (M_lvl-1.0);

  }

}



//Scale with simulations for the first level M
void EI_min_max_GM_MMM_MLMC(double CRRA_delta, double X0, double r, ArrayXd mu, ArrayXd vols, MatrixXd cov_mat, 
									double alpha_penalty, ArrayXd Y0, ArrayXd beta, ArrayXd kappa, ArrayXd s,
                  double current_age, double phi_GM, double m_GM, double b_GM,  
                  double min_gua_start, double max_gua_start, double min_rate, double max_rate, double perc_prem_death, 
                  double TtoMat, int start_pow, int n_lvl, int M,
                  std::vector<double> &price_lvl, std::vector<double> &price_var_lvl, 
                  std::vector<double> &delta_lvl, std::vector<double> &delta_var_lvl)
{	
  // Term Insurance option with min and max guaranteed threshold 
  // option evalueted under the minimal martingale probability
  // MULTI LEVEL MONTE CARLO

  // for a second order scheme to 
	VectorXd mu_mod = (mu-r).matrix();
	ArrayXd vols_sq_penalised = alpha_penalty*vols.square();
	MatrixXd d_cov_mat = CRRA_delta * cov_mat;
  int N = 1.0;
  for(int l = 0; l < start_pow; ++l) N *= 2;

	double h = TtoMat/double(N);
  //std::cout << "\nAt level 0,  h = " << h << std::endl;
	auto dim = (beta.size()); 
	ArrayXd C0(dim), C1(dim), C2(dim), C3(dim);
	prm_NV_CIR(h,  beta, kappa, s, C0,   C1,   C2,   C3); //setting parameters for NV scheme for CIR

  

	ArrayXd e_CI = Y0 * vols_sq_penalised;

	// for large n add and subctrat e_CI is faster than allocate new matrixes everytime
	d_cov_mat.diagonal().array() += e_CI;
	Eigen::LLT<Eigen::MatrixXd> llt_solver_start(d_cov_mat);
	VectorXd pi_start = llt_solver_start.solve(mu_mod);
	d_cov_mat.diagonal().array() -= e_CI;


	double var_start   = h  * 0.5 * pi_start.dot(cov_mat*pi_start); 

	thread_local boost::random::xoshiro256pp gen(boost::random::random_device{}());    // thread safe
  boost::random::uniform_real_distribution<double> U(0, 1); // uni dist over (0,1)

	std::fill(price_lvl.begin(), price_lvl.end(), 0.0);
	std::fill(price_var_lvl.begin(), price_var_lvl.end(), 0.0);
  std::fill(delta_lvl.begin(), delta_lvl.end(), 0.0);
  std::fill(delta_var_lvl.begin(), delta_var_lvl.end(), 0.0);

  //double KL = 0.0;
  //for(int l=1; l<n_lvl; l++)  KL += pow(2, l/2.0);
  //KL = sqrt(3/2)*KL +1.0; //adjusting and adding first level contribution


	for(int j=0; j<M; j++)
	{
    double price_tot = 0.0;
    double delta_tot = 0.0;

    double fact = GM_spXmi_discounted(current_age, phi_GM, m_GM, b_GM, 0.0, r);

    double payoff_0;
    double delta_0  = 0.0;
    if     (X0<min_gua_start) payoff_0 = min_gua_start;
    else if(X0>max_gua_start) payoff_0 = max_gua_start;
    else{
                              payoff_0 = X0;
                               delta_0 = 1.0;
    }
    price_tot += 0.5*h*fact*perc_prem_death*payoff_0;
    delta_tot += 0.5*h*fact*perc_prem_death*delta_0;

    
    double var_increment = var_start;
    double var_val = 0.0;
		ArrayXd Y = Y0;
		ArrayXd G(dim);
    double TimeWindow = 0.0;

		for(int i=0;i<N;i++)
		{ 
      TimeWindow += h;
      var_val += var_increment;
			for(int l=0;l<dim;l++) G(l) = normcdfinv_as241(U(gen));
      NV_CIR(Y, C0, C1, C2, C3, G);

      std::pair<double,double> pricedelta = 
            utility_func_TI(X0, h, r, TimeWindow, var_increment, var_val, 
                      Y, vols_sq_penalised, mu_mod, cov_mat, d_cov_mat,
                      min_gua_start, max_gua_start, min_rate, max_rate);

      fact = GM_spXmi_discounted(current_age, phi_GM, m_GM, b_GM, TimeWindow, r);

      if(i<N-1) {
        price_tot   +=     h*perc_prem_death*pricedelta.first    * fact;
        delta_tot   +=     h*perc_prem_death*pricedelta.second   * fact;
      }
      else {
        double fact2 = GM_survival_proba_discounted(current_age, phi_GM, m_GM, b_GM, TimeWindow,r);
        price_tot += pricedelta.first *(0.5*h*perc_prem_death*fact + fact2);
        delta_tot += pricedelta.second*(0.5*h*perc_prem_death*fact + fact2);
      }
            
		}

    price_lvl[0]      += price_tot;
		price_var_lvl[0]  += price_tot*price_tot;
    delta_lvl[0]      += delta_tot;
	  delta_var_lvl[0]  += delta_tot*delta_tot;
	}

  price_lvl[0]     /= double(M);
  price_var_lvl[0] = (price_var_lvl[0] - M*price_lvl[0]*price_lvl[0]) / (M-1.0);
  delta_lvl[0]     /= double(M);
  delta_var_lvl[0] = (delta_var_lvl[0] - M*delta_lvl[0]*delta_lvl[0]) / (M-1.0);

  
  for(int lvl=1;lvl<n_lvl;lvl++)
  { 
    N = 1.0;
    for(int l = 0; l < start_pow+lvl-1; ++l) N *= 2;
    
    h         = TtoMat/double(N);
    double h2 = TtoMat/double(2*N);
    //std::cout << "\nAt level " << lvl << " h2 = " << h2 << std::endl;
    prm_NV_CIR(h,  beta, kappa, s, C0,   C1,   C2,   C3); //resetting parameters for CIR, new h

    ArrayXd C0_r(dim), C1_r(dim), C2_r(dim), C3_r(dim);
	  prm_NV_CIR(h2, beta, kappa, s, C0_r,   C1_r,   C2_r,   C3_r);

    var_start          = h  * 0.5 * pi_start.dot(cov_mat*pi_start);  //new h
    double var_start_r = h2 * 0.5 * pi_start.dot(cov_mat*pi_start);

    int M_lvl = std::ceil(std::max(sqrt(2.0/3.0)*pow(2, -3*lvl/2.0)*M, 2.0));

 
    for(int j=0; j<M_lvl; j++)
    {
      double price_tot = 0.0;
      double delta_tot = 0.0;

      double price_tot_r = 0.0;
      double delta_tot_r = 0.0;

      double fact = GM_spXmi_discounted(current_age, phi_GM, m_GM, b_GM, 0.0, r);

      double payoff_0;
      double delta_0  = 0.0;
      if     (X0<min_gua_start) payoff_0 = min_gua_start;
      else if(X0>max_gua_start) payoff_0 = max_gua_start;
      else{
                                payoff_0 = X0;
                                delta_0  = 1.0;
      }
      price_tot   += 0.5*h *fact*perc_prem_death*payoff_0;
      delta_tot   += 0.5*h *fact*perc_prem_death*delta_0;
      price_tot_r += 0.5*h2*fact*perc_prem_death*payoff_0;
      delta_tot_r += 0.5*h2*fact*perc_prem_death*delta_0;
      
      double var_increment   = var_start;
      double var_val = 0.0;
      double var_increment_r = var_start_r;
      double var_val_r = 0.0;
      ArrayXd Y   = Y0;
      ArrayXd Y_r = Y0;
      ArrayXd G(dim);
      ArrayXd G_sum(dim);
      double TimeWindow = 0.0;

      for(int i=0;i<N;i++)
      { 
        TimeWindow += h2;
        var_val_r += var_increment_r;
        for(int l=0;l<dim;l++) G(l) = normcdfinv_as241(U(gen));
        G_sum = G;
        NV_CIR(Y_r, C0_r, C1_r, C2_r, C3_r, G);

        std::pair<double,double> pricedelta_r = 
              utility_func_TI(X0, h2, r, TimeWindow, var_increment_r, var_val_r, 
                        Y_r, vols_sq_penalised, mu_mod, cov_mat, d_cov_mat,
                        min_gua_start, max_gua_start, min_rate, max_rate);

        fact = GM_spXmi_discounted(current_age, phi_GM, m_GM, b_GM, TimeWindow, r);
        price_tot_r += h2*perc_prem_death*pricedelta_r.first *fact;
        delta_tot_r += h2*perc_prem_death*pricedelta_r.second*fact;

        TimeWindow += h2;
        var_val_r += var_increment_r;
        for(int l=0;l<dim;l++) G(l) = normcdfinv_as241(U(gen));
        G_sum += G;
        G_sum /= sqrt(2.0); //normalizing to get std gaussian r.v.
        NV_CIR(Y_r, C0_r, C1_r, C2_r, C3_r, G);

        pricedelta_r = 
              utility_func_TI(X0, h2, r, TimeWindow, var_increment_r, var_val_r, 
                        Y_r, vols_sq_penalised, mu_mod, cov_mat, d_cov_mat,
                        min_gua_start, max_gua_start, min_rate, max_rate);

        fact = GM_spXmi_discounted(current_age, phi_GM, m_GM, b_GM, TimeWindow, r);

        // rough trajectory
        var_val += var_increment;
        NV_CIR(Y, C0, C1, C2, C3, G_sum);

        std::pair<double,double> pricedelta = 
                    utility_func_TI(X0, h, r, TimeWindow, var_increment, var_val, 
                        Y, vols_sq_penalised, mu_mod, cov_mat, d_cov_mat,
                        min_gua_start, max_gua_start, min_rate, max_rate);

        if(i<N-1) {
          price_tot_r +=     h2*perc_prem_death*pricedelta_r.first  * fact;
          delta_tot_r +=     h2*perc_prem_death*pricedelta_r.second * fact;
          price_tot   +=     h *perc_prem_death*pricedelta.first    * fact;
          delta_tot   +=     h *perc_prem_death*pricedelta.second   * fact;
        }
        else {
          double fact2 = GM_survival_proba_discounted(current_age, phi_GM, m_GM, b_GM, TimeWindow,r);
          price_tot_r += pricedelta_r.first *(0.5*h2*perc_prem_death*fact + fact2);
          delta_tot_r += pricedelta_r.second*(0.5*h2*perc_prem_death*fact + fact2);
          price_tot   += pricedelta.first   *(0.5*h *perc_prem_death*fact + fact2);
          delta_tot   += pricedelta.second  *(0.5*h *perc_prem_death*fact + fact2);
        }
              
      }

      double temp         = price_tot_r - price_tot;

      double delta_temp   = delta_tot_r - delta_tot;

      price_lvl[lvl]      += temp;
      price_var_lvl[lvl]  += temp*temp;
      delta_lvl[lvl]      += delta_temp;
      delta_var_lvl[lvl]  += delta_temp*delta_temp;

    }

    price_lvl[lvl]     /= double(M_lvl);
		price_var_lvl[lvl] = (price_var_lvl[lvl] - M_lvl*price_lvl[lvl]*price_lvl[lvl]) / (M_lvl-1.0);
    delta_lvl[lvl]     /= double(M_lvl);
	  delta_var_lvl[lvl] = (delta_var_lvl[lvl] - M_lvl*delta_lvl[lvl]*delta_lvl[lvl]) / (M_lvl-1.0);

  }

}




void EI_NOVR_min_max_GM_MMM(double CRRA_delta, double X0, double r, ArrayXd mu, ArrayXd vols, MatrixXd cov_mat, 
									double alpha_penalty, ArrayXd Y0, ArrayXd beta, ArrayXd kappa, ArrayXd s,
                  double current_age, double phi_GM, double m_GM, double b_GM,  
                  double min_gua_start, double max_gua_start, double min_rate, double max_rate, double perc_prem_death,
                  double TtoMat, int N, int M,
                  double &price, double &var_price, double &delta, double &var_delta)
{	
  // Term Insurance option with min and max guaranteed threshold 
  // option evalueted under the minimal martingale probability
	VectorXd mu_mod = (mu-r).matrix();
	ArrayXd vols_sq_penalised = alpha_penalty*vols.square();
	MatrixXd d_cov_mat = CRRA_delta * cov_mat;
	double h = TtoMat/double(N);
	auto dim = (beta.size()); 
	ArrayXd C0(dim), C1(dim), C2(dim), C3(dim);
	prm_NV_CIR(h, beta, kappa, s, C0, C1, C2, C3); //setting parameters for NV scheme for CIR

	ArrayXd e_CI = Y0 * vols_sq_penalised;

	// for large n add and subctrat e_CI is faster than allocate new matrixes everytime
	d_cov_mat.diagonal().array() += e_CI;
	Eigen::LLT<Eigen::MatrixXd> llt_solver_start(d_cov_mat);
	VectorXd pi_start = llt_solver_start.solve(mu_mod);
	d_cov_mat.diagonal().array() -= e_CI;


	double var_start = h * 0.5 * pi_start.dot(cov_mat*pi_start); 

	thread_local boost::random::xoshiro256pp gen(boost::random::random_device{}());    // thread safe
  boost::random::uniform_real_distribution<double> U(0, 1); // uni dist over (0,1)

	price      = 0.0;
	var_price  = 0.0;
  delta      = 0.0;
	var_delta  = 0.0;

	for(int j=0; j<M; j++)
	{
    double price_tot = 0.0;
    double delta_tot = 0.0;

    double fact = GM_spXmi_discounted(current_age, phi_GM, m_GM, b_GM, 0.0, r);

    if     (X0<min_gua_start)
    {
      price_tot += 0.5*h*perc_prem_death*fact*min_gua_start;
    }
    else if(X0>max_gua_start)
    {
      price_tot += 0.5*h*perc_prem_death*fact*max_gua_start;
    }
    else{
      price_tot += 0.5*h*perc_prem_death*fact*X0;
      delta_tot += 0.5*h*perc_prem_death*fact;
    }
    
    double var_increment = var_start;
    double var_val = 0.0;
		ArrayXd Y = Y0;
		ArrayXd G(dim);
    double TimeWindow = 0.0;

		for(int i=0;i<N;i++)
		{ 
      TimeWindow +=h;
      var_val += var_increment;
			for(int l=0;l<dim;l++) G(l) = normcdfinv_as241(U(gen));
			NV_CIR(Y, C0, C1, C2, C3, G);

			e_CI = Y * vols_sq_penalised;
			d_cov_mat.diagonal().array() += e_CI;
			Eigen::LLT<Eigen::MatrixXd> llt_solver(d_cov_mat);
			VectorXd pi = llt_solver.solve(mu_mod);
			d_cov_mat.diagonal().array() -= e_CI;

      var_increment = h* 0.5 * pi.dot(cov_mat*pi);
			var_val += var_increment;

      double min_gua = min_gua_start*exp(min_rate * TimeWindow);
      double max_gua = max_gua_start*exp(max_rate * TimeWindow);

      double X = X0*exp(r*TimeWindow -0.5*var_val + sqrt(var_val)*normcdfinv_as241(U(gen)));

      double price_PE;
      double delta_PE = 0.0;
      if     (X<min_gua)
      {
        price_PE = min_gua;
      }
      else if(X>max_gua)
      {
        price_PE = max_gua;
      }
      else{
        price_PE = X;
        delta_PE = X/X0;
      }

      fact = GM_spXmi_discounted(current_age, phi_GM, m_GM, b_GM, TimeWindow, r);
      if(i<N-1)
      {
        price_tot +=     h*perc_prem_death*price_PE*fact;
        delta_tot +=     h*perc_prem_death*delta_PE*fact;
      }
      else
      {
        price_tot += 0.5*h*perc_prem_death*price_PE*fact
                    + price_PE*GM_survival_proba_discounted(current_age, phi_GM, m_GM, b_GM, TimeWindow,r);
        delta_tot += 0.5*h*perc_prem_death*delta_PE*fact
                    + delta_PE*GM_survival_proba_discounted(current_age, phi_GM, m_GM, b_GM, TimeWindow,r);;
      }
            
		}

		price      += price_tot;
		var_price  += price_tot*price_tot;
    delta      += delta_tot;
	  var_delta  += delta_tot*delta_tot;
	}

  delta /= double(M);
  var_delta = (var_delta - M * delta * delta) / double(M-1.0);

	price /= double(M);
	var_price = (var_price - M * price * price) / double(M-1.0);
	
  //double discount = exp(-r*TtoMat); // discount is smoothed on the integral
  //price      *= discount;
  //delta      *= discount;
  //var_price  *= discount*discount;
  //var_delta  *= discount*discount;
  //std::cout << "price = "     << price     << std::endl;
  //std::cout << "var_price = " << var_price << std::endl;
  //std::cout << "delta = "     << delta     << std::endl;
  //std::cout << "var_delta = " << var_delta << std::endl;
}




void EI_min_max_GM_MMM(double CRRA_delta, double X0, double r, ArrayXd mu, ArrayXd vols, MatrixXd cov_mat, 
									double alpha_penalty, ArrayXd Y0, ArrayXd beta, ArrayXd kappa, ArrayXd s,
                  double current_age, double phi_GM, double m_GM, double b_GM,  
                  double min_gua_start, double max_gua_start, double min_rate, double max_rate, double perc_prem_death, 
                  double TtoMat, int N, int M,
                  double &price, double &var_price, double &delta, double &var_delta)
{	
  // Endowment Insurance option with min and max guaranteed threshold 
  // option evalueted under the minimal martingale probability
  // thanks to perc_prem_death now\in(0,1] now dying before maturity has different payoff
  // here min_gua and max_gua are revaluated at rate min_gua and max_gua while time passing
	VectorXd mu_mod = (mu-r).matrix();
	ArrayXd vols_sq_penalised = alpha_penalty*vols.square();
	MatrixXd d_cov_mat = CRRA_delta * cov_mat;
	double h = TtoMat/double(N);
	auto dim = (beta.size()); 
	ArrayXd C0(dim), C1(dim), C2(dim), C3(dim);
	prm_NV_CIR(h, beta, kappa, s, C0, C1, C2, C3); //setting parameters for NV scheme for CIR

	ArrayXd e_CI = Y0 * vols_sq_penalised;

	// for large n add and subctrat e_CI is faster than allocate new matrixes everytime
	d_cov_mat.diagonal().array() += e_CI;
	Eigen::LLT<Eigen::MatrixXd> llt_solver_start(d_cov_mat);
	VectorXd pi_start = llt_solver_start.solve(mu_mod);
	d_cov_mat.diagonal().array() -= e_CI;


	double var_start = h * 0.5 * pi_start.dot(cov_mat*pi_start); 

	thread_local boost::random::xoshiro256pp gen(boost::random::random_device{}());    // thread safe
  boost::random::uniform_real_distribution<double> U(0, 1); // uni dist over (0,1)

	price      = 0.0;
	var_price  = 0.0;
  delta      = 0.0;
	var_delta  = 0.0;

	for(int j=0; j<M; j++)
	{
    double price_tot = 0.0;
    double delta_tot = 0.0;

    double fact = GM_spXmi_discounted(current_age, phi_GM, m_GM, b_GM, 0.0, r);

    if(X0<min_gua_start)      price_tot += 0.5*h*perc_prem_death*fact*min_gua_start;

    else if(X0>max_gua_start) price_tot += 0.5*h*perc_prem_death*fact*max_gua_start;
    else{
                              price_tot += 0.5*h*perc_prem_death*fact*X0;
                              delta_tot += 0.5*h*perc_prem_death*fact;
    }

    double var_increment = var_start;
    double var_val = 0.0;
		ArrayXd Y = Y0;
		ArrayXd G(dim);
    double TimeWindow = 0.0;

		for(int i=0;i<N;i++)
		{ 
      TimeWindow += h;
      var_val += var_increment;
			for(int l=0;l<dim;l++) G(l) = normcdfinv_as241(U(gen));
			NV_CIR(Y, C0, C1, C2, C3, G);

			std::pair<double,double> pricedelta =
                utility_func_TI(X0, h, r, TimeWindow, var_increment, var_val, Y, 
                                vols_sq_penalised, mu_mod, cov_mat, d_cov_mat,
                              min_gua_start, max_gua_start, min_rate, max_rate);


      fact = GM_spXmi_discounted(current_age, phi_GM, m_GM, b_GM, TimeWindow, r);
      if(i<N-1)
      {
        price_tot +=     h*perc_prem_death*pricedelta.first *fact;
        delta_tot +=     h*perc_prem_death*pricedelta.second*fact;
      }
      else // at Maturity T
      {
        double fact2 = GM_survival_proba_discounted(current_age, phi_GM, m_GM, b_GM, TimeWindow,r);
        price_tot += pricedelta.first *(0.5*h*perc_prem_death*fact + fact2);
        delta_tot += pricedelta.second*(0.5*h*perc_prem_death*fact + fact2);
      }
            
		}

		price      += price_tot;
		var_price  += price_tot*price_tot;
    delta      += delta_tot;
	  var_delta  += delta_tot*delta_tot;
	}

  delta /= double(M);
  var_delta = (var_delta - M * delta * delta) / double(M-1.0);

	price /= double(M);
	var_price = (var_price - M * price * price) / double(M-1.0);
	
}




void PETIEI_min_max_GM_MMM(double CRRA_delta, double X0, double r, ArrayXd mu, ArrayXd vols, MatrixXd cov_mat, 
									double alpha_penalty, ArrayXd Y0, ArrayXd beta, ArrayXd kappa, ArrayXd s,
                  double current_age, double phi_GM, double m_GM, double b_GM,  
                  double min_gua_start, double max_gua_start, double min_rate, double max_rate, double perc_prem_death,
                  double TtoMat, int N, int M,
                  double &price_PE, double &var_price_PE, double &delta_PE, double &var_delta_PE,
                  double &price_TI, double &var_price_TI, double &delta_TI, double &var_delta_TI,
                  double &price_EI, double &var_price_EI, double &delta_EI, double &var_delta_EI)
{	
  // Term Insurance option with min and max guaranteed threshold 
  // option evalueted under the minimal martingale probability
	VectorXd mu_mod = (mu-r).matrix();
	ArrayXd vols_sq_penalised = alpha_penalty*vols.square();
	MatrixXd d_cov_mat = CRRA_delta * cov_mat;
	double h = TtoMat/double(N);
	auto dim = (beta.size()); 
	ArrayXd C0(dim), C1(dim), C2(dim), C3(dim);
	prm_NV_CIR(h, beta, kappa, s, C0, C1, C2, C3); //setting parameters for NV scheme for CIR

	ArrayXd e_CI = Y0 * vols_sq_penalised;

	// for large n add and subctrat e_CI is faster than allocate new matrixes everytime
	d_cov_mat.diagonal().array() += e_CI;
	Eigen::LLT<Eigen::MatrixXd> llt_solver_start(d_cov_mat);
	VectorXd pi_start = llt_solver_start.solve(mu_mod);
	d_cov_mat.diagonal().array() -= e_CI;


	double var_start = h * 0.5 * pi_start.dot(cov_mat*pi_start); 

	thread_local boost::random::xoshiro256pp gen(boost::random::random_device{}());    // thread safe
  boost::random::uniform_real_distribution<double> U(0, 1); // uni dist over (0,1)

  price_PE      = 0.0; var_price_PE  = 0.0;
  delta_PE      = 0.0; var_delta_PE  = 0.0;

	price_TI      = 0.0; var_price_TI  = 0.0;
  delta_TI      = 0.0; var_delta_TI  = 0.0;

  price_EI      = 0.0; var_price_EI  = 0.0; 
  delta_EI      = 0.0; var_delta_EI  = 0.0;

  double disc_PE = GM_survival_proba_discounted(current_age, phi_GM, m_GM, b_GM, TtoMat,r);
	for(int j=0; j<M; j++)
	{
    double price_TI_tot = 0.0;
    double delta_TI_tot = 0.0;

    double fact = GM_spXmi_discounted(current_age, phi_GM, m_GM, b_GM, 0.0, r);

    if     (X0<min_gua_start) price_TI_tot += 0.5*h*fact*min_gua_start;
    else if(X0>max_gua_start) price_TI_tot += 0.5*h*fact*max_gua_start;
    else{                     
                              price_TI_tot += 0.5*h*fact*X0;
                              delta_TI_tot += 0.5*h*fact;
    }
    
    double var_increment = var_start;
    double var_val = 0.0;
		ArrayXd Y = Y0;
		ArrayXd G(dim);
    double TimeWindow = 0.0;

		for(int i=0;i<N;i++)
		{ 
      TimeWindow +=h;
      var_val += var_increment;
			for(int l=0;l<dim;l++) G(l) = normcdfinv_as241(U(gen));
			NV_CIR(Y, C0, C1, C2, C3, G);

      std::pair<double,double>  pricedelta =
            utility_func_TI(X0, h, r, TimeWindow, var_increment, var_val,  Y,  vols_sq_penalised, 
                    mu_mod, cov_mat, d_cov_mat, min_gua_start, max_gua_start, min_rate, max_rate);


      fact = GM_spXmi_discounted(current_age, phi_GM, m_GM, b_GM, TimeWindow, r);
      if(i<N-1)
      {  
        price_TI_tot +=     h*pricedelta.first *fact;
        delta_TI_tot +=     h*pricedelta.second*fact;
      }
      else
      { 
        price_TI_tot += 0.5*h*pricedelta.first*fact;
        delta_TI_tot += 0.5*h*pricedelta.second*fact;
        
        price_TI      += price_TI_tot;
        var_price_TI  += price_TI_tot*price_TI_tot;
        delta_TI      += delta_TI_tot;
        var_delta_TI  += delta_TI_tot*delta_TI_tot;
        
        double temp_PE       = pricedelta.first  * disc_PE;                  
        double delta_temp_PE = pricedelta.second * disc_PE;

        price_PE     += temp_PE;
        var_price_PE += temp_PE*temp_PE;
        delta_PE     += delta_temp_PE;
        var_delta_PE += delta_temp_PE*delta_temp_PE;

        double temp_EI       = perc_prem_death*price_TI_tot + temp_PE;
        double delta_temp_EI = perc_prem_death*delta_TI_tot + delta_temp_PE;

        price_EI     += temp_EI;
        var_price_EI += temp_EI*temp_EI;
        delta_EI     += delta_temp_EI;
        var_delta_EI += delta_temp_EI*delta_temp_EI;
      }
            
		}
    
	}

  delta_PE /= double(M);
  var_delta_PE = (var_delta_PE - M * delta_PE * delta_PE) / double(M-1.0);

	price_PE /= double(M);
	var_price_PE = (var_price_PE - M * price_PE * price_PE) / double(M-1.0);


  delta_TI /= double(M);
  var_delta_TI = (var_delta_TI - M * delta_TI * delta_TI) / double(M-1.0);

	price_TI /= double(M);
	var_price_TI = (var_price_TI - M * price_TI * price_TI) / double(M-1.0);


  delta_EI /= double(M);
  var_delta_EI = (var_delta_EI - M * delta_EI * delta_EI) / double(M-1.0);

	price_EI /= double(M);
	var_price_EI = (var_price_EI - M * price_EI * price_EI) / double(M-1.0);
	
}



void PETIEI_min_max_GM_MMM_MLMC(double CRRA_delta, double X0, double r, ArrayXd mu, ArrayXd vols, MatrixXd cov_mat, 
									double alpha_penalty, ArrayXd Y0, ArrayXd beta, ArrayXd kappa, ArrayXd s,
                  double current_age, double phi_GM, double m_GM, double b_GM,  
                  double min_gua_start, double max_gua_start, double min_rate, double max_rate, double perc_prem_death,
                  double TtoMat, int N_lvl, int M,
                  double &price_PE, double &CI_price_PE, double &delta_PE, double &CI_delta_PE,
                  double &price_TI, double &CI_price_TI, double &delta_TI, double &CI_delta_TI,
                  double &price_EI, double &CI_price_EI, double &delta_EI, double &CI_delta_EI)
{	

  std::vector<std::vector<double>> prices_lvl(    3, std::vector<double>(N_lvl, 0.0));
  std::vector<std::vector<double>> deltas_lvl(    3, std::vector<double>(N_lvl, 0.0));
  std::vector<std::vector<double>> prices_var_lvl(3, std::vector<double>(N_lvl, 0.0));
  std::vector<std::vector<double>> deltas_var_lvl(3, std::vector<double>(N_lvl, 0.0));

  // Term Insurance option with min and max guaranteed threshold 
  // option evalueted under the minimal martingale probability
	VectorXd mu_mod = (mu-r).matrix();
	ArrayXd vols_sq_penalised = alpha_penalty*vols.square();
	MatrixXd d_cov_mat = CRRA_delta * cov_mat;
  int N = 1.0; 
	double h = TtoMat/double(N);
	auto dim = (beta.size()); 
	ArrayXd C0(dim), C1(dim), C2(dim), C3(dim);
	prm_NV_CIR(h, beta, kappa, s, C0, C1, C2, C3); //setting parameters for NV scheme for CIR

	ArrayXd e_CI = Y0 * vols_sq_penalised; // for large n add and subtract e_CI is faster than allocate new matrix everytime
	d_cov_mat.diagonal().array() += e_CI;
	Eigen::LLT<Eigen::MatrixXd> llt_solver_start(d_cov_mat);
	VectorXd pi_start = llt_solver_start.solve(mu_mod);
	d_cov_mat.diagonal().array() -= e_CI;

	double var_start   = h  * 0.5 * pi_start.dot(cov_mat*pi_start);


	thread_local boost::random::xoshiro256pp gen(boost::random::random_device{}());    // thread safe
  boost::random::uniform_real_distribution<double> U(0, 1); // uni dist over (0,1)

  double disc_PE = GM_survival_proba_discounted(current_age, phi_GM, m_GM, b_GM, TtoMat,r);

  price_PE = 0.0; CI_price_PE  = 0.0;
  delta_PE = 0.0; CI_delta_PE  = 0.0;

	price_TI = 0.0; CI_price_TI  = 0.0;
  delta_TI = 0.0; CI_delta_TI  = 0.0;

  price_EI = 0.0; CI_price_EI  = 0.0; 
  delta_EI = 0.0; CI_delta_EI  = 0.0;

	for(int j=0; j<M; j++)
	{
    double price_TI_tot = 0.0;
    double delta_TI_tot = 0.0;

    double fact = GM_spXmi_discounted(current_age, phi_GM, m_GM, b_GM, 0.0, r);

    if     (X0<min_gua_start) price_TI_tot += 0.5*h*fact*min_gua_start;
    else if(X0>max_gua_start) price_TI_tot += 0.5*h*fact*max_gua_start;
    else{                     
                              price_TI_tot += 0.5*h*fact*X0;
                              delta_TI_tot += 0.5*h*fact;
    }
    
    double var_increment = var_start;
    double var_val = 0.0;
		ArrayXd Y = Y0;
		ArrayXd G(dim);
    double TimeWindow = 0.0;

		for(int i=0;i<N;i++)
		{ 
      TimeWindow +=h;
      var_val += var_increment;
			for(int l=0;l<dim;l++) G(l) = normcdfinv_as241(U(gen));
			NV_CIR(Y, C0, C1, C2, C3, G);

      std::pair<double,double>  pricedelta =
            utility_func_TI(X0, h, r, TimeWindow, var_increment, var_val,  Y,  vols_sq_penalised, 
                    mu_mod, cov_mat, d_cov_mat, min_gua_start, max_gua_start, min_rate, max_rate);


      fact = GM_spXmi_discounted(current_age, phi_GM, m_GM, b_GM, TimeWindow, r);
      if(i<N-1)
      {  
        price_TI_tot +=     h*pricedelta.first *fact;
        delta_TI_tot +=     h*pricedelta.second*fact;
      }
      else
      { 
        price_TI_tot += 0.5*h*pricedelta.first *fact;
        delta_TI_tot += 0.5*h*pricedelta.second*fact;

        prices_lvl[1][0]     += price_TI_tot;
        prices_var_lvl[1][0] += price_TI_tot*price_TI_tot;
        deltas_lvl[1][0]     += delta_TI_tot;
        deltas_var_lvl[1][0] += delta_TI_tot*delta_TI_tot;
        
        double temp_PE       = pricedelta.first  * disc_PE;                  
        double delta_temp_PE = pricedelta.second * disc_PE;

        prices_lvl[0][0]     += temp_PE;
        prices_var_lvl[0][0] += temp_PE*temp_PE;
        deltas_lvl[0][0]     += delta_temp_PE;
        deltas_var_lvl[0][0] += delta_temp_PE*delta_temp_PE;

        double temp_EI       = perc_prem_death*price_TI_tot + temp_PE;
        double delta_temp_EI = perc_prem_death*delta_TI_tot + delta_temp_PE;

        prices_lvl[2][0]     += temp_EI;
        prices_var_lvl[2][0] += temp_EI*temp_EI;
        deltas_lvl[2][0]     += delta_temp_EI;
        deltas_var_lvl[2][0] += delta_temp_EI*delta_temp_EI;
      }
            
		}
    
	}

  for(int i=0;i<prices_lvl.size();i++)
  {
    prices_lvl[i][0]     /= double(M);
    prices_var_lvl[i][0] = (prices_var_lvl[i][0] - M*prices_lvl[i][0]*prices_lvl[i][0]) / (M-1.0);
    deltas_lvl[i][0]     /= double(M);
    deltas_var_lvl[i][0] = (deltas_var_lvl[i][0] - M*deltas_lvl[i][0]*deltas_lvl[i][0]) / (M-1.0);
  }

  price_PE += prices_lvl[0][0]; CI_price_PE += prices_var_lvl[0][0] / M ; 
  delta_PE += deltas_lvl[0][0]; CI_delta_PE += deltas_var_lvl[0][0] / M ; 

  price_TI += prices_lvl[1][0]; CI_price_TI += prices_var_lvl[1][0] / M ; 
  delta_TI += deltas_lvl[1][0]; CI_delta_TI += deltas_var_lvl[1][0] / M ; 

  price_EI += prices_lvl[2][0]; CI_price_EI += prices_var_lvl[2][0] / M ; 
  delta_EI += deltas_lvl[2][0]; CI_delta_EI += deltas_var_lvl[2][0] / M ; 

	for(int lvl=1;lvl<N_lvl;lvl++)
  { 
    N = 1.0;
    for(int l = 0; l < lvl-1; ++l) N *= 2;
    
    h         = TtoMat/double(N);
    double h2 = TtoMat/double(2*N);

    prm_NV_CIR(h,  beta, kappa, s, C0,   C1,   C2,   C3); //resetting parameters for CIR, new h

    ArrayXd C0_r(dim), C1_r(dim), C2_r(dim), C3_r(dim);
    prm_NV_CIR(h2, beta, kappa, s, C0_r, C1_r, C2_r, C3_r); //setting parameters for NV scheme for CIR

    var_start          = h  * 0.5 * pi_start.dot(cov_mat*pi_start);  //new h
    double var_start_r = h2 * 0.5 * pi_start.dot(cov_mat*pi_start);

    int M_lvl = std::ceil(std::max(sqrt(2.0/3.0)*pow(2, -3*lvl/2.0)*M, 1.0)); //no minimum
    //int M_lvl = std::ceil(std::max(sqrt(2.0/3.0)*pow(2, -3*lvl/2.0)*M, 2.0)); // minimum of 2 simulations

    for(int j=0; j<M_lvl; j++)
    {
      double price_TI_tot = 0.0;
      double delta_TI_tot = 0.0;

      double price_TI_tot_r = 0.0;
      double delta_TI_tot_r = 0.0;

      double fact = GM_spXmi_discounted(current_age, phi_GM, m_GM, b_GM, 0.0, r);

      if     (X0<min_gua_start){
        price_TI_tot_r += 0.5*h2*fact*min_gua_start;
        price_TI_tot   += 0.5*h *fact*min_gua_start;
      } 
      else if(X0>max_gua_start)
      {
        price_TI_tot_r += 0.5*h2*fact*max_gua_start;
        price_TI_tot   += 0.5*h *fact*max_gua_start;
      } 
      else{                     
        price_TI_tot_r += 0.5*h2*fact*X0;
        price_TI_tot   += 0.5*h *fact*X0;

        delta_TI_tot_r += 0.5*h2*fact;
        delta_TI_tot   += 0.5*h *fact;
      }


      double var_increment   = var_start;
      double var_increment_r = var_start_r;
      double var_val   = 0.0;
      double var_val_r = 0.0;
      ArrayXd Y   = Y0;
      ArrayXd Y_r = Y0;
      ArrayXd G(dim);
      ArrayXd G_sum(dim);
      double TimeWindow = 0.0;

      for(int i=0;i<N;i++)
      { 
        TimeWindow += h2;
        var_val_r  += var_increment_r;
        for(int l=0;l<dim;l++) G(l) = normcdfinv_as241(U(gen));
        G_sum = G;
        NV_CIR(Y_r, C0_r, C1_r, C2_r, C3_r, G);

        std::pair<double,double> pricedelta_r =
              utility_func_TI(X0, h2, r, TimeWindow, var_increment_r, var_val_r, Y_r, vols_sq_penalised, 
                        mu_mod, cov_mat, d_cov_mat, min_gua_start, max_gua_start, min_rate, max_rate);


        fact = GM_spXmi_discounted(current_age, phi_GM, m_GM, b_GM, TimeWindow, r);
        price_TI_tot_r += h2*pricedelta_r.first *fact;
        delta_TI_tot_r += h2*pricedelta_r.second*fact;

        TimeWindow += h2;
        var_val_r += var_increment_r;
        for(int l=0;l<dim;l++) G(l) = normcdfinv_as241(U(gen));
        G_sum += G;
        G_sum /= sqrt(2.0);
        NV_CIR(Y_r, C0_r, C1_r, C2_r, C3_r, G);

        pricedelta_r = 
              utility_func_TI(X0, h2, r, TimeWindow, var_increment_r, var_val_r, Y_r, vols_sq_penalised, 
                        mu_mod, cov_mat, d_cov_mat, min_gua_start, max_gua_start, min_rate, max_rate);

        fact = GM_spXmi_discounted(current_age, phi_GM, m_GM, b_GM, TimeWindow, r);

        // rough trajectory
        var_val += var_increment;
        NV_CIR(Y, C0, C1, C2, C3, G_sum);

        std::pair<double,double> pricedelta = 
              utility_func_TI(X0, h, r, TimeWindow, var_increment, var_val, Y, vols_sq_penalised, 
                    mu_mod, cov_mat, d_cov_mat, min_gua_start, max_gua_start, min_rate, max_rate);

        if(i<N-1)
        { 
          price_TI_tot_r +=     h2*pricedelta_r.first *fact;
          delta_TI_tot_r +=     h2*pricedelta_r.second*fact;

          price_TI_tot   +=     h *pricedelta.first   *fact;
          delta_TI_tot   +=     h *pricedelta.second  *fact;
        }
        else
        { 
          price_TI_tot_r += 0.5*h2*pricedelta_r.first *fact;
          delta_TI_tot_r += 0.5*h2*pricedelta_r.second*fact;
          price_TI_tot   += 0.5*h *pricedelta.first   *fact;
          delta_TI_tot   += 0.5*h *pricedelta.second  *fact;

          double temp_TI         = price_TI_tot_r - price_TI_tot;
          double delta_temp_TI   = delta_TI_tot_r - delta_TI_tot;
          
          
          prices_lvl[1][lvl]     += temp_TI;
          prices_var_lvl[1][lvl] += temp_TI*temp_TI;
          deltas_lvl[1][lvl]     += delta_temp_TI;
          deltas_var_lvl[1][lvl] += delta_temp_TI*delta_temp_TI;

          
          double temp_PE       = (pricedelta_r.first  - pricedelta.first ) * disc_PE;                  
          double delta_temp_PE = (pricedelta_r.second - pricedelta.second) * disc_PE;

          prices_lvl[0][lvl]     += temp_PE;
          prices_var_lvl[0][lvl] += temp_PE*temp_PE;
          deltas_lvl[0][lvl]     += delta_temp_PE;
          deltas_var_lvl[0][lvl] += delta_temp_PE*delta_temp_PE;

          double temp_EI       = perc_prem_death*temp_TI       + temp_PE;
          double delta_temp_EI = perc_prem_death*delta_temp_TI + delta_temp_PE;

          prices_lvl[2][lvl]     += temp_EI;
          prices_var_lvl[2][lvl] += temp_EI*temp_EI;
          deltas_lvl[2][lvl]     += delta_temp_EI;
          deltas_var_lvl[2][lvl] += delta_temp_EI*delta_temp_EI;
        }
              
      }
      
    }

    for(int i=0;i<prices_lvl.size();i++)
    {
      prices_lvl[i][lvl]     /= double(M_lvl);
      prices_var_lvl[i][lvl] = (prices_var_lvl[i][lvl] - M_lvl*prices_lvl[i][lvl]*prices_lvl[i][lvl]) / (M_lvl); //biased
      deltas_lvl[i][lvl]     /= double(M_lvl);
	    deltas_var_lvl[i][lvl] = (deltas_var_lvl[i][lvl] - M_lvl*deltas_lvl[i][lvl]*deltas_lvl[i][lvl]) / (M_lvl); //biased
    }

    price_PE += prices_lvl[0][lvl]; CI_price_PE += prices_var_lvl[0][lvl] / M_lvl ; 
    delta_PE += deltas_lvl[0][lvl]; CI_delta_PE += deltas_var_lvl[0][lvl] / M_lvl ; 

    price_TI += prices_lvl[1][lvl]; CI_price_TI += prices_var_lvl[1][lvl] / M_lvl ; 
    delta_TI += deltas_lvl[1][lvl]; CI_delta_TI += deltas_var_lvl[1][lvl] / M_lvl ; 

    price_EI += prices_lvl[2][lvl]; CI_price_EI += prices_var_lvl[2][lvl] / M_lvl ; 
    delta_EI += deltas_lvl[2][lvl]; CI_delta_EI += deltas_var_lvl[2][lvl] / M_lvl ; 

  }

  CI_price_PE = 1.96 * sqrt( CI_price_PE ); 
  CI_delta_PE = 1.96 * sqrt( CI_delta_PE ); 

  CI_price_TI = 1.96 * sqrt( CI_price_TI ); 
  CI_delta_TI = 1.96 * sqrt( CI_delta_TI );  

  CI_price_EI = 1.96 * sqrt( CI_price_EI );  
  CI_delta_EI = 1.96 * sqrt( CI_delta_EI ); 

}




void PETIEI_min_max_GM_MMM_mc(double CRRA_delta, double X0, double r, ArrayXd mu, ArrayXd vols, MatrixXd cov_mat, 
									double alpha_penalty, ArrayXd Y0, ArrayXd beta, ArrayXd kappa, ArrayXd s,
                  std::vector<double> nominals_PE, std::vector<double> current_ages_PE,
                  std::vector<double> nominals_TI, std::vector<double> current_ages_TI,
                  std::vector<double> nominals_EI, std::vector<double> current_ages_EI, 
                  double phi_GM, double m_GM, double b_GM,  
                  double min_gua_start, double max_gua_start, double min_rate, double max_rate, double perc_prem_death,
                  double TtoMat, int N, int M,
                  double &price_PE, double &var_price_PE, double &delta_PE, double &var_delta_PE,
                  double &price_TI, double &var_price_TI, double &delta_TI, double &var_delta_TI,
                  double &price_EI, double &var_price_EI, double &delta_EI, double &var_delta_EI)
{	
  // Term Insurance option with min and max guaranteed threshold 
  // option evalueted under the minimal martingale probability
	VectorXd mu_mod = (mu-r).matrix();
	ArrayXd vols_sq_penalised = alpha_penalty*vols.square();
	MatrixXd d_cov_mat = CRRA_delta * cov_mat;
	double h = TtoMat/double(N);
	auto dim = (beta.size()); 
	ArrayXd C0(dim), C1(dim), C2(dim), C3(dim);
	prm_NV_CIR(h, beta, kappa, s, C0, C1, C2, C3); //setting parameters for NV scheme for CIR

	ArrayXd e_CI = Y0 * vols_sq_penalised;

	// for large n add and subctrat e_CI is faster than allocate new matrixes everytime
	d_cov_mat.diagonal().array() += e_CI;
	Eigen::LLT<Eigen::MatrixXd> llt_solver_start(d_cov_mat);
	VectorXd pi_start = llt_solver_start.solve(mu_mod);
	d_cov_mat.diagonal().array() -= e_CI;


	double var_start = h * 0.5 * pi_start.dot(cov_mat*pi_start); 

	thread_local boost::random::xoshiro256pp gen(boost::random::random_device{}());    // thread safe
  boost::random::uniform_real_distribution<double> U(0, 1); // uni dist over (0,1)

  price_PE = 0.0; var_price_PE  = 0.0;
  delta_PE = 0.0; var_delta_PE  = 0.0;

	price_TI = 0.0; var_price_TI  = 0.0;
  delta_TI = 0.0; var_delta_TI  = 0.0;

  price_EI = 0.0; var_price_EI  = 0.0;
  delta_EI = 0.0; var_delta_EI  = 0.0;

  double disc_PE = GM_sp_discounted_mc(nominals_PE, current_ages_PE, phi_GM, m_GM, b_GM, TtoMat, r);
  double disc_EI = GM_sp_discounted_mc(nominals_EI, current_ages_EI, phi_GM, m_GM, b_GM, TtoMat, r);
  
	for(int j=0; j<M; j++)
	{
    double price_TI_tot = 0.0;
    double delta_TI_tot = 0.0;

    double price_EI_tot = 0.0;
    double delta_EI_tot = 0.0;

    double fact_TI = GM_spXmi_discounted_mc(nominals_TI, current_ages_TI, phi_GM, m_GM, b_GM, 0.0, r);
    double fact_EI = GM_spXmi_discounted_mc(nominals_EI, current_ages_EI, phi_GM, m_GM, b_GM, 0.0, r);

    double payoff_0;
    double delta_0  = 0.0;
    if     (X0<min_gua_start) payoff_0 = min_gua_start;
    else if(X0>max_gua_start) payoff_0 = max_gua_start;
    else{
                              payoff_0 = X0;
                              delta_0  = 1.0;
    }
    price_TI_tot += 0.5*h*fact_TI*payoff_0;
    price_EI_tot += 0.5*h*fact_EI*payoff_0;

    delta_TI_tot += 0.5*h*fact_TI*delta_0;
    delta_EI_tot += 0.5*h*fact_EI*delta_0;

    
    double var_increment = var_start;
    double var_val = 0.0;
		ArrayXd Y = Y0;
		ArrayXd G(dim);
    double TimeWindow = 0.0;

		for(int i=0;i<N;i++)
		{ 
      TimeWindow +=h;
      var_val += var_increment;
			for(int l=0;l<dim;l++) G(l) = normcdfinv_as241(U(gen));
			NV_CIR(Y, C0, C1, C2, C3, G);

			std::pair<double,double>  pricedelta =
            utility_func_TI(X0, h, r, TimeWindow, var_increment, var_val,  Y,  vols_sq_penalised, 
                    mu_mod, cov_mat, d_cov_mat, min_gua_start, max_gua_start, min_rate, max_rate);

      fact_TI = GM_spXmi_discounted_mc(nominals_TI, current_ages_TI, phi_GM, m_GM, b_GM, TimeWindow, r);
      fact_EI = GM_spXmi_discounted_mc(nominals_EI, current_ages_EI, phi_GM, m_GM, b_GM, TimeWindow, r);
      if(i<N-1)
      {
        price_TI_tot +=     h*pricedelta.first *fact_TI;
        delta_TI_tot +=     h*pricedelta.second*fact_TI;
        price_EI_tot +=     h*pricedelta.first *fact_EI;
        delta_EI_tot +=     h*pricedelta.second*fact_EI;
      }
      else 
      {
        price_TI_tot += 0.5*h*pricedelta.first *fact_TI;
        delta_TI_tot += 0.5*h*pricedelta.second*fact_TI;
        price_EI_tot += 0.5*h*pricedelta.first *fact_EI;
        delta_EI_tot += 0.5*h*pricedelta.second*fact_EI;

        price_TI      += price_TI_tot;
        var_price_TI  += price_TI_tot*price_TI_tot;
        delta_TI      += delta_TI_tot;
        var_delta_TI  += delta_TI_tot*delta_TI_tot;


        double temp_PE       = pricedelta.first  * disc_PE;                  
        double delta_temp_PE = pricedelta.second * disc_PE;

        price_PE     += temp_PE;
        var_price_PE += temp_PE*temp_PE;
        delta_PE     += delta_temp_PE;
        var_delta_PE += delta_temp_PE*delta_temp_PE;

        double temp_EI       = perc_prem_death*price_EI_tot + pricedelta.first  * disc_EI;
        double delta_temp_EI = perc_prem_death*delta_EI_tot + pricedelta.second * disc_EI;

        price_EI     += temp_EI;
        var_price_EI += temp_EI*temp_EI;
        delta_EI     += delta_temp_EI;
        var_delta_EI += delta_temp_EI*delta_temp_EI;
      }
            
		}

	}

  delta_PE /= double(M);
  var_delta_PE = (var_delta_PE - M * delta_PE * delta_PE) / double(M-1.0);

	price_PE /= double(M);
	var_price_PE = (var_price_PE - M * price_PE * price_PE) / double(M-1.0);


  delta_TI /= double(M);
  var_delta_TI = (var_delta_TI - M * delta_TI * delta_TI) / double(M-1.0);

	price_TI /= double(M);
	var_price_TI = (var_price_TI - M * price_TI * price_TI) / double(M-1.0);


  delta_EI /= double(M);
  var_delta_EI = (var_delta_EI - M * delta_EI * delta_EI) / double(M-1.0);

	price_EI /= double(M);
	var_price_EI = (var_price_EI - M * price_EI * price_EI) / double(M-1.0);
}




// Ordinary Monte Carlo general-model pricing on N time intervals.
// The var_* outputs are sample variances; a 95% half-width is 1.96*sqrt(var_*/M).
void PETIEI_min_max_GM_MMM_mc_general_model(double CRRA_delta, double X0, double r, 
                  const ArrayXd &mu, const ArrayXd &vols, const MatrixXd &cov_mat, 
									double alpha_penalty, const ArrayXd &Y0, const ArrayXd &beta, const ArrayXd &kappa, const ArrayXd &s,
                  const std::vector<double> &nominals_PE, const std::vector<double> &current_ages_PE,
                  const std::vector<double> &nominals_TI, const std::vector<double> &current_ages_TI,
                  const std::vector<double> &nominals_EI, const std::vector<double> &current_ages_EI, 
                  double phi_GM, double m_GM, double b_GM,  
                  double min_gua_start, double max_gua_start, double min_rate, double max_rate, double perc_prem_death,
                  double TtoMat, int N, int M,
                  double &price_PE, double &var_price_PE, double &delta_PE, double &var_delta_PE,
                  double &price_TI, double &var_price_TI, double &delta_TI, double &var_delta_TI,
                  double &price_EI, double &var_price_EI, double &delta_EI, double &var_delta_EI,
                  prm_comp prm_func, scheme_update scheme)
{	
  // Term Insurance option with min and max guaranteed threshold 
  // option evalueted under the minimal martingale probability
	VectorXd mu_mod = (mu-r).matrix();
	ArrayXd vols_sq_penalised = alpha_penalty*vols.square();
	MatrixXd d_cov_mat = CRRA_delta * cov_mat;
	double h = TtoMat/double(N);
	auto dim = (beta.size()); 
	ArrayXd C0(dim), C1(dim), C2(dim), C3(dim);
	prm_func(h, beta, kappa, s, C0, C1, C2, C3); //setting parameters for the selected carbon model


  const double half_h = 0.5 * h;

  ArrayXd base_penalised_diagonal =
    d_cov_mat.diagonal().array();
  
  Eigen::LLT<MatrixXd> llt_solver;

  VectorXd pi(dim);
  VectorXd cov_times_pi(dim);

  // Apply the same equation-(2.2) indicator to the initial carbon state.
  d_cov_mat.diagonal().array() = base_penalised_diagonal + Y0.max(0.0) * vols_sq_penalised;
	llt_solver.compute(d_cov_mat);
	pi = llt_solver.solve(mu_mod);
	

  cov_times_pi.noalias() = cov_mat * pi;

	double var_start = half_h * pi.dot(cov_times_pi); 

	thread_local boost::random::xoshiro256pp gen(boost::random::random_device{}());    // thread safe
  boost::random::uniform_real_distribution<double> U(0, 1); // uni dist over (0,1)

  price_PE = 0.0; var_price_PE  = 0.0;
  delta_PE = 0.0; var_delta_PE  = 0.0;

	price_TI = 0.0; var_price_TI  = 0.0;
  delta_TI = 0.0; var_delta_TI  = 0.0;

  price_EI = 0.0; var_price_EI  = 0.0;
  delta_EI = 0.0; var_delta_EI  = 0.0;

  double disc_PE = GM_sp_discounted_mc(nominals_PE, current_ages_PE, phi_GM, m_GM, b_GM, TtoMat, r);
  double disc_EI = GM_sp_discounted_mc(nominals_EI, current_ages_EI, phi_GM, m_GM, b_GM, TtoMat, r);

  std::vector<double> fact_TI(N + 1);
  std::vector<double> fact_EI(N + 1);
  std::vector<double> min_guarantee(N + 1);
  std::vector<double> max_guarantee(N + 1);
  std::vector<double> risk_free_growth(N + 1);

  std::vector<double> log_min_forward_moneyness(N + 1);
  std::vector<double> log_max_forward_moneyness(N + 1);

  double t = 0.0;

  for (int i = 0; i <= N; ++i) {
    fact_TI[i] =
      GM_spXmi_discounted_mc(nominals_TI, current_ages_TI, phi_GM, m_GM, b_GM, t, r);
    fact_TI[i] *= h * ((i == 0 || i == N) ? 0.5 : 1.0);
    fact_EI[i] =
      GM_spXmi_discounted_mc(nominals_EI, current_ages_EI, phi_GM, m_GM, b_GM, t, r);
    fact_EI[i] *= h * ((i == 0 || i == N) ? 0.5 : 1.0);

    min_guarantee[i] = min_gua_start * exp(min_rate * t);
    max_guarantee[i] = max_gua_start * exp(max_rate * t);
    risk_free_growth[i] = exp(r * t);

    log_min_forward_moneyness[i] =
      log(min_gua_start / X0) + (min_rate - r) * t;

    log_max_forward_moneyness[i] =
      log(max_gua_start / X0) + (max_rate - r) * t;

    t += h;
  }


  double payoff_0;
  double delta_0  = 0.0;
  if     (X0<min_gua_start) payoff_0 = min_gua_start;
  else if(X0>max_gua_start) payoff_0 = max_gua_start;
  else{
                            payoff_0 = X0;
                            delta_0  = 1.0;
  }

  
	for(int j=0; j<M; j++)
	{ 
    double price_TI_tot = fact_TI[0]*payoff_0;
    double delta_TI_tot = fact_TI[0]*delta_0;

    double price_EI_tot = fact_EI[0]*payoff_0;
    double delta_EI_tot = fact_EI[0]*delta_0;

    double var_increment = var_start;
    double var_val = 0.0;
		ArrayXd Y = Y0;
		ArrayXd G(dim);

		for(int i=0;i<N;i++)
		{ 
			for(int l=0;l<dim;l++) G(l) = normcdfinv_as241(U(gen));
			scheme(Y, C0, C1, C2, C3, G);

			std::pair<double,double>  pricedelta =
            utility_func_TI_precomputed(X0, half_h, 
              min_guarantee[i+1], max_guarantee[i+1], log_min_forward_moneyness[i+1], 
              log_max_forward_moneyness[i+1],  risk_free_growth[i+1],
              var_increment, var_val,  Y,  vols_sq_penalised, mu_mod, cov_mat, 
              base_penalised_diagonal, d_cov_mat, llt_solver, pi, cov_times_pi);


      price_TI_tot += pricedelta.first *fact_TI[i+1];
      delta_TI_tot += pricedelta.second*fact_TI[i+1];
      price_EI_tot += pricedelta.first *fact_EI[i+1];
      delta_EI_tot += pricedelta.second*fact_EI[i+1];
      if (i==N-1)
      {
        price_TI      += price_TI_tot;
        var_price_TI  += price_TI_tot*price_TI_tot;
        delta_TI      += delta_TI_tot;
        var_delta_TI  += delta_TI_tot*delta_TI_tot;


        double temp_PE       = pricedelta.first  * disc_PE;                  
        double delta_temp_PE = pricedelta.second * disc_PE;

        price_PE     += temp_PE;
        var_price_PE += temp_PE*temp_PE;
        delta_PE     += delta_temp_PE;
        var_delta_PE += delta_temp_PE*delta_temp_PE;

        double temp_EI       = perc_prem_death*price_EI_tot + pricedelta.first  * disc_EI;
        double delta_temp_EI = perc_prem_death*delta_EI_tot + pricedelta.second * disc_EI;

        price_EI     += temp_EI;
        var_price_EI += temp_EI*temp_EI;
        delta_EI     += delta_temp_EI;
        var_delta_EI += delta_temp_EI*delta_temp_EI;
      }
            
		}

	}

  delta_PE /= double(M);
  var_delta_PE = (var_delta_PE - M * delta_PE * delta_PE) / double(M-1.0);

	price_PE /= double(M);
	var_price_PE = (var_price_PE - M * price_PE * price_PE) / double(M-1.0);


  delta_TI /= double(M);
  var_delta_TI = (var_delta_TI - M * delta_TI * delta_TI) / double(M-1.0);

	price_TI /= double(M);
	var_price_TI = (var_price_TI - M * price_TI * price_TI) / double(M-1.0);


  delta_EI /= double(M);
  var_delta_EI = (var_delta_EI - M * delta_EI * delta_EI) / double(M-1.0);

	price_EI /= double(M);
	var_price_EI = (var_price_EI - M * price_EI * price_EI) / double(M-1.0);
}


// Sampling policy changes only the four Gaussian carbon shocks per time step.
// The inverse CDFs require 0 < p < 1; uniform_real_distribution can return zero.
inline double open_unit_uniform(boost::random::uniform_real_distribution<double> &uniform,
                                boost::random::xoshiro256pp &gen) {
  double p;
  do {
    p = uniform(gen);
  } while (p == 0.0);
  return p;
}

struct As241NormalSampler {
  boost::random::uniform_real_distribution<double> uniform{0.0, 1.0};
  double operator()(boost::random::xoshiro256pp &gen) {
    return normcdfinv_as241(open_unit_uniform(uniform, gen));
  }
};

struct AcklamNormalSampler {
  boost::random::uniform_real_distribution<double> uniform{0.0, 1.0};
  double operator()(boost::random::xoshiro256pp &gen) {
    return normcdfinv_acklam(open_unit_uniform(uniform, gen));
  }
};

struct ZigguratNormalSampler {
  boost::random::normal_distribution<double> normal{0.0, 1.0};
  double operator()(boost::random::xoshiro256pp &gen) {
    return normal(gen);
  }
};

// Four-factor pricer: the model and all carbon/covariance workspaces are fixed at compile time.
template <class CarbonModel, class NormalSampler = As241NormalSampler>
void PETIEI_min_max_GM_MMM_mc_general_model(double CRRA_delta, double X0, double r, 
                  const Eigen::Vector4d &mu, const Eigen::Vector4d &vols, const Eigen::Matrix4d &cov_mat, 
									double alpha_penalty, const Eigen::Vector4d &Y0, const Eigen::Vector4d &beta, const Eigen::Vector4d &kappa, const Eigen::Vector4d &s,
                  const std::vector<double> &nominals_PE, const std::vector<double> &current_ages_PE,
                  const std::vector<double> &nominals_TI, const std::vector<double> &current_ages_TI,
                  const std::vector<double> &nominals_EI, const std::vector<double> &current_ages_EI, 
                  double phi_GM, double m_GM, double b_GM,  
                  double min_gua_start, double max_gua_start, double min_rate, double max_rate, double perc_prem_death,
                  double TtoMat, int N, int M,
                  double &price_PE, double &var_price_PE, double &delta_PE, double &var_delta_PE,
                  double &price_TI, double &var_price_TI, double &delta_TI, double &var_delta_TI,
                  double &price_EI, double &var_price_EI, double &delta_EI, double &var_delta_EI)
{	
  // Term Insurance option with min and max guaranteed threshold 
  // option evalueted under the minimal martingale probability
  Eigen::Vector4d mu_mod = (mu.array()-r).matrix();
  Eigen::Vector4d vols_sq_penalised = (alpha_penalty*vols.array().square()).matrix();
  Eigen::Matrix4d d_cov_mat = CRRA_delta * cov_mat;
	double h = TtoMat/double(N);
  constexpr int dim = 4;
  Eigen::Vector4d C0, C1, C2, C3;
  CarbonModel::prepare(h, beta, kappa, s, C0, C1, C2, C3);


  const double half_h = 0.5 * h;

  Eigen::Vector4d base_penalised_diagonal = d_cov_mat.diagonal();
  
  Eigen::LLT<Eigen::Matrix4d> llt_solver;

  Eigen::Vector4d pi;
  Eigen::Vector4d cov_times_pi;

  // Apply the same equation-(2.2) indicator to the initial carbon state.
  d_cov_mat.diagonal().array() = base_penalised_diagonal.array()
                               + Y0.array().max(0.0) * vols_sq_penalised.array();
  solve_penalised_covariance(d_cov_mat, mu_mod, llt_solver, pi);
	

  cov_times_pi.noalias() = cov_mat * pi;

	double var_start = half_h * pi.dot(cov_times_pi); 

	thread_local boost::random::xoshiro256pp gen(boost::random::random_device{}());    // thread safe
  NormalSampler normal;

  price_PE = 0.0; var_price_PE  = 0.0;
  delta_PE = 0.0; var_delta_PE  = 0.0;

	price_TI = 0.0; var_price_TI  = 0.0;
  delta_TI = 0.0; var_delta_TI  = 0.0;

  price_EI = 0.0; var_price_EI  = 0.0;
  delta_EI = 0.0; var_delta_EI  = 0.0;

  double disc_PE = GM_sp_discounted_mc(nominals_PE, current_ages_PE, phi_GM, m_GM, b_GM, TtoMat, r);
  double disc_EI = GM_sp_discounted_mc(nominals_EI, current_ages_EI, phi_GM, m_GM, b_GM, TtoMat, r);

  std::vector<double> fact_TI(N + 1);
  std::vector<double> fact_EI(N + 1);
  std::vector<double> min_guarantee(N + 1);
  std::vector<double> max_guarantee(N + 1);
  std::vector<double> risk_free_growth(N + 1);

  std::vector<double> log_min_forward_moneyness(N + 1);
  std::vector<double> log_max_forward_moneyness(N + 1);

  double t = 0.0;

  for (int i = 0; i <= N; ++i) {
    fact_TI[i] =
      GM_spXmi_discounted_mc(nominals_TI, current_ages_TI, phi_GM, m_GM, b_GM, t, r);
    fact_TI[i] *= h * ((i == 0 || i == N) ? 0.5 : 1.0);
    fact_EI[i] =
      GM_spXmi_discounted_mc(nominals_EI, current_ages_EI, phi_GM, m_GM, b_GM, t, r);
    fact_EI[i] *= h * ((i == 0 || i == N) ? 0.5 : 1.0);

    min_guarantee[i] = min_gua_start * exp(min_rate * t);
    max_guarantee[i] = max_gua_start * exp(max_rate * t);
    risk_free_growth[i] = exp(r * t);

    log_min_forward_moneyness[i] =
      log(min_gua_start / X0) + (min_rate - r) * t;

    log_max_forward_moneyness[i] =
      log(max_gua_start / X0) + (max_rate - r) * t;

    t += h;
  }


  double payoff_0;
  double delta_0  = 0.0;
  if     (X0<min_gua_start) payoff_0 = min_gua_start;
  else if(X0>max_gua_start) payoff_0 = max_gua_start;
  else{
                            payoff_0 = X0;
                            delta_0  = 1.0;
  }

  
  Eigen::Vector4d Y, G;
	for(int j=0; j<M; j++)
	{ 
    double price_TI_tot = fact_TI[0]*payoff_0;
    double delta_TI_tot = fact_TI[0]*delta_0;

    double price_EI_tot = fact_EI[0]*payoff_0;
    double delta_EI_tot = fact_EI[0]*delta_0;

    double var_increment = var_start;
    double var_val = 0.0;
    Y = Y0;

		for(int i=0;i<N;i++)
		{ 
			for(int l=0;l<dim;l++) G(l) = normal(gen);
      CarbonModel::step(Y, C0, C1, C2, C3, G);

			std::pair<double,double>  pricedelta =
            utility_func_TI_precomputed(X0, half_h, 
              min_guarantee[i+1], max_guarantee[i+1], log_min_forward_moneyness[i+1], 
              log_max_forward_moneyness[i+1],  risk_free_growth[i+1],
              var_increment, var_val,  Y,  vols_sq_penalised, mu_mod, cov_mat, 
              base_penalised_diagonal, d_cov_mat, llt_solver, pi, cov_times_pi);


      price_TI_tot += pricedelta.first *fact_TI[i+1];
      delta_TI_tot += pricedelta.second*fact_TI[i+1];
      price_EI_tot += pricedelta.first *fact_EI[i+1];
      delta_EI_tot += pricedelta.second*fact_EI[i+1];
      if (i==N-1)
      {
        price_TI      += price_TI_tot;
        var_price_TI  += price_TI_tot*price_TI_tot;
        delta_TI      += delta_TI_tot;
        var_delta_TI  += delta_TI_tot*delta_TI_tot;


        double temp_PE       = pricedelta.first  * disc_PE;                  
        double delta_temp_PE = pricedelta.second * disc_PE;

        price_PE     += temp_PE;
        var_price_PE += temp_PE*temp_PE;
        delta_PE     += delta_temp_PE;
        var_delta_PE += delta_temp_PE*delta_temp_PE;

        double temp_EI       = perc_prem_death*price_EI_tot + pricedelta.first  * disc_EI;
        double delta_temp_EI = perc_prem_death*delta_EI_tot + pricedelta.second * disc_EI;

        price_EI     += temp_EI;
        var_price_EI += temp_EI*temp_EI;
        delta_EI     += delta_temp_EI;
        var_delta_EI += delta_temp_EI*delta_temp_EI;
      }
            
		}

	}

  delta_PE /= double(M);
  var_delta_PE = (var_delta_PE - M * delta_PE * delta_PE) / double(M-1.0);

	price_PE /= double(M);
	var_price_PE = (var_price_PE - M * price_PE * price_PE) / double(M-1.0);


  delta_TI /= double(M);
  var_delta_TI = (var_delta_TI - M * delta_TI * delta_TI) / double(M-1.0);

	price_TI /= double(M);
	var_price_TI = (var_price_TI - M * price_TI * price_TI) / double(M-1.0);


  delta_EI /= double(M);
  var_delta_EI = (var_delta_EI - M * delta_EI * delta_EI) / double(M-1.0);

	price_EI /= double(M);
	var_price_EI = (var_price_EI - M * price_EI * price_EI) / double(M-1.0);
}

// Four-factor general-model pricer using Boost Ziggurat and xoshiro256pp.
template <class CarbonModel>
void PETIEI_min_max_GM_MMM_mc_general_model_ziggurat_xoshiro(double CRRA_delta, double X0, double r, 
                  const Eigen::Vector4d &mu, const Eigen::Vector4d &vols, const Eigen::Matrix4d &cov_mat, 
									double alpha_penalty, const Eigen::Vector4d &Y0, const Eigen::Vector4d &beta, const Eigen::Vector4d &kappa, const Eigen::Vector4d &s,
                  const std::vector<double> &nominals_PE, const std::vector<double> &current_ages_PE,
                  const std::vector<double> &nominals_TI, const std::vector<double> &current_ages_TI,
                  const std::vector<double> &nominals_EI, const std::vector<double> &current_ages_EI, 
                  double phi_GM, double m_GM, double b_GM,  
                  double min_gua_start, double max_gua_start, double min_rate, double max_rate, double perc_prem_death,
                  double TtoMat, int N, int M,
                  double &price_PE, double &var_price_PE, double &delta_PE, double &var_delta_PE,
                  double &price_TI, double &var_price_TI, double &delta_TI, double &var_delta_TI,
                  double &price_EI, double &var_price_EI, double &delta_EI, double &var_delta_EI)
{
  PETIEI_min_max_GM_MMM_mc_general_model<CarbonModel, ZigguratNormalSampler>(
      CRRA_delta, X0, r, mu, vols, cov_mat, alpha_penalty,
      Y0, beta, kappa, s, nominals_PE, current_ages_PE,
      nominals_TI, current_ages_TI, nominals_EI, current_ages_EI,
      phi_GM, m_GM, b_GM, min_gua_start, max_gua_start,
      min_rate, max_rate, perc_prem_death, TtoMat, N, M,
      price_PE, var_price_PE, delta_PE, var_delta_PE,
      price_TI, var_price_TI, delta_TI, var_delta_TI,
      price_EI, var_price_EI, delta_EI, var_delta_EI);
}

// Four-factor general-model pricer using Acklam and xoshiro256pp.
template <class CarbonModel>
void PETIEI_min_max_GM_MMM_mc_general_model_acklam_xoshiro(double CRRA_delta, double X0, double r, 
                  const Eigen::Vector4d &mu, const Eigen::Vector4d &vols, const Eigen::Matrix4d &cov_mat, 
									double alpha_penalty, const Eigen::Vector4d &Y0, const Eigen::Vector4d &beta, const Eigen::Vector4d &kappa, const Eigen::Vector4d &s,
                  const std::vector<double> &nominals_PE, const std::vector<double> &current_ages_PE,
                  const std::vector<double> &nominals_TI, const std::vector<double> &current_ages_TI,
                  const std::vector<double> &nominals_EI, const std::vector<double> &current_ages_EI, 
                  double phi_GM, double m_GM, double b_GM,  
                  double min_gua_start, double max_gua_start, double min_rate, double max_rate, double perc_prem_death,
                  double TtoMat, int N, int M,
                  double &price_PE, double &var_price_PE, double &delta_PE, double &var_delta_PE,
                  double &price_TI, double &var_price_TI, double &delta_TI, double &var_delta_TI,
                  double &price_EI, double &var_price_EI, double &delta_EI, double &var_delta_EI)
{
  PETIEI_min_max_GM_MMM_mc_general_model<CarbonModel, AcklamNormalSampler>(
      CRRA_delta, X0, r, mu, vols, cov_mat, alpha_penalty,
      Y0, beta, kappa, s, nominals_PE, current_ages_PE,
      nominals_TI, current_ages_TI, nominals_EI, current_ages_EI,
      phi_GM, m_GM, b_GM, min_gua_start, max_gua_start,
      min_rate, max_rate, perc_prem_death, TtoMat, N, M,
      price_PE, var_price_PE, delta_PE, var_delta_PE,
      price_TI, var_price_TI, delta_TI, var_delta_TI,
      price_EI, var_price_EI, delta_EI, var_delta_EI);
}




void PETIEI_min_max_GM_MMM_mc_MLMC(double CRRA_delta, double X0, double r, ArrayXd mu, ArrayXd vols, MatrixXd cov_mat, 
									double alpha_penalty, ArrayXd Y0, ArrayXd beta, ArrayXd kappa, ArrayXd s,
                  std::vector<double> nominals_PE, std::vector<double> current_ages_PE,
                  std::vector<double> nominals_TI, std::vector<double> current_ages_TI,
                  std::vector<double> nominals_EI, std::vector<double> current_ages_EI, 
                  double phi_GM, double m_GM, double b_GM,  
                  double min_gua_start, double max_gua_start, double min_rate, double max_rate, double perc_prem_death,
                  double TtoMat, int N_lvl, int M,
                  double &price_PE, double &CI_price_PE, double &delta_PE, double &CI_delta_PE,
                  double &price_TI, double &CI_price_TI, double &delta_TI, double &CI_delta_TI,
                  double &price_EI, double &CI_price_EI, double &delta_EI, double &CI_delta_EI)
{	

  std::vector<std::vector<double>> prices_lvl(    3, std::vector<double>(N_lvl, 0.0));
  std::vector<std::vector<double>> deltas_lvl(    3, std::vector<double>(N_lvl, 0.0));
  std::vector<std::vector<double>> prices_var_lvl(3, std::vector<double>(N_lvl, 0.0));
  std::vector<std::vector<double>> deltas_var_lvl(3, std::vector<double>(N_lvl, 0.0));

  // Term Insurance option with min and max guaranteed threshold 
  // option evalueted under the minimal martingale probability
	VectorXd mu_mod = (mu-r).matrix();
	ArrayXd vols_sq_penalised = alpha_penalty*vols.square();
	MatrixXd d_cov_mat = CRRA_delta * cov_mat;
  int N = 1; 
	double h = TtoMat/double(N);
	auto dim = (beta.size()); 
	ArrayXd C0(dim), C1(dim), C2(dim), C3(dim);
	prm_NV_CIR(h, beta, kappa, s, C0, C1, C2, C3); //setting parameters for NV scheme for CIR

	ArrayXd e_CI = Y0 * vols_sq_penalised; // for large n add and subtract e_CI is faster than allocate new matrix everytime
	d_cov_mat.diagonal().array() += e_CI;
	Eigen::LLT<Eigen::MatrixXd> llt_solver_start(d_cov_mat);
	VectorXd pi_start = llt_solver_start.solve(mu_mod);
	d_cov_mat.diagonal().array() -= e_CI;

	double var_start   = h  * 0.5 * pi_start.dot(cov_mat*pi_start);


	thread_local boost::random::xoshiro256pp gen(boost::random::random_device{}());    // thread safe
  boost::random::uniform_real_distribution<double> U(0, 1); // uni dist over (0,1)

  price_PE = 0.0; CI_price_PE  = 0.0;
  delta_PE = 0.0; CI_delta_PE  = 0.0;

	price_TI = 0.0; CI_price_TI  = 0.0;
  delta_TI = 0.0; CI_delta_TI  = 0.0;

  price_EI = 0.0; CI_price_EI  = 0.0; 
  delta_EI = 0.0; CI_delta_EI  = 0.0;

  double disc_PE = GM_sp_discounted_mc(nominals_PE, current_ages_PE, phi_GM, m_GM, b_GM, TtoMat, r);
  double disc_EI = GM_sp_discounted_mc(nominals_EI, current_ages_EI, phi_GM, m_GM, b_GM, TtoMat, r);

	for(int j=0; j<M; j++)
	{
    double price_TI_tot = 0.0;
    double delta_TI_tot = 0.0;

    double price_EI_tot = 0.0;
    double delta_EI_tot = 0.0;

    double fact_TI = GM_spXmi_discounted_mc(nominals_TI, current_ages_TI, phi_GM, m_GM, b_GM, 0.0, r);
    double fact_EI = GM_spXmi_discounted_mc(nominals_EI, current_ages_EI, phi_GM, m_GM, b_GM, 0.0, r);

    double payoff_0;
    double delta_0  = 0.0;
    if     (X0<min_gua_start) payoff_0 = min_gua_start;
    else if(X0>max_gua_start) payoff_0 = max_gua_start;
    else{
                              payoff_0 = X0;
                              delta_0  = 1.0;
    }
    price_TI_tot += 0.5*h*fact_TI*payoff_0;
    price_EI_tot += 0.5*h*fact_EI*payoff_0;

    delta_TI_tot += 0.5*h*fact_TI*delta_0;
    delta_EI_tot += 0.5*h*fact_EI*delta_0;
    
    double var_increment = var_start;
    double var_val = 0.0;
		ArrayXd Y = Y0;
		ArrayXd G(dim);
    double TimeWindow = 0.0;

		for(int i=0;i<N;i++)
		{ 
      TimeWindow +=h;
      var_val += var_increment;
			for(int l=0;l<dim;l++) G(l) = normcdfinv_as241(U(gen));
			NV_CIR(Y, C0, C1, C2, C3, G);

      std::pair<double,double>  pricedelta =
            utility_func_TI(X0, h, r, TimeWindow, var_increment, var_val,  Y,  vols_sq_penalised, 
                    mu_mod, cov_mat, d_cov_mat, min_gua_start, max_gua_start, min_rate, max_rate);


      fact_TI = GM_spXmi_discounted_mc(nominals_TI, current_ages_TI, phi_GM, m_GM, b_GM, TimeWindow, r);
      fact_EI = GM_spXmi_discounted_mc(nominals_EI, current_ages_EI, phi_GM, m_GM, b_GM, TimeWindow, r);
      if(i<N-1)
      {  
        price_TI_tot +=     h*pricedelta.first *fact_TI;
        delta_TI_tot +=     h*pricedelta.second*fact_TI;
        price_EI_tot +=     h*pricedelta.first *fact_EI;
        delta_EI_tot +=     h*pricedelta.second*fact_EI;
      }
      else
      { 
        price_TI_tot += 0.5*h*pricedelta.first *fact_TI;
        delta_TI_tot += 0.5*h*pricedelta.second*fact_TI;
        price_EI_tot += 0.5*h*pricedelta.first *fact_EI;
        delta_EI_tot += 0.5*h*pricedelta.second*fact_EI;

        prices_lvl[1][0]     += price_TI_tot;
        prices_var_lvl[1][0] += price_TI_tot*price_TI_tot;
        deltas_lvl[1][0]     += delta_TI_tot;
        deltas_var_lvl[1][0] += delta_TI_tot*delta_TI_tot;
        
        double temp_PE       = pricedelta.first  * disc_PE;                  
        double delta_temp_PE = pricedelta.second * disc_PE;

        prices_lvl[0][0]     += temp_PE;
        prices_var_lvl[0][0] += temp_PE*temp_PE;
        deltas_lvl[0][0]     += delta_temp_PE;
        deltas_var_lvl[0][0] += delta_temp_PE*delta_temp_PE;

        double temp_EI       = perc_prem_death*price_EI_tot + pricedelta.first  * disc_EI;
        double delta_temp_EI = perc_prem_death*delta_EI_tot + pricedelta.second * disc_EI;

        prices_lvl[2][0]     += temp_EI;
        prices_var_lvl[2][0] += temp_EI*temp_EI;
        deltas_lvl[2][0]     += delta_temp_EI;
        deltas_var_lvl[2][0] += delta_temp_EI*delta_temp_EI;
      }
            
		}
    
	}

  for(int i=0;i<prices_lvl.size();i++)
  {
    prices_lvl[i][0]     /= double(M);
    prices_var_lvl[i][0] = (prices_var_lvl[i][0] - M*prices_lvl[i][0]*prices_lvl[i][0]) / (M-1.0);
    deltas_lvl[i][0]     /= double(M);
    deltas_var_lvl[i][0] = (deltas_var_lvl[i][0] - M*deltas_lvl[i][0]*deltas_lvl[i][0]) / (M-1.0);
  }

  price_PE += prices_lvl[0][0]; CI_price_PE += prices_var_lvl[0][0] / M ; 
  delta_PE += deltas_lvl[0][0]; CI_delta_PE += deltas_var_lvl[0][0] / M ; 

  price_TI += prices_lvl[1][0]; CI_price_TI += prices_var_lvl[1][0] / M ; 
  delta_TI += deltas_lvl[1][0]; CI_delta_TI += deltas_var_lvl[1][0] / M ; 

  price_EI += prices_lvl[2][0]; CI_price_EI += prices_var_lvl[2][0] / M ; 
  delta_EI += deltas_lvl[2][0]; CI_delta_EI += deltas_var_lvl[2][0] / M ; 

	for(int lvl=1;lvl<N_lvl;lvl++)
  { 
    N = 1.0;
    for(int l = 0; l < lvl-1; ++l) N *= 2;
    
    h         = TtoMat/double(N);
    double h2 = TtoMat/double(2*N);

    prm_NV_CIR(h,  beta, kappa, s, C0,   C1,   C2,   C3); //resetting parameters for CIR, new h

    ArrayXd C0_r(dim), C1_r(dim), C2_r(dim), C3_r(dim);
    prm_NV_CIR(h2, beta, kappa, s, C0_r, C1_r, C2_r, C3_r); //setting parameters for NV scheme for CIR

    var_start          = h  * 0.5 * pi_start.dot(cov_mat*pi_start);  //new h
    double var_start_r = h2 * 0.5 * pi_start.dot(cov_mat*pi_start);

    int M_lvl = std::ceil(std::max(sqrt(2.0/3.0)*pow(2, -3*lvl/2.0)*M, 1.0)); //no minimum
    //int M_lvl = std::ceil(std::max(sqrt(2.0/3.0)*pow(2, -3*lvl/2.0)*M, 2.0)); // minimum of 2 simulations

    for(int j=0; j<M_lvl; j++)
    {
      double price_TI_tot = 0.0;  double price_TI_tot_r = 0.0;
      double delta_TI_tot = 0.0;  double delta_TI_tot_r = 0.0;

      double price_EI_tot = 0.0;  double price_EI_tot_r = 0.0;
      double delta_EI_tot = 0.0;  double delta_EI_tot_r = 0.0;
      

      double fact_TI = GM_spXmi_discounted_mc(nominals_TI, current_ages_TI, phi_GM, m_GM, b_GM, 0.0, r);
      double fact_EI = GM_spXmi_discounted_mc(nominals_EI, current_ages_EI, phi_GM, m_GM, b_GM, 0.0, r);

      double payoff_0;
      double delta_0  = 0.0;
      if     (X0<min_gua_start) payoff_0 = min_gua_start;
      else if(X0>max_gua_start) payoff_0 = max_gua_start;
      else{
                                payoff_0 = X0;
                                delta_0  = 1.0;
      }
      price_TI_tot   += 0.5*h *fact_TI*payoff_0; price_EI_tot   += 0.5*h *fact_EI*payoff_0;
      price_TI_tot_r += 0.5*h2*fact_TI*payoff_0; price_EI_tot_r += 0.5*h2*fact_EI*payoff_0;

      delta_TI_tot   += 0.5*h *fact_TI*delta_0;  delta_EI_tot   += 0.5*h *fact_EI*delta_0;
      delta_TI_tot_r += 0.5*h2*fact_TI*delta_0;  delta_EI_tot_r += 0.5*h2*fact_EI*delta_0;


      double var_increment   = var_start;
      double var_increment_r = var_start_r;
      double var_val   = 0.0;
      double var_val_r = 0.0;
      ArrayXd Y   = Y0;
      ArrayXd Y_r = Y0;
      ArrayXd G(dim);
      ArrayXd G_sum(dim);
      double TimeWindow = 0.0;

      for(int i=0;i<N;i++)
      { 
        TimeWindow += h2;
        var_val_r  += var_increment_r;
        for(int l=0;l<dim;l++) G(l) = normcdfinv_as241(U(gen));
        G_sum = G;
        NV_CIR(Y_r, C0_r, C1_r, C2_r, C3_r, G);

        std::pair<double,double> pricedelta_r =
              utility_func_TI(X0, h2, r, TimeWindow, var_increment_r, var_val_r, Y_r, vols_sq_penalised, 
                        mu_mod, cov_mat, d_cov_mat, min_gua_start, max_gua_start, min_rate, max_rate);


        fact_TI = GM_spXmi_discounted_mc(nominals_TI, current_ages_TI, phi_GM, m_GM, b_GM, TimeWindow, r);
        fact_EI = GM_spXmi_discounted_mc(nominals_EI, current_ages_EI, phi_GM, m_GM, b_GM, TimeWindow, r);
        price_TI_tot_r += h2*pricedelta_r.first *fact_TI;
        delta_TI_tot_r += h2*pricedelta_r.second*fact_TI;

        price_EI_tot_r += h2*pricedelta_r.first *fact_EI;
        delta_EI_tot_r += h2*pricedelta_r.second*fact_EI;

        TimeWindow += h2;
        var_val_r += var_increment_r;
        for(int l=0;l<dim;l++) G(l) = normcdfinv_as241(U(gen));
        G_sum += G;
        G_sum /= sqrt(2.0);
        NV_CIR(Y_r, C0_r, C1_r, C2_r, C3_r, G);

        pricedelta_r = 
              utility_func_TI(X0, h2, r, TimeWindow, var_increment_r, var_val_r, Y_r, vols_sq_penalised, 
                        mu_mod, cov_mat, d_cov_mat, min_gua_start, max_gua_start, min_rate, max_rate);

        fact_TI = GM_spXmi_discounted_mc(nominals_TI, current_ages_TI, phi_GM, m_GM, b_GM, TimeWindow, r);
        fact_EI = GM_spXmi_discounted_mc(nominals_EI, current_ages_EI, phi_GM, m_GM, b_GM, TimeWindow, r);

        // rough trajectory
        var_val += var_increment;
        NV_CIR(Y, C0, C1, C2, C3, G_sum);

        std::pair<double,double> pricedelta = 
              utility_func_TI(X0, h, r, TimeWindow, var_increment, var_val, Y, vols_sq_penalised, 
                    mu_mod, cov_mat, d_cov_mat, min_gua_start, max_gua_start, min_rate, max_rate);

        if(i<N-1)
        { 
          price_TI_tot_r +=     h2*pricedelta_r.first *fact_TI; delta_TI_tot_r += h2*pricedelta_r.second*fact_TI;
          price_TI_tot   +=     h *pricedelta.first   *fact_TI; delta_TI_tot   += h *pricedelta.second  *fact_TI;

          price_EI_tot_r +=     h2*pricedelta_r.first *fact_EI; delta_EI_tot_r += h2*pricedelta_r.second*fact_EI;
          price_EI_tot   +=     h *pricedelta.first   *fact_EI; delta_EI_tot   += h *pricedelta.second  *fact_EI;
        }
        else
        { 
          price_TI_tot_r += 0.5*h2*pricedelta_r.first *fact_TI; delta_TI_tot_r += 0.5*h2*pricedelta_r.second*fact_TI;
          price_TI_tot   += 0.5*h *pricedelta.first   *fact_TI; delta_TI_tot   += 0.5*h *pricedelta.second  *fact_TI;

          price_EI_tot_r += 0.5*h2*pricedelta_r.first *fact_EI; delta_EI_tot_r += 0.5*h2*pricedelta_r.second*fact_EI;
          price_EI_tot   += 0.5*h *pricedelta.first   *fact_EI; delta_EI_tot   += 0.5*h *pricedelta.second  *fact_EI;

          double temp_TI         = price_TI_tot_r - price_TI_tot;
          double delta_temp_TI   = delta_TI_tot_r - delta_TI_tot;
          
          
          prices_lvl[1][lvl]     += temp_TI;
          prices_var_lvl[1][lvl] += temp_TI*temp_TI;
          deltas_lvl[1][lvl]     += delta_temp_TI;
          deltas_var_lvl[1][lvl] += delta_temp_TI*delta_temp_TI;

          
          double temp_PE       = (pricedelta_r.first  - pricedelta.first ) * disc_PE;                  
          double delta_temp_PE = (pricedelta_r.second - pricedelta.second) * disc_PE;

          prices_lvl[0][lvl]     += temp_PE;
          prices_var_lvl[0][lvl] += temp_PE*temp_PE;
          deltas_lvl[0][lvl]     += delta_temp_PE;
          deltas_var_lvl[0][lvl] += delta_temp_PE*delta_temp_PE;

          double temp_EI       = perc_prem_death*(price_EI_tot_r - price_EI_tot) 
                                          + (pricedelta_r.first  - pricedelta.first ) * disc_EI;
          double delta_temp_EI = perc_prem_death*(delta_EI_tot_r - delta_EI_tot) 
                                          + (pricedelta_r.second - pricedelta.second) * disc_EI;

          prices_lvl[2][lvl]     += temp_EI;
          prices_var_lvl[2][lvl] += temp_EI*temp_EI;
          deltas_lvl[2][lvl]     += delta_temp_EI;
          deltas_var_lvl[2][lvl] += delta_temp_EI*delta_temp_EI;
        }
              
      }
      
    }

    for(int i=0;i<prices_lvl.size();i++)
    {
      prices_lvl[i][lvl]     /= double(M_lvl);
      prices_var_lvl[i][lvl] = (prices_var_lvl[i][lvl] - M_lvl*prices_lvl[i][lvl]*prices_lvl[i][lvl]) / (M_lvl); //biased
      deltas_lvl[i][lvl]     /= double(M_lvl);
	    deltas_var_lvl[i][lvl] = (deltas_var_lvl[i][lvl] - M_lvl*deltas_lvl[i][lvl]*deltas_lvl[i][lvl]) / (M_lvl); //biased
    }

    price_PE += prices_lvl[0][lvl]; CI_price_PE += prices_var_lvl[0][lvl] / M_lvl ; 
    delta_PE += deltas_lvl[0][lvl]; CI_delta_PE += deltas_var_lvl[0][lvl] / M_lvl ; 

    price_TI += prices_lvl[1][lvl]; CI_price_TI += prices_var_lvl[1][lvl] / M_lvl ; 
    delta_TI += deltas_lvl[1][lvl]; CI_delta_TI += deltas_var_lvl[1][lvl] / M_lvl ; 

    price_EI += prices_lvl[2][lvl]; CI_price_EI += prices_var_lvl[2][lvl] / M_lvl ; 
    delta_EI += deltas_lvl[2][lvl]; CI_delta_EI += deltas_var_lvl[2][lvl] / M_lvl ; 

  }

  CI_price_PE = 1.96 * sqrt( CI_price_PE ); 
  CI_delta_PE = 1.96 * sqrt( CI_delta_PE ); 

  CI_price_TI = 1.96 * sqrt( CI_price_TI ); 
  CI_delta_TI = 1.96 * sqrt( CI_delta_TI );  

  CI_price_EI = 1.96 * sqrt( CI_price_EI );  
  CI_delta_EI = 1.96 * sqrt( CI_delta_EI ); 

}



Output_hedging
PE_hedge_cost_paral(double CRRA_delta, double X0, double r, ArrayXd mu, ArrayXd vols, MatrixXd cov_mat, 
									double alpha_penalty, ArrayXd Y0, ArrayXd beta, ArrayXd kappa, ArrayXd s,
                  double start_age, double phi_GM, double m_GM, double b_GM, 
                  double min_gua, double max_gua, double T, int N, int N_delta, int M, int M_delta)

{
  // different N for number fo times of rebalancing portfolio and computing deltas
  // compute of prices and delta are all in the MMM 

  double price_start, var_price_start, delta_start, var_delta_start;


  PE_min_max_GM_MMM(CRRA_delta, X0, r, mu, vols, cov_mat, alpha_penalty, Y0, beta, kappa, s,
                  start_age, phi_GM, m_GM, b_GM,
                  min_gua, max_gua, T, N_delta, M_delta,
                  price_start, var_price_start, delta_start, var_delta_start);
  
  std::cout << "price in MMM    = " << price_start  << std::endl;
  std::cout << "   price   CI95 = " << 1.96*sqrt(price_start/M_delta)  << std::endl;
  std::cout << "delta at time 0 = " << delta_start  << std::endl;
  std::cout << "   delta   CI95 = " << 1.96*sqrt(var_delta_start/M_delta)  << std::endl;

  ArrayXd bonds(M);
  double bond_start = price_start - delta_start*X0;
  bonds.setConstant(price_start - delta_start*X0);
  ArrayXd fund_shares(M);
  fund_shares.setConstant(delta_start);
  ArrayXd trading_gains(M);
  trading_gains.setConstant(0.0);
  std::vector<double> X_vec(M, 0.0);
  std::vector<double> COST(M, 0.0);

  std::vector<double> no_hedge(M, price_start);
  std::vector<double> static_hedge(M);
  
  #pragma omp parallel num_threads(8) //16 threads possible but bottlenecked by memory bandwith
  { 
    
    VectorXd mu_mod = (mu-r).matrix();
    ArrayXd vols_sq_penalised = alpha_penalty * vols.square();
    MatrixXd d_cov_mat = CRRA_delta * cov_mat;
    double h = T / double(N);
    auto dim = (beta.size()); 
    ArrayXd C0(dim), C1(dim), C2(dim), C3(dim);
    prm_NV_CIR(h, beta, kappa, s, C0, C1, C2, C3); //setting parameters for NV scheme for CIR
    double ratio_N = N / double(N_delta);
    auto& gen = parallel_thread_rng<__LINE__>();
    boost::random::uniform_real_distribution<double> U(0, 1); // uni dist over (0,1)

    double price, var_price, delta, var_delta;
    #pragma omp for
    for(int j=0; j<M; j++)
    { 
      double current_age = start_age;
      double TtoMat = T;
      double  X = X0;
      ArrayXd Y = Y0;
      ArrayXd G(dim);

      ArrayXd e_CI = Y0 * vols_sq_penalised;
      d_cov_mat.diagonal().array() += e_CI;
      Eigen::LLT<Eigen::MatrixXd> llt_solver_start(d_cov_mat);
      VectorXd pi = llt_solver_start.solve(mu_mod);
      d_cov_mat.diagonal().array() -= e_CI;

      
      bool is_alive = true;

      
      for(int i=0; (i<N) & is_alive; i++)
      { 
        
        double var_val  = 0.5 * pi.dot(cov_mat*pi);
        double mean_val = 0.5 * pi.dot(mu_mod);

        for(int l=0;l<dim;l++) G(l) = normcdfinv_as241(U(gen));

        //Carbon intensity evolution
        NV_CIR(Y, C0, C1, C2, C3, G);
        // Computation of portfolio weigths
        e_CI = Y * vols_sq_penalised;
        d_cov_mat.diagonal().array() += e_CI;
        Eigen::LLT<Eigen::MatrixXd> llt_solver(d_cov_mat);
        pi = llt_solver.solve(mu_mod);
        d_cov_mat.diagonal().array() -= e_CI;
        // computation of the mean and variance of the logstock
        var_val  = h * ( var_val  + 0.5 * pi.dot(cov_mat*pi) );
        mean_val = h * ( mean_val + 0.5 * pi.dot(mu_mod) + r) - 0.5 * var_val;
        //mean_val = h * r - 0.5 * var_val; // for risk free scenarios
        //evolution of fund price
        double X2 = X*exp( mean_val + sqrt(var_val)*normcdfinv_as241(U(gen)) );

        //calculating the hedge gains
        trading_gains(j) += (X2 - X) * fund_shares(j);
        //std::cout << "trading_gains(j) = " << trading_gains(j)   << std::endl;

        //updating the value of the fund
        X = X2;
        
        // updating status of the Policyowner
        double survival_proba = GM_survival_proba(current_age, phi_GM, m_GM, b_GM, h);
        //std::cout << "survival_proba = " << survival_proba    << std::endl;
        if(U(gen) > survival_proba) is_alive = false;
 


        if(is_alive)
        {
          TtoMat      -= h;
          current_age += h;
          if(i<N-1)
          {
            PE_min_max_GM_MMM(CRRA_delta, X, r, mu, vols, cov_mat, alpha_penalty, Y, beta, kappa, s,
                          current_age, phi_GM, m_GM, b_GM,
                          min_gua, max_gua, TtoMat, std::ceil((N-i-1)/ratio_N), M_delta,
                          price, var_price, delta, var_delta);
            //std::cout << "\ndelta = " << delta << std::endl;
            //std::cout << "\nfund_shares:\n" << fund_shares << std::endl;
            COST[j]       += price - (bonds(j)*exp(r*h) + fund_shares(j)*X);
            bonds(j)       = price - delta*X;
            fund_shares(j) = delta;
          }
          else
          {
            if      (X < min_gua) price = min_gua;
            else if (X > max_gua) price = max_gua;
            else                  price = X;
            COST[j]       += price - (bonds(j)*exp(r*h) + fund_shares(j)*X);
            no_hedge[j]     = price -no_hedge[j]*exp(r*(T-TtoMat));
            static_hedge[j] = price -bond_start *exp(r*(T-TtoMat)) - delta_start*X;
          }
        }
        else
        { // price = 0, delta = 0
          COST[j]       -= bonds(j)*exp(r*h) + fund_shares(j)*X;
          bonds(j)       = 0.0; //price - delta*X;
          fund_shares(j) = 0.0; //delta;
          no_hedge[j]     = -no_hedge[j]*exp(r*(T-TtoMat+h));
          static_hedge[j] = -bond_start *exp(r*(T-TtoMat)) - delta_start*X;
        }

      }
      X_vec[j] = X;
    }
  } //end parallel zone
  //std::cout << "\ntrading_gains:\n" << trading_gains << std::endl;
  //std::cout << "\nfund_shares:\n" << fund_shares << std::endl;
  double mean_hedging_gains = trading_gains.mean();
  double std_dev_hedging_gains = sqrt( ((trading_gains -mean_hedging_gains).square()).mean() );
  //std::cout << "expected trading gains = " << mean_hedging_gains    << std::endl;
  //std::cout << "CI95                   = " << 1.96*std_dev_hedging_gains/sqrt(M) << std::endl;
  //std::cout << "std_dev trading gains  = " << std_dev_hedging_gains << std::endl;

  double mean_cost = mean(COST);
  std::pair<double,double> output_var = empirical_var_CI95(COST);
  std::cout << "expected value of cost = " << mean_cost    << std::endl;
  std::cout << "CI95  expected value   = " << 1.96*sqrt(output_var.first/M) << std::endl;
  std::cout << "     var cost = " << output_var.first << std::endl;
  std::cout << "CI95 var cost = " << output_var.second << std::endl;

  std::vector<double> Trading_gains(trading_gains.size(),0.0);
  for(int i=0;i<trading_gains.size();i++)
    Trading_gains[i] = trading_gains(i);

  Output_hedging output;
  output.no_hedge = no_hedge;
  output.static_hedge = static_hedge;
  output.trading_gains = Trading_gains;
  output.hedging_cost = COST;
  output.start_price = price_start;
  output.X = X_vec;
  return output;
}




Output_hedging
TI_hedge_cost_paral(double CRRA_delta, double X0, double r, ArrayXd mu, ArrayXd vols, MatrixXd cov_mat, 
									double alpha_penalty, ArrayXd Y0, ArrayXd beta, ArrayXd kappa, ArrayXd s,
                  double start_age, double phi_GM, double m_GM, double b_GM, 
                  double min_rate, double max_rate, double T, int N, int N_delta, int M, int M_delta)
{
  // different N for number of times of rebalancing portfolio and computing deltas
  // compute of prices and delta are all in the MMM
  // In output 3 strategies:
  // 1) no hedging investment on the cash account
  // 2) static hedge
  // 3) continous hedging

  double price_start, var_price_start, delta_start, var_delta_start;


  TI_min_max_GM_MMM(CRRA_delta, X0, r, mu, vols, cov_mat, alpha_penalty, Y0, beta, kappa, s,
                  start_age, phi_GM, m_GM, b_GM,
                  X0, X0, min_rate, max_rate, T, N_delta, M_delta,
                  price_start, var_price_start, delta_start, var_delta_start);
  
  std::cout << "price in MMM    = " << price_start  << std::endl;
  std::cout << "   price   CI95 = " << 1.96*sqrt(price_start/M_delta)  << std::endl;
  std::cout << "delta at time 0 = " << delta_start  << std::endl;
  std::cout << "   delta   CI95 = " << 1.96*sqrt(var_delta_start/M_delta)  << std::endl;

  ArrayXd bonds(M);
  double bond_start = price_start - delta_start*X0;
  bonds.setConstant(  price_start - delta_start*X0);
  ArrayXd fund_shares(M);
  fund_shares.setConstant(delta_start);
  std::vector<double> trading_gains(M, 0.0);
  std::vector<double> X_vec(M, 0.0);
  std::vector<double> COST(M, 0.0);

  std::vector<double> no_hedge(M, price_start);
  std::vector<double> static_hedge(M);
  
  #pragma omp parallel num_threads(8) //16 threads possible but bottlenecked by memory bandwith
  { 
    
    VectorXd mu_mod = (mu-r).matrix();
    ArrayXd vols_sq_penalised = alpha_penalty * vols.square();
    MatrixXd d_cov_mat = CRRA_delta * cov_mat;
    double h = T / double(N);
    auto dim = (beta.size()); 
    ArrayXd C0(dim), C1(dim), C2(dim), C3(dim);
    prm_NV_CIR(h, beta, kappa, s, C0, C1, C2, C3); //setting parameters for NV scheme for CIR
    double ratio_N = N / double(N_delta);
    auto& gen = parallel_thread_rng<__LINE__>();
    boost::random::uniform_real_distribution<double> U(0, 1); // uni dist over (0,1)

    double price, var_price, delta, var_delta;
    #pragma omp for
    for(int j=0; j<M; j++)
    { 
      double current_age = start_age;
      double TtoMat = T;
      double  X = X0;
      ArrayXd Y = Y0;
      ArrayXd G(dim);

      ArrayXd e_CI = Y0 * vols_sq_penalised;
      d_cov_mat.diagonal().array() += e_CI;
      Eigen::LLT<Eigen::MatrixXd> llt_solver_start(d_cov_mat);
      VectorXd pi = llt_solver_start.solve(mu_mod);
      d_cov_mat.diagonal().array() -= e_CI;

      
      bool is_alive = true;

      //int i = 0;
      //while( (i<N) & is_alive)
      for(int i=0;(i<N) & is_alive;i++)
      { 
        double var_val  = 0.5 * pi.dot(cov_mat*pi);
        double mean_val = 0.5 * pi.dot(mu_mod);

        for(int l=0;l<dim;l++) G(l) = normcdfinv_as241(U(gen));

        //Carbon intensity evolution
        NV_CIR(Y, C0, C1, C2, C3, G);
        // Computation of portfolio weigths
        e_CI = Y * vols_sq_penalised; //carbon penalty = carbon transation cost
        d_cov_mat.diagonal().array() += e_CI;
        Eigen::LLT<Eigen::MatrixXd> llt_solver(d_cov_mat);
        pi = llt_solver.solve(mu_mod);
        d_cov_mat.diagonal().array() -= e_CI;
        // computation of the mean and variance of the logstock
        var_val  = h * ( var_val  + 0.5 * pi.dot(cov_mat*pi) );
        mean_val = h * ( mean_val + 0.5 * pi.dot(mu_mod) + r ) - 0.5 * var_val;
        //evolution of fund price
        double X2 = X*exp( mean_val + sqrt(var_val)*normcdfinv_as241(U(gen)) );

        //calculating the hedge gains
        trading_gains[j] += (X2 - X) * fund_shares(j);
        //std::cout << "trading_gains[j] = " << trading_gains[j]   << std::endl;

        //updating the value of the fund
        X = X2;
        
        // updating status of the Policyowner
        double survival_proba = GM_survival_proba(current_age, phi_GM, m_GM, b_GM, h);
        //std::cout << "survival_proba = " << survival_proba    << std::endl;
        if(U(gen) > survival_proba) is_alive = false;

        TtoMat      -= h;
        current_age += h;
        double min_gua = X0*exp(min_rate*(T-TtoMat));
        double max_gua = X0*exp(max_rate*(T-TtoMat));
        if(is_alive)
        {
          if(i<N-1)
          {
            TI_min_max_GM_MMM(CRRA_delta, X, r, mu, vols, cov_mat, alpha_penalty, Y, beta, kappa, s,
                          current_age, phi_GM, m_GM, b_GM,
                          min_gua, max_gua, min_rate, max_rate, 
                          TtoMat, std::ceil((N-i-1)/ratio_N), M_delta,
                          price, var_price, delta, var_delta);
            //std::cout << "std::ceil((N-i-1)/ratio_N) =" << std::ceil((N-i-1)/ratio_N) << std::endl;
            //std::cout << "\ndelta = " << delta << std::endl;
            //std::cout << "\nfund_shares:\n" << fund_shares << std::endl;
            COST[j]       += price - (bonds(j)*exp(r*h) + fund_shares(j)*X);
            bonds(j)       = price - delta*X;
            fund_shares(j) = delta;
          }
          else
          { // at expiration T payoff=0 (price=0)  TO VERIFY
            COST[j]         -= (bonds(j)*exp(r*h) + fund_shares(j)*X);
            no_hedge[j]      = -price_start*exp(r*(T-TtoMat));
            static_hedge[j]  = -bond_start *exp(r*(T-TtoMat)) - delta_start*X;
          }
        }
        else
        { // policy owner death, then payment
          if      (X < min_gua) price = min_gua;
          else if (X > max_gua) price = max_gua;
          else                  price = X;
          COST[j]       += price - (bonds(j)*exp(r*h) + fund_shares(j)*X);
          bonds(j)       = 0.0; //price - delta*X;
          fund_shares(j) = 0.0; //delta;
          no_hedge[j]     = price - price_start*exp(r*(T-TtoMat));
          static_hedge[j] = price - bond_start *exp(r*(T-TtoMat)) - delta_start*X;
        }

        //i++; //increment the counter i
      }
      X_vec[j] = X; //X a time Tau
    }
  } //end parallel zone

  double mean_cost = mean(COST);
  std::pair<double,double> output_var = empirical_var_CI95(COST);
  std::cout << "expected value of cost = " << mean_cost    << std::endl;
  std::cout << "CI95  expected value   = " << 1.96*sqrt(output_var.first/M) << std::endl;
  std::cout << "     var cost = " << output_var.first << std::endl;
  std::cout << "CI95 var cost = " << output_var.second << std::endl;


  Output_hedging output;
  output.no_hedge = no_hedge;
  output.static_hedge = static_hedge;
  output.trading_gains = trading_gains;
  output.hedging_cost = COST;
  output.start_price = price_start;
  output.X = X_vec;
  return output;
}




//adattata per min_max_rate e perc_prem_death
Output_hedging
EI_hedge_cost_paral(double CRRA_delta, double X0, double r, ArrayXd mu, ArrayXd vols, MatrixXd cov_mat, 
									double alpha_penalty, ArrayXd Y0, ArrayXd beta, ArrayXd kappa, ArrayXd s,
                  double start_age, double phi_GM, double m_GM, double b_GM, 
                  double min_rate, double max_rate, double perc_prem_death,
                  double T, int N, int N_delta, int M, int M_delta)
{
  // different N for number of times of rebalancing portfolio and computing deltas
  // compute of prices and delta are all in the MMM
  // In output 3 strategies:
  // 1) no hedging investment on the cash account
  // 2) static hedge
  // 3) continous hedging

  double price_start, var_price_start, delta_start, var_delta_start;


  EI_min_max_GM_MMM(CRRA_delta, X0, r, mu, vols, cov_mat, alpha_penalty, Y0, beta, kappa, s,
                  start_age, phi_GM, m_GM, b_GM,
                  X0, X0, min_rate, max_rate, perc_prem_death, T, N_delta, M_delta,
                  price_start, var_price_start, delta_start, var_delta_start);
  
  std::cout << "price in MMM    = " << price_start  << std::endl;
  std::cout << "delta at time 0 = " << delta_start  << std::endl;
  std::cout << "           CI95 = " << 1.96*sqrt(var_delta_start/M_delta)  << std::endl;

  ArrayXd bonds(M);
  double bond_start = price_start - delta_start*X0;
  bonds.setConstant(price_start - delta_start*X0);
  ArrayXd fund_shares(M);
  fund_shares.setConstant(delta_start);
  ArrayXd trading_gains(M);
  trading_gains.setConstant(0.0);
  std::vector<double> X_vec(M, 0.0);
  std::vector<double> COST(M, 0.0);

  std::vector<double> no_hedge(M, price_start);
  std::vector<double> static_hedge(M);

  
  #pragma omp parallel num_threads(8) //16 threads possible but bottlenecked by memory bandwith
  { 
    
    VectorXd mu_mod = (mu-r).matrix();
    ArrayXd vols_sq_penalised = alpha_penalty * vols.square();
    MatrixXd d_cov_mat = CRRA_delta * cov_mat;
    double h = T / double(N);
    auto dim = (beta.size()); 
    ArrayXd C0(dim), C1(dim), C2(dim), C3(dim);
    prm_NV_CIR(h, beta, kappa, s, C0, C1, C2, C3); //setting parameters for NV scheme for CIR
    double ratio_N = N / double(N_delta);
    auto& gen = parallel_thread_rng<__LINE__>();
    boost::random::uniform_real_distribution<double> U(0, 1); // uni dist over (0,1)

    double price, var_price, delta, var_delta;
    #pragma omp for
    for(int j=0; j<M; j++)
    { 
      double current_age = start_age;
      double TtoMat = T;
      double  X = X0;
      ArrayXd Y = Y0;
      ArrayXd G(dim);

      ArrayXd e_CI = Y0 * vols_sq_penalised;
      d_cov_mat.diagonal().array() += e_CI;
      Eigen::LLT<Eigen::MatrixXd> llt_solver_start(d_cov_mat);
      VectorXd pi = llt_solver_start.solve(mu_mod);
      d_cov_mat.diagonal().array() -= e_CI;

      
      bool is_alive = true;

      //int i = 0;
      //while( (i<N) & is_alive)
      for(int i=0;(i<N) & is_alive;i++)
      { 
        
        double var_val  = 0.5 * pi.dot(cov_mat*pi);
        double mean_val = 0.5 * pi.dot(mu_mod);

        for(int l=0;l<dim;l++) G(l) = normcdfinv_as241(U(gen));

        //Carbon intensity evolution
        NV_CIR(Y, C0, C1, C2, C3, G);
        // Computation of portfolio weigths
        e_CI = Y * vols_sq_penalised; //carbon penalty = carbon transation cost
        d_cov_mat.diagonal().array() += e_CI;
        Eigen::LLT<Eigen::MatrixXd> llt_solver(d_cov_mat);
        pi = llt_solver.solve(mu_mod);
        d_cov_mat.diagonal().array() -= e_CI;
        // computation of the mean and variance of the logstock
        var_val  = h * ( var_val  + 0.5 * pi.dot(cov_mat*pi) );
        mean_val = h * ( mean_val + 0.5 * pi.dot(mu_mod) + r ) - 0.5 * var_val;
        //evolution of fund price
        double X2 = X*exp( mean_val + sqrt(var_val)*normcdfinv_as241(U(gen)) );

        //calculating the hedge gains
        trading_gains(j) += (X2 - X) * fund_shares(j);
        //std::cout << "trading_gains(j) = " << trading_gains(j)   << std::endl;

        //updating the value of the fund
        X = X2;
        
        // updating status of the Policyowner
        double survival_proba = GM_survival_proba(current_age, phi_GM, m_GM, b_GM, h);
        //std::cout << "survival_proba = " << survival_proba    << std::endl;
        if(U(gen) > survival_proba) is_alive = false;

        TtoMat      -= h;
        current_age += h;
        double min_gua = X0*exp(min_rate*(T-TtoMat));
        double max_gua = X0*exp(max_rate*(T-TtoMat));
        if(is_alive)
        { 
          if(i<N-1)
          { 
            EI_min_max_GM_MMM(CRRA_delta, X, r, mu, vols, cov_mat, alpha_penalty, Y, beta, kappa, s,
                          current_age, phi_GM, m_GM, b_GM,
                          min_gua, max_gua, min_rate, max_rate, perc_prem_death, 
                          TtoMat, std::ceil((N-i-1)/ratio_N), M_delta,
                          price, var_price, delta, var_delta);
            //std::cout << "\ndelta = " << delta << std::endl;
            //std::cout << "\nfund_shares:\n" << fund_shares << std::endl;
            COST[j]       += price - (bonds(j)*exp(r*h) + fund_shares(j)*X);
            bonds(j)       = price - delta*X;
            fund_shares(j) = delta;
          }
          else
          { // at expiration T
            if      (X < min_gua) price = min_gua;
            else if (X > max_gua) price = max_gua;
            else                  price = X;
            COST[j]       += price - (bonds(j)*exp(r*h) + fund_shares(j)*X);
            bonds(j)       = 0.0; //price - delta*X;
            fund_shares(j) = 0.0; //delta;
            no_hedge[j]     = price - price_start*exp(r*(T-TtoMat));
            static_hedge[j] = price - bond_start *exp(r*(T-TtoMat)) - delta_start*X;
          }
        }
        else
        { // policy owner death, then payment
          if      (X < min_gua) price = min_gua;
          else if (X > max_gua) price = max_gua;
          else                  price = X;
          COST[j]       += perc_prem_death*price - (bonds(j)*exp(r*h) + fund_shares(j)*X);
          bonds(j)       = 0.0; //price - delta*X;
          fund_shares(j) = 0.0; //delta;

          no_hedge[j]     = perc_prem_death*price -no_hedge[j]*exp(r*(T-TtoMat));
          static_hedge[j] = perc_prem_death*price -bond_start *exp(r*(T-TtoMat)) - delta_start*X;
        }

        //i++; //increment the counter i
      }
      X_vec[j] = X; //X a time Tau
    }
  } //end parallel zone
  //std::cout << "\ntrading_gains:\n" << trading_gains << std::endl;
  //std::cout << "\nfund_shares:\n" << fund_shares << std::endl;
  double mean_hedging_gains = trading_gains.mean();
  double std_dev_hedging_gains = sqrt( ((trading_gains -mean_hedging_gains).square()).mean() );
  //std::cout << "expected trading gains = " << mean_hedging_gains    << std::endl;
  //std::cout << "CI95                   = " << 1.96*std_dev_hedging_gains/sqrt(M) << std::endl;
  //std::cout << "std_dev trading gains  = " << std_dev_hedging_gains << std::endl;

  double mean_cost = mean(COST);
  std::pair<double,double> output_var = empirical_var_CI95(COST);
  std::cout << "expected value of cost = " << mean_cost    << std::endl;
  std::cout << "CI95  expected value   = " << 1.96*sqrt(output_var.first/M) << std::endl;
  std::cout << "     var cost = " << output_var.first << std::endl;
  std::cout << "CI95 var cost = " << output_var.second << std::endl;

  std::vector<double> Trading_gains(trading_gains.size(),0.0);
  for(int i=0;i<trading_gains.size();i++)
    Trading_gains[i] = trading_gains(i);

  Output_hedging output;
  output.no_hedge = no_hedge;
  output.static_hedge = static_hedge;
  output.trading_gains = Trading_gains;
  output.hedging_cost = COST;
  output.start_price = price_start;
  output.X = X_vec;
  return output;
}




Output_hedging
PETIEI_hedge_cost_paral(double CRRA_delta, ArrayXd X_PE, ArrayXd X_TI, ArrayXd X_EI, 
                      double r, ArrayXd mu, ArrayXd vols, MatrixXd cov_mat, 
									    double alpha_penalty, ArrayXd Y0, ArrayXd beta, ArrayXd kappa, ArrayXd s,
                      double start_age, double phi_GM, double m_GM, double b_GM, 
                      double min_rate, double max_rate, double perc_prem_death, 
                      double T, int N, int N_delta, int M, int M_delta)
{
  // different N for number of times of rebalancing portfolio and computing deltas
  // compute of prices and delta are all in the MMM
  // X_PE contains the nominal value of fund's units that each PE policy holder has
  // X_TI contains the nominal value of fund's units that each TI policy holder has
  // In output 3 strategies:
  // 1) no hedging investment on the cash account
  // 2) static hedge
  // 3) continous hedging

  double price_start_PE, var_price_start_PE, delta_start_PE, var_delta_start_PE;
  double price_start_TI, var_price_start_TI, delta_start_TI, var_delta_start_TI;
  double price_start_EI, var_price_start_EI, delta_start_EI, var_delta_start_EI;

  PETIEI_min_max_GM_MMM(CRRA_delta, 1.0, r, mu, vols, cov_mat, alpha_penalty, Y0, beta, kappa, s,
                  start_age, phi_GM, m_GM, b_GM,
                  1.0, 1.0, min_rate, max_rate, perc_prem_death, T, N_delta, M_delta,
                  price_start_PE, var_price_start_PE, delta_start_PE, var_delta_start_PE,
                  price_start_TI, var_price_start_TI, delta_start_TI, var_delta_start_TI,
                  price_start_EI, var_price_start_EI, delta_start_EI, var_delta_start_EI);
  
  std::cout << "price_PE in MMM = " << price_start_PE  << std::endl;
  std::cout << "           CI95 = " << 1.96*sqrt(var_price_start_PE/M_delta)  << std::endl;
  std::cout << "price_TI in MMM = " << price_start_TI  << std::endl;
  std::cout << "           CI95 = " << 1.96*sqrt(var_price_start_TI/M_delta)  << std::endl;
  std::cout << "price_EI in MMM = " << price_start_EI  << std::endl;
  std::cout << "           CI95 = " << 1.96*sqrt(var_price_start_EI/M_delta)  << std::endl;
  std::cout << "delta_PE at time 0 = " << delta_start_PE  << std::endl;
  std::cout << "              CI95 = " << 1.96*sqrt(var_delta_start_PE/M_delta)  << std::endl;
  std::cout << "delta_TI at time 0 = " << delta_start_TI  << std::endl;
  std::cout << "              CI95 = " << 1.96*sqrt(var_delta_start_TI/M_delta)  << std::endl;
  std::cout << "delta_EI at time 0 = " << delta_start_EI  << std::endl;
  std::cout << "              CI95 = " << 1.96*sqrt(var_delta_start_EI/M_delta)  << std::endl;

  double start_nominal_PE = X_PE.sum();
  double start_nominal_TI = X_TI.sum();
  double start_nominal_EI = X_EI.sum();

  double price_start = start_nominal_PE*price_start_PE + start_nominal_TI*price_start_TI
                                              + start_nominal_EI*price_start_EI;

  double delta_start = start_nominal_PE*delta_start_PE + start_nominal_TI*delta_start_TI
                                              + start_nominal_EI*delta_start_EI;

  std::cout << "price_start = " << price_start   << std::endl;
  std::cout << "delta_start = " << delta_start   << std::endl;

  ArrayXd bonds(M);
  double bond_start = price_start  - delta_start; //X0=1.0
  bonds.setConstant(bond_start);
  ArrayXd fund_shares(M);
  fund_shares.setConstant(delta_start);
  std::vector<double> trading_gains(M, 0.0);
  std::vector<double> X_vec(M, 0.0);
  std::vector<double> COST(M, 0.0);

  std::vector<double> no_hedge(M, price_start);
  std::vector<double> static_hedge(M);
  std::vector<double> bonds_static_hedge(M,bond_start);

  
  #pragma omp parallel num_threads(8) //16 threads possible but bottlenecked by memory bandwith
  { 
    double stock_position_static_hedge = delta_start;
    VectorXd mu_mod = (mu-r).matrix();
    ArrayXd vols_sq_penalised = alpha_penalty * vols.square();
    MatrixXd d_cov_mat = CRRA_delta * cov_mat;
    double h = T / double(N);
    auto dim = beta.size(); 
    ArrayXd C0(dim), C1(dim), C2(dim), C3(dim);
    prm_NV_CIR(h, beta, kappa, s, C0, C1, C2, C3); //setting parameters for NV scheme for CIR
    double ratio_N = N / double(N_delta);
    auto& gen = parallel_thread_rng<__LINE__>();
    boost::random::uniform_real_distribution<double> U(0, 1); // uni dist over (0,1)

    double price_PE, var_price_PE, delta_PE, var_delta_PE;
    double price_TI, var_price_TI, delta_TI, var_delta_TI;
    double price_EI, var_price_EI, delta_EI, var_delta_EI;
    #pragma omp for
    for(int j=0; j<M; j++)
    { 
      double current_age = start_age;
      double TtoMat = T;
      double  X = 1.0; //we start the fund as 1 to have the dynamic of 1 unit of fund
      ArrayXd Y = Y0;
      ArrayXd G(dim);

      ArrayXd e_CI = Y0 * vols_sq_penalised;
      d_cov_mat.diagonal().array() += e_CI;
      Eigen::LLT<Eigen::MatrixXd> llt_solver_start(d_cov_mat);
      VectorXd pi = llt_solver_start.solve(mu_mod);
      d_cov_mat.diagonal().array() -= e_CI;

      auto n_PE = X_PE.size(); // if this not works we pass with double and we use 0,1
      ArrayXb is_alive_PE(n_PE);
      is_alive_PE.setConstant(1);

      auto n_TI = X_TI.size();
      ArrayXb is_alive_TI(n_TI);
      is_alive_TI.setConstant(1);

      auto n_EI = X_EI.size();
      ArrayXb is_alive_EI(n_EI);
      is_alive_EI.setConstant(1);

      std::stack<int> TI_toPay;

      std::stack<int> EI_toPay;

      //int i = 0;
      //while( (i<N) & is_alive)
      for(int i=0; i<N; i++)
      { 
        double var_val  = 0.5 * pi.dot(cov_mat*pi);
        double mean_val = 0.5 * pi.dot(mu_mod);

        for(int l=0;l<dim;l++) G(l) = normcdfinv_as241(U(gen));

        //Carbon intensity evolution
        NV_CIR(Y, C0, C1, C2, C3, G);
        // Computation of portfolio weigths
        e_CI = Y * vols_sq_penalised; //carbon penalty = carbon transation cost
        d_cov_mat.diagonal().array() += e_CI;
        Eigen::LLT<Eigen::MatrixXd> llt_solver(d_cov_mat);
        pi = llt_solver.solve(mu_mod);
        d_cov_mat.diagonal().array() -= e_CI;
        // computation of the mean and variance of the logstock
        var_val  = h * ( var_val  + 0.5 * pi.dot(cov_mat*pi) );
        mean_val = h * ( mean_val + 0.5 * pi.dot(mu_mod) + r ) - 0.5 * var_val;
        //evolution of fund price
        double X2 = X*exp( mean_val + sqrt(var_val)*normcdfinv_as241(U(gen)) );

        //calculating the hedge gains
        trading_gains[j] += (X2 - X) * fund_shares(j);
        //std::cout << "trading_gains[j] = " << trading_gains[j]   << std::endl;

        //updating the value of the fund
        X = X2;
        
        // updating status of the Policyholders
        double survival_proba = GM_survival_proba(current_age, phi_GM, m_GM, b_GM, h);
        //std::cout << "survival_proba = " << survival_proba    << std::endl;
        //std::cout << "i = " << i   << std::endl;
        //std::cout << "ALIVE_PE = " << (is_alive_PE.template cast<double>()).sum()   << std::endl;
        //std::cout << "ALIVE_TI = " << (is_alive_TI.template cast<double>()).sum()   << std::endl;
        //std::cout << "NOMIN_PE = " << (X_PE * (is_alive_PE.template cast<double>())).sum()   << std::endl;
        //std::cout << "NOMIN_TI = " << (X_TI * (is_alive_TI.template cast<double>())).sum()   << std::endl;
        for(int l=0; l<is_alive_PE.size(); l++)
        {
          if(is_alive_PE(l)==true)
          {
            if(U(gen) > survival_proba) is_alive_PE(l) = false;
          }
        }
        for(int l=0;l<is_alive_TI.size();l++)
        {
          if(is_alive_TI(l)==true)
          {
            if(U(gen) > survival_proba)
            { 
              is_alive_TI(l) = false;
              TI_toPay.push(l); // i add the i-th ph to the list to be payed
            }
          }
        }

        for(int l=0;l<is_alive_EI.size();l++)
        {
          if(is_alive_EI(l)==true)
          {
            if(U(gen) > survival_proba)
            { 
              is_alive_EI(l) = false;
              EI_toPay.push(l); // i add the i-th ph to the list to be payed
            }
          }
        }

        TtoMat      -= h;
        current_age += h;
        double min_gua = exp(min_rate*(T-TtoMat));
        double max_gua = exp(max_rate*(T-TtoMat));

        // This part is the same if if(i<N-1) and if(i==N-1) so is outside

        double PRICE_DEAD = 0.0;
        double market_payoff;
        if      (X < min_gua) market_payoff = min_gua;
        else if (X > max_gua) market_payoff = max_gua;
        else                  market_payoff = X;
        while (!TI_toPay.empty()) { // Paying all the TI policy holder dead
          //std::cout << "Ti polizza da liquidare N = " << TI_toPay.top() << std::endl;
          PRICE_DEAD += X_TI(TI_toPay.top()); // Pay the last client
          TI_toPay.pop(); // Remove the last client from the list "toPay"
        }
        while (!EI_toPay.empty()) { // Paying all the TI policy holder dead
          //std::cout << "Ei polizza da liquidare N = " << EI_toPay.top() << std::endl;
          PRICE_DEAD += X_EI(EI_toPay.top())*perc_prem_death; // Pay the last client
          EI_toPay.pop(); // Remove the last client from the list "toPay"
        }
        PRICE_DEAD *=market_payoff;

        if(i<N-1)
        { 
          PETIEI_min_max_GM_MMM(CRRA_delta, X, r, mu, vols, cov_mat, alpha_penalty, Y, beta, kappa, s,
                        current_age, phi_GM, m_GM, b_GM,
                        min_gua, max_gua, min_rate, max_rate, perc_prem_death, 
                        TtoMat, std::ceil((N-i-1)/ratio_N), M_delta,
                        price_PE, var_price_PE, delta_PE, var_delta_PE,
                        price_TI, var_price_TI, delta_TI, var_delta_TI,
                        price_EI, var_price_EI, delta_EI, var_delta_EI);
          
          double nominal_PE = (X_PE * (is_alive_PE.template cast<double>())).sum();
          double nominal_TI = (X_TI * (is_alive_TI.template cast<double>())).sum();
          double nominal_EI = (X_EI * (is_alive_EI.template cast<double>())).sum();
          double PRICE_ALIVE = nominal_PE*price_PE + nominal_TI*price_TI + nominal_EI*price_EI;
          
          COST[j]       += PRICE_ALIVE + PRICE_DEAD - (bonds(j)*exp(r*h) + fund_shares(j)*X);
          fund_shares(j) = nominal_PE*delta_PE + nominal_TI*delta_TI + nominal_EI*delta_EI;
          bonds(j)       = PRICE_ALIVE - fund_shares(j)*X;
          
          no_hedge[j]    = no_hedge[j]*exp(r*h) - PRICE_DEAD;

          static_hedge[j] = stock_position_static_hedge*X + bonds_static_hedge[j]*exp(r*h) 
                            - PRICE_DEAD;
          stock_position_static_hedge = nominal_PE*delta_start_PE + nominal_TI*delta_start_TI
                                                        + nominal_EI*delta_start_EI;     
          bonds_static_hedge[j] = static_hedge[j] - stock_position_static_hedge*X;
          //std::cout << "PRICE_ALIVE = " << PRICE_ALIVE << std::endl;
          //std::cout << "PRICE_DEAD  = " << PRICE_DEAD << std::endl;
          //std::cout << "COST =" << COST[j] << std::endl;
        }
        else
        { // at expiration T we need to pay the TI/EI dead in [T-h,T] and the PE/EI ALIVE at T

          double nominal_alive_to_pay = (X_PE * (is_alive_PE.template cast<double>())).sum()
                                      + (X_EI * (is_alive_EI.template cast<double>())).sum();
          COST[j]       += PRICE_DEAD + nominal_alive_to_pay*market_payoff - (bonds(j)*exp(r*h) + fund_shares(j)*X);
          no_hedge[j]    = PRICE_DEAD + nominal_alive_to_pay*market_payoff - no_hedge[j]*exp(r*h);


          static_hedge[j] = stock_position_static_hedge*X + bonds_static_hedge[j]*exp(r*h) 
                            - PRICE_DEAD - nominal_alive_to_pay*market_payoff;
          static_hedge[j] = -static_hedge[j];
          bonds(j)       = 0.0; //price - delta*X;
          fund_shares(j) = 0.0; //delta;
        }

      }
      X_vec[j] = X; //X a time T (maturity)

    }
  } //end parallel zone


  double mean_cost = mean(COST);
  std::pair<double,double> output_var = empirical_var_CI95(COST);
  std::cout << "expected value of cost = " << mean_cost    << std::endl;
  std::cout << "CI95  expected value   = " << 1.96*sqrt(output_var.first/M) << std::endl;
  std::cout << "     var cost = " << output_var.first << std::endl;
  std::cout << "CI95 var cost = " << output_var.second << std::endl;


  Output_hedging output;
  output.no_hedge = no_hedge;
  output.static_hedge = static_hedge;
  output.trading_gains = trading_gains;
  output.hedging_cost = COST;
  output.start_price = price_start_PE+price_start_TI+price_start_EI;
  output.X = X_vec;
  return output;
}


Output_hedging
PETIEI_hedge_cost_paral_MLMC(double CRRA_delta, ArrayXd X_PE, ArrayXd X_TI, ArrayXd X_EI, 
                      double r, ArrayXd mu, ArrayXd vols, MatrixXd cov_mat, 
									    double alpha_penalty, ArrayXd Y0, ArrayXd beta, ArrayXd kappa, ArrayXd s,
                      double start_age, double phi_GM, double m_GM, double b_GM, 
                      double min_rate, double max_rate, double perc_prem_death, 
                      double T, int N, int N_delta_lvl, int M, int M_delta)
{
  // different N for number of times of rebalancing portfolio and computing deltas
  // compute of prices and delta are all in the MMM
  // X_PE contains the nominal value of fund's units that each PE policy holder has
  // X_TI contains the nominal value of fund's units that each TI policy holder has
  // In output 3 strategies:
  // 1) no hedging investment on the cash account
  // 2) static hedge
  // 3) continous hedging

  double price_start_PE, CI_price_start_PE, delta_start_PE, CI_delta_start_PE;
  double price_start_TI, CI_price_start_TI, delta_start_TI, CI_delta_start_TI;
  double price_start_EI, CI_price_start_EI, delta_start_EI, CI_delta_start_EI;

  PETIEI_min_max_GM_MMM_MLMC(CRRA_delta, 1.0, r, mu, vols, cov_mat, alpha_penalty, Y0, beta, kappa, s,
                  start_age, phi_GM, m_GM, b_GM,
                  1.0, 1.0, min_rate, max_rate, perc_prem_death, T, N_delta_lvl, M_delta,
                  price_start_PE, CI_price_start_PE, delta_start_PE, CI_delta_start_PE,
                  price_start_TI, CI_price_start_TI, delta_start_TI, CI_delta_start_TI,
                  price_start_EI, CI_price_start_EI, delta_start_EI, CI_delta_start_EI);
  
  std::cout << "price_PE in MMM = " << price_start_PE  << std::endl;
  std::cout << "           CI95 = " << CI_price_start_PE  << std::endl;
  std::cout << "price_TI in MMM = " << price_start_TI  << std::endl;
  std::cout << "           CI95 = " << CI_price_start_TI  << std::endl;
  std::cout << "price_EI in MMM = " << price_start_EI  << std::endl;
  std::cout << "           CI95 = " << CI_price_start_EI  << std::endl;
  std::cout << "delta_PE at time 0 = " << delta_start_PE  << std::endl;
  std::cout << "              CI95 = " << CI_delta_start_PE  << std::endl;
  std::cout << "delta_TI at time 0 = " << delta_start_TI  << std::endl;
  std::cout << "              CI95 = " << CI_delta_start_TI  << std::endl;
  std::cout << "delta_EI at time 0 = " << delta_start_EI  << std::endl;
  std::cout << "              CI95 = " << CI_delta_start_EI  << std::endl;

  double start_nominal_PE = X_PE.sum();
  double start_nominal_TI = X_TI.sum();
  double start_nominal_EI = X_EI.sum();

  double price_start = start_nominal_PE*price_start_PE + start_nominal_TI*price_start_TI
                                              + start_nominal_EI*price_start_EI;

  double delta_start = start_nominal_PE*delta_start_PE + start_nominal_TI*delta_start_TI
                                              + start_nominal_EI*delta_start_EI;

  std::cout << "price_start = " << price_start   << std::endl;
  std::cout << "delta_start = " << delta_start   << std::endl;

  ArrayXd bonds(M);
  double bond_start = price_start  - delta_start; //X0=1.0
  bonds.setConstant(bond_start);
  ArrayXd fund_shares(M);
  fund_shares.setConstant(delta_start);
  std::vector<double> trading_gains(M, 0.0);
  std::vector<double> X_vec(M, 0.0);
  std::vector<double> COST(M, 0.0);

  std::vector<double> no_hedge(M, price_start);
  std::vector<double> static_hedge(M);
  std::vector<double> bonds_static_hedge(M,bond_start);

  
  #pragma omp parallel num_threads(8) //16 threads possible but bottlenecked by memory bandwith
  { 
    double stock_position_static_hedge = delta_start;
    VectorXd mu_mod = (mu-r).matrix();
    ArrayXd vols_sq_penalised = alpha_penalty * vols.square();
    MatrixXd d_cov_mat = CRRA_delta * cov_mat;
    double h = T / double(N);
    auto dim = beta.size(); 
    ArrayXd C0(dim), C1(dim), C2(dim), C3(dim);
    prm_NV_CIR(h, beta, kappa, s, C0, C1, C2, C3); //setting parameters for NV scheme for CIR
    double ratio_N = N / double(N_delta_lvl);
    auto& gen = parallel_thread_rng<__LINE__>();
    boost::random::uniform_real_distribution<double> U(0, 1); // uni dist over (0,1)

    double price_PE, CI_price_PE, delta_PE, CI_delta_PE;
    double price_TI, CI_price_TI, delta_TI, CI_delta_TI;
    double price_EI, CI_price_EI, delta_EI, CI_delta_EI;
    #pragma omp for
    for(int j=0; j<M; j++)
    { 
      double current_age = start_age;
      double TtoMat = T;
      double  X = 1.0; //we start the fund as 1 to have the dynamic of 1 unit of fund
      ArrayXd Y = Y0;
      ArrayXd G(dim);

      ArrayXd e_CI = Y0 * vols_sq_penalised;
      d_cov_mat.diagonal().array() += e_CI;
      Eigen::LLT<Eigen::MatrixXd> llt_solver_start(d_cov_mat);
      VectorXd pi = llt_solver_start.solve(mu_mod);
      d_cov_mat.diagonal().array() -= e_CI;

      auto n_PE = X_PE.size(); // if this not works we pass with double and we use 0,1
      ArrayXb is_alive_PE(n_PE);
      is_alive_PE.setConstant(1);

      auto n_TI = X_TI.size();
      ArrayXb is_alive_TI(n_TI);
      is_alive_TI.setConstant(1);

      auto n_EI = X_EI.size();
      ArrayXb is_alive_EI(n_EI);
      is_alive_EI.setConstant(1);

      std::stack<int> TI_toPay;

      std::stack<int> EI_toPay;

      //int i = 0;
      //while( (i<N) & is_alive)
      for(int i=0; i<N; i++)
      { 
        double var_val  = 0.5 * pi.dot(cov_mat*pi);
        double mean_val = 0.5 * pi.dot(mu_mod);

        for(int l=0;l<dim;l++) G(l) = normcdfinv_as241(U(gen));

        //Carbon intensity evolution
        NV_CIR(Y, C0, C1, C2, C3, G);
        // Computation of portfolio weigths
        e_CI = Y * vols_sq_penalised; //carbon penalty = carbon transation cost
        d_cov_mat.diagonal().array() += e_CI;
        Eigen::LLT<Eigen::MatrixXd> llt_solver(d_cov_mat);
        pi = llt_solver.solve(mu_mod);
        d_cov_mat.diagonal().array() -= e_CI;
        // computation of the mean and variance of the logstock
        var_val  = h * ( var_val  + 0.5 * pi.dot(cov_mat*pi) );
        mean_val = h * ( mean_val + 0.5 * pi.dot(mu_mod) + r ) - 0.5 * var_val;
        //evolution of fund price
        double X2 = X*exp( mean_val + sqrt(var_val)*normcdfinv_as241(U(gen)) );

        //calculating the hedge gains
        trading_gains[j] += (X2 - X) * fund_shares(j);
        //std::cout << "trading_gains[j] = " << trading_gains[j]   << std::endl;

        //updating the value of the fund
        X = X2;
        
        // updating status of the Policyholders
        double survival_proba = GM_survival_proba(current_age, phi_GM, m_GM, b_GM, h);
        //std::cout << "survival_proba = " << survival_proba    << std::endl;
        //std::cout << "i = " << i   << std::endl;
        //std::cout << "ALIVE_PE = " << (is_alive_PE.template cast<double>()).sum()   << std::endl;
        //std::cout << "ALIVE_TI = " << (is_alive_TI.template cast<double>()).sum()   << std::endl;
        //std::cout << "NOMIN_PE = " << (X_PE * (is_alive_PE.template cast<double>())).sum()   << std::endl;
        //std::cout << "NOMIN_TI = " << (X_TI * (is_alive_TI.template cast<double>())).sum()   << std::endl;
        for(int l=0; l<is_alive_PE.size(); l++)
        {
          if(is_alive_PE(l)==true)
          {
            if(U(gen) > survival_proba) is_alive_PE(l) = false;
          }
        }
        for(int l=0;l<is_alive_TI.size();l++)
        {
          if(is_alive_TI(l)==true)
          {
            if(U(gen) > survival_proba)
            { 
              is_alive_TI(l) = false;
              TI_toPay.push(l); // i add the i-th ph to the list to be payed
            }
          }
        }

        for(int l=0;l<is_alive_EI.size();l++)
        {
          if(is_alive_EI(l)==true)
          {
            if(U(gen) > survival_proba)
            { 
              is_alive_EI(l) = false;
              EI_toPay.push(l); // i add the i-th ph to the list to be payed
            }
          }
        }

        TtoMat      -= h;
        current_age += h;
        double min_gua = exp(min_rate*(T-TtoMat));
        double max_gua = exp(max_rate*(T-TtoMat));

        // This part is the same if if(i<N-1) and if(i==N-1) so is outside

        double PRICE_DEAD = 0.0;
        double market_payoff;
        if      (X < min_gua) market_payoff = min_gua;
        else if (X > max_gua) market_payoff = max_gua;
        else                  market_payoff = X;
        while (!TI_toPay.empty()) { // Paying all the TI policy holder dead
          //std::cout << "Ti polizza da liquidare N = " << TI_toPay.top() << std::endl;
          PRICE_DEAD += X_TI(TI_toPay.top()); // Pay the last client
          TI_toPay.pop(); // Remove the last client from the list "toPay"
        }
        while (!EI_toPay.empty()) { // Paying all the TI policy holder dead
          //std::cout << "Ei polizza da liquidare N = " << EI_toPay.top() << std::endl;
          PRICE_DEAD += X_EI(EI_toPay.top())*perc_prem_death; // Pay the last client
          EI_toPay.pop(); // Remove the last client from the list "toPay"
        }
        PRICE_DEAD *=market_payoff;

        if(i<N-1)
        { 
          PETIEI_min_max_GM_MMM_MLMC(CRRA_delta, X, r, mu, vols, cov_mat, alpha_penalty, Y, beta, kappa, s,
                        current_age, phi_GM, m_GM, b_GM,
                        min_gua, max_gua, min_rate, max_rate, perc_prem_death, 
                        TtoMat, N_delta_lvl, M_delta,
                        price_PE, CI_price_PE, delta_PE, CI_delta_PE,
                        price_TI, CI_price_TI, delta_TI, CI_delta_TI,
                        price_EI, CI_price_EI, delta_EI, CI_delta_EI);

          //std::ceil((N-i-1)/ratio_N) //how can i optimize?

          double nominal_PE = (X_PE * (is_alive_PE.template cast<double>())).sum();
          double nominal_TI = (X_TI * (is_alive_TI.template cast<double>())).sum();
          double nominal_EI = (X_EI * (is_alive_EI.template cast<double>())).sum();
          double PRICE_ALIVE = nominal_PE*price_PE + nominal_TI*price_TI + nominal_EI*price_EI;
          
          COST[j]       += PRICE_ALIVE + PRICE_DEAD - (bonds(j)*exp(r*h) + fund_shares(j)*X);
          fund_shares(j) = nominal_PE*delta_PE + nominal_TI*delta_TI + nominal_EI*delta_EI;
          bonds(j)       = PRICE_ALIVE - fund_shares(j)*X;
          
          no_hedge[j]    = no_hedge[j]*exp(r*h) - PRICE_DEAD;

          static_hedge[j] = stock_position_static_hedge*X + bonds_static_hedge[j]*exp(r*h) 
                            - PRICE_DEAD;
          stock_position_static_hedge = nominal_PE*delta_start_PE + nominal_TI*delta_start_TI
                                                        + nominal_EI*delta_start_EI;     
          bonds_static_hedge[j] = static_hedge[j] - stock_position_static_hedge*X;
          //std::cout << "PRICE_ALIVE = " << PRICE_ALIVE << std::endl;
          //std::cout << "PRICE_DEAD  = " << PRICE_DEAD << std::endl;
          //std::cout << "COST =" << COST[j] << std::endl;
        }
        else
        { // at expiration T we need to pay the TI/EI dead in [T-h,T] and the PE/EI ALIVE at T

          double nominal_alive_to_pay = (X_PE * (is_alive_PE.template cast<double>())).sum()
                                      + (X_EI * (is_alive_EI.template cast<double>())).sum();
          COST[j]       += PRICE_DEAD + nominal_alive_to_pay*market_payoff - (bonds(j)*exp(r*h) + fund_shares(j)*X);
          no_hedge[j]    = PRICE_DEAD + nominal_alive_to_pay*market_payoff - no_hedge[j]*exp(r*h);


          static_hedge[j] = stock_position_static_hedge*X + bonds_static_hedge[j]*exp(r*h) 
                            - PRICE_DEAD - nominal_alive_to_pay*market_payoff;
          static_hedge[j] = -static_hedge[j];
          bonds(j)       = 0.0; //price - delta*X;
          fund_shares(j) = 0.0; //delta;
        }

      }
      X_vec[j] = X; //X a time T (maturity)

    }
  } //end parallel zone


  double mean_cost = mean(COST);
  std::pair<double,double> output_var = empirical_var_CI95(COST);
  std::cout << "expected value of cost = " << mean_cost    << std::endl;
  std::cout << "CI95  expected value   = " << 1.96*sqrt(output_var.first/M) << std::endl;
  std::cout << "     var cost = " << output_var.first << std::endl;
  std::cout << "CI95 var cost = " << output_var.second << std::endl;


  Output_hedging output;
  output.no_hedge = no_hedge;
  output.static_hedge = static_hedge;
  output.trading_gains = trading_gains;
  output.hedging_cost = COST;
  output.start_price = price_start_PE+price_start_TI+price_start_EI;
  output.X = X_vec;
  return output;
}



Scenarios
fund_ci_scenario_generator(double CRRA_delta, double r, ArrayXd mu, ArrayXd vols, MatrixXd cov_mat, 
									      double alpha_penalty, ArrayXd Y0, ArrayXd beta, ArrayXd kappa, ArrayXd s,
                        double T, int N, int M)
{ 
  std::vector<std::vector<double>>  X_scenarios(M, std::vector<double >(N+1, 1.0));
  std::vector<std::vector<ArrayXd>> Y_scenarios(M, std::vector<ArrayXd>(N+1, Y0) );

  
  #pragma omp parallel num_threads(8) //16 threads possible but bottlenecked by memory bandwith
  { 

    VectorXd mu_mod = (mu-r).matrix();
    ArrayXd vols_sq_penalised = alpha_penalty * vols.square();
    MatrixXd d_cov_mat = CRRA_delta * cov_mat;
    double h = T / double(N);
    auto dim = beta.size(); 
    ArrayXd C0(dim), C1(dim), C2(dim), C3(dim);
    prm_NV_CIR(h, beta, kappa, s, C0, C1, C2, C3); //setting parameters for NV scheme for CIR

    auto& gen = parallel_thread_rng<__LINE__>();
    boost::random::uniform_real_distribution<double> U(0, 1); // uni dist over (0,1)


    #pragma omp for
    for(int j=0; j<M; j++)
    { 
      ArrayXd Y = Y0;
      ArrayXd G(dim);

      ArrayXd e_CI = Y0 * vols_sq_penalised;
      d_cov_mat.diagonal().array() += e_CI;
      Eigen::LLT<Eigen::MatrixXd> llt_solver_start(d_cov_mat);
      VectorXd pi = llt_solver_start.solve(mu_mod);
      d_cov_mat.diagonal().array() -= e_CI;


      for(int i=0; i<N; i++)
      { 
        double var_val  = 0.5 * pi.dot(cov_mat*pi);
        double mean_val = 0.5 * pi.dot(mu_mod);

        for(int l=0;l<dim;l++) G(l) = normcdfinv_as241(U(gen));

        //Carbon intensity evolution
        NV_CIR(Y, C0, C1, C2, C3, G);
        Y_scenarios[j][i+1] = Y;
        // Computation of portfolio weigths
        e_CI = Y * vols_sq_penalised; //carbon penalty = carbon transation cost
        d_cov_mat.diagonal().array() += e_CI;
        Eigen::LLT<Eigen::MatrixXd> llt_solver(d_cov_mat);
        pi = llt_solver.solve(mu_mod);
        d_cov_mat.diagonal().array() -= e_CI;
        // computation of the mean and variance of the logstock
        var_val  = h * ( var_val  + 0.5 * pi.dot(cov_mat*pi) );
        mean_val = h * ( mean_val + 0.5 * pi.dot(mu_mod) + r ) - 0.5 * var_val;
        //evolution of fund price
        X_scenarios[j][i+1] = X_scenarios[j][i]*exp( mean_val + sqrt(var_val)*normcdfinv_as241(U(gen)) );
      }
    }
  }
  Scenarios XY_scenarios;
  XY_scenarios.X = X_scenarios;
  XY_scenarios.Y = Y_scenarios;

  return XY_scenarios;
}



template <class NormalSampler = As241NormalSampler>
Scenarios
fund_ci_scenario_generator_new(double CRRA_delta, double r, ArrayXd mu, ArrayXd vols, MatrixXd cov_mat, 
									      double alpha_penalty, ArrayXd Y0, ArrayXd beta, ArrayXd kappa, ArrayXd s,
                        double T, int N, int M, prm_comp prm_func, scheme_update scheme)
{ 
  std::vector<std::vector<double>>  X_scenarios(M, std::vector<double >(N+1, 1.0));
  std::vector<std::vector<ArrayXd>> Y_scenarios(M, std::vector<ArrayXd>(N+1, Y0) );

  
  #pragma omp parallel num_threads(8) //16 threads possible but bottlenecked by memory bandwith
  { 

    VectorXd mu_mod = (mu-r).matrix();
    ArrayXd vols_sq_penalised = alpha_penalty * vols.square();
    MatrixXd d_cov_mat = CRRA_delta * cov_mat;
    double h = T / double(N);
    auto dim = beta.size(); 
    ArrayXd C0(dim), C1(dim), C2(dim), C3(dim);
    prm_func(h, beta, kappa, s, C0, C1, C2, C3); //setting parameters for NV scheme for CIR

    auto& gen = parallel_thread_rng<__LINE__>();
    NormalSampler normal;


    #pragma omp for
    for(int j=0; j<M; j++)
    { 
      ArrayXd Y = Y0;
      ArrayXd G(dim);

      ArrayXd e_CI = Y0 * vols_sq_penalised;
      d_cov_mat.diagonal().array() += e_CI;
      Eigen::LLT<Eigen::MatrixXd> llt_solver_start(d_cov_mat);
      VectorXd pi = llt_solver_start.solve(mu_mod);
      d_cov_mat.diagonal().array() -= e_CI;


      for(int i=0; i<N; i++)
      { 
        double var_val  = 0.5 * pi.dot(cov_mat*pi);
        double mean_val = 0.5 * pi.dot(mu_mod);

        for(int l=0;l<dim;l++) G(l) = normal(gen);

        //Carbon intensity evolution
        scheme(Y, C0, C1, C2, C3, G);
        Y_scenarios[j][i+1] = Y;
        // Computation of portfolio weigths
        e_CI = Y * vols_sq_penalised; //carbon penalty = carbon transation cost
        d_cov_mat.diagonal().array() += e_CI;
        Eigen::LLT<Eigen::MatrixXd> llt_solver(d_cov_mat);
        pi = llt_solver.solve(mu_mod);
        d_cov_mat.diagonal().array() -= e_CI;
        // computation of the mean and variance of the logstock
        var_val  = h * ( var_val  + 0.5 * pi.dot(cov_mat*pi) );
        mean_val = h * ( mean_val + 0.5 * pi.dot(mu_mod) + r ) - 0.5 * var_val;
        //evolution of fund price
        X_scenarios[j][i+1] = X_scenarios[j][i]*exp( mean_val + sqrt(var_val)*normal(gen) );
      }
    }
  }
  Scenarios XY_scenarios;
  XY_scenarios.X = X_scenarios;
  XY_scenarios.Y = Y_scenarios;

  return XY_scenarios;
}



Output_hedging
PETIEI_hedge_cost_paral_mc(double CRRA_delta, double r, ArrayXd mu, ArrayXd vols, MatrixXd cov_mat, 
									double alpha_penalty, ArrayXd Y0, ArrayXd beta, ArrayXd kappa, ArrayXd s,
                  PETIEI_polyciholders PH_DATA, double phi_GM, double m_GM, double b_GM, 
                  double min_rate, double max_rate, double  perc_prem_death, 
                  double T, int N, int N_delta, int M, int M_delta)
{
  // different N for number of times of rebalancing portfolio and computing deltas
  // compute of prices and deltas are all in the MMM
  // In output 3 strategies:
  // 1) no hedging investment on the cash account
  // 2) static hedge
  // 3) continous hedging

  // several classe ages

  double price_start_PE, var_price_start_PE, delta_start_PE, var_delta_start_PE;
  double price_start_TI, var_price_start_TI, delta_start_TI, var_delta_start_TI;
  double price_start_EI, var_price_start_EI, delta_start_EI, var_delta_start_EI;

  //X0 = 1.0
  PETIEI_min_max_GM_MMM_mc(CRRA_delta, 1.0, r, mu, vols, cov_mat, alpha_penalty, Y0, beta, kappa, s,
                  PH_DATA.nominals_sum_PE, PH_DATA.age_PE, 
                  PH_DATA.nominals_sum_TI, PH_DATA.age_TI,
                  PH_DATA.nominals_sum_EI, PH_DATA.age_EI,
                  phi_GM, m_GM, b_GM,
                  1.0, 1.0, min_rate, max_rate, perc_prem_death, T, N_delta, M_delta, 
                  price_start_PE, var_price_start_PE, delta_start_PE, var_delta_start_PE,
                  price_start_TI, var_price_start_TI, delta_start_TI, var_delta_start_TI,
                  price_start_EI, var_price_start_EI, delta_start_EI, var_delta_start_EI);
  
  std::cout << "price_PE in MMM = " << price_start_PE  << std::endl;
  std::cout << "           CI95 = " << 1.96*sqrt(var_price_start_PE/M_delta)  << std::endl;
  std::cout << "price_TI in MMM = " << price_start_TI  << std::endl;
  std::cout << "           CI95 = " << 1.96*sqrt(var_price_start_TI/M_delta)  << std::endl;
  std::cout << "price_EI in MMM = " << price_start_EI  << std::endl;
  std::cout << "           CI95 = " << 1.96*sqrt(var_price_start_EI/M_delta)  << std::endl;
  std::cout << "delta_PE at time 0 = " << delta_start_PE  << std::endl;
  std::cout << "              CI95 = " << 1.96*sqrt(var_delta_start_PE/M_delta)  << std::endl;
  std::cout << "delta_TI at time 0 = " << delta_start_TI  << std::endl;
  std::cout << "              CI95 = " << 1.96*sqrt(var_delta_start_TI/M_delta)  << std::endl;
  std::cout << "delta_EI at time 0 = " << delta_start_EI  << std::endl;
  std::cout << "              CI95 = " << 1.96*sqrt(var_delta_start_EI/M_delta)  << std::endl;
  

  double price_start = price_start_PE + price_start_TI + price_start_EI;
  double delta_start = delta_start_PE + delta_start_TI + delta_start_EI;

  std::cout << "price_start = " << price_start   << std::endl;
  std::cout << "delta_start = " << delta_start   << std::endl;


  double bond_start = price_start  - delta_start; //X0=1.0
  std::vector<double> bonds(M, bond_start);

  std::vector<double> fund_shares(M, delta_start);

  std::vector<double> trading_gains(M, 0.0);
  std::vector<double> X_vec(M, 0.0);
  std::vector<double> COST(M, 0.0);

  std::vector<double> no_hedge(M, price_start);
  std::vector<double> static_hedge(M);
  std::vector<double> bonds_static_hedge(M,bond_start);

  
  #pragma omp parallel num_threads(8) firstprivate(PH_DATA) //16 threads possible but bottlenecked by memory bandwith
  { 

    VectorXd mu_mod = (mu-r).matrix();
    ArrayXd vols_sq_penalised = alpha_penalty * vols.square();
    MatrixXd d_cov_mat = CRRA_delta * cov_mat;
    double h = T / double(N);
    auto dim = beta.size(); 
    ArrayXd C0(dim), C1(dim), C2(dim), C3(dim);
    prm_NV_CIR(h, beta, kappa, s, C0, C1, C2, C3); //setting parameters for NV scheme for CIR
    double ratio_N = N / double(N_delta);
    auto& gen = parallel_thread_rng<__LINE__>();
    boost::random::uniform_real_distribution<double> U(0, 1); // uni dist over (0,1)

    double price_PE, var_price_PE, delta_PE, var_delta_PE;
    double price_TI, var_price_TI, delta_TI, var_delta_TI;
    double price_EI, var_price_EI, delta_EI, var_delta_EI;

    std::vector<double> start_age_PE = PH_DATA.age_PE;
    std::vector<double> start_age_TI = PH_DATA.age_TI;
    std::vector<double> start_age_EI = PH_DATA.age_EI;

    #pragma omp for
    for(int j=0; j<M; j++)
    { 
      double stock_position_static_hedge = delta_start;
      double TtoMat = T;
      double  X = 1.0; //we start the fund as 1 to have the dynamic of 1 unit of fund
      ArrayXd Y = Y0;
      ArrayXd G(dim);

      ArrayXd e_CI = Y0 * vols_sq_penalised;
      d_cov_mat.diagonal().array() += e_CI;
      Eigen::LLT<Eigen::MatrixXd> llt_solver_start(d_cov_mat);
      VectorXd pi = llt_solver_start.solve(mu_mod);
      d_cov_mat.diagonal().array() -= e_CI;
      
      // resetting for each simulations the good starting values for the ph
      // ages, nominals_sum and live status
      PH_DATA.age_PE = start_age_PE;
      PH_DATA.age_TI = start_age_TI;
      PH_DATA.age_EI = start_age_EI;
      for(int q=0; q<PH_DATA.is_alive_PE.size(); q++)
      { 
        PH_DATA.nominals_sum_PE[q] = PH_DATA.nominals_PE[q].sum();
        PH_DATA.is_alive_PE[q].setConstant(true);
      } 
      for(int q=0; q<PH_DATA.is_alive_TI.size(); q++)
      {
        PH_DATA.nominals_sum_TI[q] = PH_DATA.nominals_TI[q].sum();
        PH_DATA.is_alive_TI[q].setConstant(true);
      }
      for(int q=0; q<PH_DATA.is_alive_EI.size(); q++)
      {
        PH_DATA.nominals_sum_EI[q] = PH_DATA.nominals_EI[q].sum();
        PH_DATA.is_alive_EI[q].setConstant(true);
      }


      for(int i=0; i<N; i++)
      { 
        double var_val  = 0.5 * pi.dot(cov_mat*pi);
        double mean_val = 0.5 * pi.dot(mu_mod);

        for(int l=0;l<dim;l++) G(l) = normcdfinv_as241(U(gen));

        //Carbon intensity evolution
        NV_CIR(Y, C0, C1, C2, C3, G);
        // Computation of portfolio weigths
        e_CI = Y * vols_sq_penalised; //carbon penalty = carbon transation cost
        d_cov_mat.diagonal().array() += e_CI;
        Eigen::LLT<Eigen::MatrixXd> llt_solver(d_cov_mat);
        pi = llt_solver.solve(mu_mod);
        d_cov_mat.diagonal().array() -= e_CI;
        // computation of the mean and variance of the logstock
        var_val  = h * ( var_val  + 0.5 * pi.dot(cov_mat*pi) );
        mean_val = h * ( mean_val + 0.5 * pi.dot(mu_mod) + r ) - 0.5 * var_val;
        //evolution of fund price
        double X2 = X*exp( mean_val + sqrt(var_val)*normcdfinv_as241(U(gen)) );

        //calculating the hedge gains
        trading_gains[j] += (X2 - X) * fund_shares[j];
        //std::cout << "trading_gains[j] = " << trading_gains[j]   << std::endl;

        //updating the value of the fund
        X = X2;

        
        // updating status of the Policyholders and calculating the claims to be paid
        double PRICE_DEAD = 0.0;
        for(int q=0; q<PH_DATA.nominals_PE.size(); q++)
        { 
          double survival_proba = GM_survival_proba(PH_DATA.age_PE[q], phi_GM, m_GM, b_GM, h);
          for(int l=0; l<PH_DATA.is_alive_PE[q].size(); l++)
          {
            if(PH_DATA.is_alive_PE[q](l)==true)
            { 
              if(U(gen) > survival_proba)
              {
                PH_DATA.is_alive_PE[q](l) = false;
                PH_DATA.nominals_sum_PE[q] -= PH_DATA.nominals_PE[q](l);
              } 
            }
          }
        }
        for(int q=0; q<PH_DATA.nominals_TI.size(); q++)
        {
          double survival_proba = GM_survival_proba(PH_DATA.age_TI[q], phi_GM, m_GM, b_GM, h);
          for(int l=0; l<PH_DATA.is_alive_TI[q].size(); l++)
          {
            if(PH_DATA.is_alive_TI[q](l)==true)
            {
              if(U(gen) > survival_proba)
              {
                PH_DATA.is_alive_TI[q](l) = false;
                PH_DATA.nominals_sum_TI[q] -= PH_DATA.nominals_TI[q](l);
                //TI_toPay.push(std::make_pair(q,l)); // add the (q,l) pair ph to the list to be payed
                PRICE_DEAD                 += PH_DATA.nominals_TI[q](l);
              } 
            }
          }
        }

        for(int q=0; q<PH_DATA.nominals_EI.size(); q++)
        {
          double survival_proba = GM_survival_proba(PH_DATA.age_EI[q], phi_GM, m_GM, b_GM, h);
          for(int l=0; l<PH_DATA.is_alive_EI[q].size(); l++)
          {
            if(PH_DATA.is_alive_EI[q](l)==true)
            {
              if(U(gen) > survival_proba)
              {
                PH_DATA.is_alive_EI[q](l) = false;
                PH_DATA.nominals_sum_EI[q] -= PH_DATA.nominals_EI[q](l);
                //EI_toPay.push(std::make_pair(q,l)); // add the (q,l) pair ph to the list to be payed
                PRICE_DEAD += perc_prem_death*PH_DATA.nominals_EI[q](l);
              } 
            }
          }
        }

        TtoMat         -= h;
        PH_DATA.age_PE += h;
        PH_DATA.age_TI += h;
        PH_DATA.age_EI += h;
        double min_gua = exp(min_rate*(T-TtoMat)); //X0=1
        double max_gua = exp(max_rate*(T-TtoMat)); //X0=1

        
        double market_payoff;
        if      (X < min_gua) market_payoff = min_gua;
        else if (X > max_gua) market_payoff = max_gua;
        else                  market_payoff = X;
        /* while (!TI_toPay.empty()) { // Paying all the TI policy holder dead
          PRICE_DEAD += PH_DATA.nominals_TI[TI_toPay.top().first](TI_toPay.top().second); // calculating the nom_sum of phs dead
          TI_toPay.pop(); // Remove the last client from the list "toPay"
        }
        while (!EI_toPay.empty()) { // Paying all the EI policy holder dead
          PRICE_DEAD += perc_prem_death*PH_DATA.nominals_EI[EI_toPay.top().first](EI_toPay.top().second); // calculating the nom_sum of phs dead
          EI_toPay.pop(); // Remove the last client from the list "toPay"
        } */
        PRICE_DEAD *= market_payoff;

        if(i<N-1)
        { 
          PETIEI_min_max_GM_MMM_mc(CRRA_delta, X, r, mu, vols, cov_mat, alpha_penalty, Y, beta, kappa, s,
                        PH_DATA.nominals_sum_PE, PH_DATA.age_PE, 
                        PH_DATA.nominals_sum_TI, PH_DATA.age_TI,
                        PH_DATA.nominals_sum_EI, PH_DATA.age_EI,
                        phi_GM, m_GM, b_GM,
                        min_gua, max_gua, min_rate, max_rate, perc_prem_death, 
                        TtoMat, std::ceil((N-i-1)/ratio_N), M_delta,
                        price_PE, var_price_PE, delta_PE, var_delta_PE,
                        price_TI, var_price_TI, delta_TI, var_delta_TI,
                        price_EI, var_price_EI, delta_EI, var_delta_EI);
          
          double PRICE_ALIVE = price_PE + price_TI + price_EI;
          

          COST[j]       += PRICE_ALIVE + PRICE_DEAD - (bonds[j]*exp(r*h) + fund_shares[j]*X);
          fund_shares[j] = delta_PE + delta_TI + delta_EI;
          bonds[j]       = PRICE_ALIVE - fund_shares[j]*X;
          no_hedge[j]    = no_hedge[j]*exp(r*h) - PRICE_DEAD;

          
          static_hedge[j] = stock_position_static_hedge*X + bonds_static_hedge[j]*exp(r*h) - PRICE_DEAD;
          stock_position_static_hedge = sum(PH_DATA.nominals_sum_PE)*delta_start_PE 
                                      + sum(PH_DATA.nominals_sum_TI)*delta_start_TI
                                      + sum(PH_DATA.nominals_sum_EI)*delta_start_EI;     
          bonds_static_hedge[j] = static_hedge[j] - stock_position_static_hedge*X;

          //std::cout << "PRICE_ALIVE = " << PRICE_ALIVE << std::endl;
          //std::cout << "PRICE_DEAD  = " << PRICE_DEAD << std::endl;
          //std::cout << "COST =" << COST[j] << std::endl;
        }
        else
        { // at expiration T we need to pay the TI and EI dead in [T-h,T] and the PE and EI ALIVE at T
          double nominals_alive = sum(PH_DATA.nominals_sum_PE) + sum(PH_DATA.nominals_sum_EI);
          COST[j]       += PRICE_DEAD + nominals_alive*market_payoff - (bonds[j]*exp(r*h) + fund_shares[j]*X);
          no_hedge[j]    = PRICE_DEAD + nominals_alive*market_payoff - no_hedge[j]*exp(r*h);

          static_hedge[j] = stock_position_static_hedge*X + bonds_static_hedge[j]*exp(r*h) 
                            - PRICE_DEAD  - nominals_alive*market_payoff;
          static_hedge[j] = -static_hedge[j];
          //bonds[j]       = 0.0; //price - delta*X;
          //fund_shares[j] = 0.0; //delta;
        }

      }
      X_vec[j] = X; //X a time T (maturity)

    }
  } //end parallel zone
  //std::cout << "\ntrading_gains:\n" << trading_gains << std::endl;
  //std::cout << "\nfund_shares:\n" << fund_shares << std::endl;
  //double mean_hedging_gains = trading_gains.mean();
  //double std_dev_hedging_gains = sqrt( ((trading_gains -mean_hedging_gains).square()).mean() );
  //std::cout << "expected trading gains = " << mean_hedging_gains    << std::endl;
  //std::cout << "CI95                   = " << 1.96*std_dev_hedging_gains/sqrt(M) << std::endl;
  //std::cout << "std_dev trading gains  = " << std_dev_hedging_gains << std::endl;

  double mean_cost = mean(COST);
  std::pair<double,double> output_var = empirical_var_CI95(COST);
  std::cout << "expected value of cost = " << mean_cost    << std::endl;
  std::cout << "CI95  expected value   = " << 1.96*sqrt(output_var.first/M) << std::endl;
  std::cout << "     var cost = " << output_var.first << std::endl;
  std::cout << "CI95 var cost = " << output_var.second << std::endl;

  Output_hedging output;
  output.no_hedge = no_hedge;
  output.static_hedge = static_hedge;
  output.trading_gains = trading_gains;
  output.hedging_cost = COST;
  output.start_price = price_start_PE+price_start_TI+price_start_EI;
  output.X = X_vec;
  return output;
}



Output_hedging
PETIEI_hedge_cost_paral_mc_MLMC(double CRRA_delta, double r, ArrayXd mu, ArrayXd vols, MatrixXd cov_mat, 
									double alpha_penalty, ArrayXd Y0, ArrayXd beta, ArrayXd kappa, ArrayXd s,
                  PETIEI_polyciholders PH_DATA, double phi_GM, double m_GM, double b_GM, 
                  double min_rate, double max_rate, double  perc_prem_death, 
                  double T, int N, int N_delta_lvl, int M, int M_delta)
{
  // different N for number of times of rebalancing portfolio and computing deltas
  // compute of prices and deltas are all in the MMM
  // In output 3 strategies:
  // 1) no hedging investment on the cash account
  // 2) static hedge
  // 3) continous hedging

  // several classe ages

  double price_start_PE, CI_price_start_PE, delta_start_PE, CI_delta_start_PE;
  double price_start_TI, CI_price_start_TI, delta_start_TI, CI_delta_start_TI;
  double price_start_EI, CI_price_start_EI, delta_start_EI, CI_delta_start_EI;

  //X0 = 1.0
  PETIEI_min_max_GM_MMM_mc_MLMC(CRRA_delta, 1.0, r, mu, vols, cov_mat, alpha_penalty, Y0, beta, kappa, s,
                  PH_DATA.nominals_sum_PE, PH_DATA.age_PE, 
                  PH_DATA.nominals_sum_TI, PH_DATA.age_TI,
                  PH_DATA.nominals_sum_EI, PH_DATA.age_EI,
                  phi_GM, m_GM, b_GM,
                  1.0, 1.0, min_rate, max_rate, perc_prem_death, 
                  T, N_delta_lvl, M_delta, 
                  price_start_PE, CI_price_start_PE, delta_start_PE, CI_delta_start_PE,
                  price_start_TI, CI_price_start_TI, delta_start_TI, CI_delta_start_TI,
                  price_start_EI, CI_price_start_EI, delta_start_EI, CI_delta_start_EI);
  
  std::cout << "price_PE in MMM = " << price_start_PE  << std::endl;
  std::cout << "           CI95 = " << CI_price_start_PE  << std::endl;
  std::cout << "price_TI in MMM = " << price_start_TI  << std::endl;
  std::cout << "           CI95 = " << CI_price_start_TI  << std::endl;
  std::cout << "price_EI in MMM = " << price_start_EI  << std::endl;
  std::cout << "           CI95 = " << CI_price_start_EI  << std::endl;
  std::cout << "delta_PE at time 0 = " << delta_start_PE  << std::endl;
  std::cout << "              CI95 = " << CI_delta_start_PE  << std::endl;
  std::cout << "delta_TI at time 0 = " << delta_start_TI  << std::endl;
  std::cout << "              CI95 = " << CI_delta_start_TI  << std::endl;
  std::cout << "delta_EI at time 0 = " << delta_start_EI  << std::endl;
  std::cout << "              CI95 = " << CI_delta_start_EI  << std::endl;
  

  double price_start = price_start_PE + price_start_TI + price_start_EI;
  double delta_start = delta_start_PE + delta_start_TI + delta_start_EI;

  std::cout << "price_start = " << price_start   << std::endl;
  std::cout << "delta_start = " << delta_start   << std::endl;


  double bond_start = price_start  - delta_start; //X0=1.0
  std::vector<double> bonds(M, bond_start);

  std::vector<double> fund_shares(M, delta_start);

  std::vector<double> trading_gains(M, 0.0);
  std::vector<double> X_vec(M, 0.0);
  std::vector<double> COST(M, 0.0);

  std::vector<double> no_hedge(M, price_start);
  std::vector<double> static_hedge(M);
  std::vector<double> bonds_static_hedge(M,bond_start);

  
  #pragma omp parallel num_threads(8) firstprivate(PH_DATA) //16 threads possible but bottlenecked by memory bandwith
  { 
    double stock_position_static_hedge = delta_start;

    VectorXd mu_mod = (mu-r).matrix();
    ArrayXd vols_sq_penalised = alpha_penalty * vols.square();
    MatrixXd d_cov_mat = CRRA_delta * cov_mat;
    double h = T / double(N);
    auto dim = beta.size(); 
    ArrayXd C0(dim), C1(dim), C2(dim), C3(dim);
    prm_NV_CIR(h, beta, kappa, s, C0, C1, C2, C3); //setting parameters for NV scheme for CIR
    double ratio_N = N / double(N_delta_lvl);
    auto& gen = parallel_thread_rng<__LINE__>();
    boost::random::uniform_real_distribution<double> U(0, 1); // uni dist over (0,1)

    double price_PE, CI_price_PE, delta_PE, CI_delta_PE,
           price_TI, CI_price_TI, delta_TI, CI_delta_TI,
           price_EI, CI_price_EI, delta_EI, CI_delta_EI;

    std::vector<double> start_age_PE = PH_DATA.age_PE;
    std::vector<double> start_age_TI = PH_DATA.age_TI;
    std::vector<double> start_age_EI = PH_DATA.age_EI;

    #pragma omp for
    for(int j=0; j<M; j++)
    { 
      double TtoMat = T;
      double  X = 1.0; //we start the fund as 1 to have the dynamic of 1 unit of fund
      ArrayXd Y = Y0;
      ArrayXd G(dim);

      ArrayXd e_CI = Y0 * vols_sq_penalised;
      d_cov_mat.diagonal().array() += e_CI;
      Eigen::LLT<Eigen::MatrixXd> llt_solver_start(d_cov_mat);
      VectorXd pi = llt_solver_start.solve(mu_mod);
      d_cov_mat.diagonal().array() -= e_CI;
      
      // resetting for each simulations the good starting values for the ph
      // ages, nominals_sum and live status
      PH_DATA.age_PE = start_age_PE;
      PH_DATA.age_TI = start_age_TI;
      PH_DATA.age_EI = start_age_EI;
      for(int q=0; q<PH_DATA.is_alive_PE.size(); q++)
      { 
        PH_DATA.nominals_sum_PE[q] = PH_DATA.nominals_PE[q].sum();
        PH_DATA.is_alive_PE[q].setConstant(true);
      } 
      for(int q=0; q<PH_DATA.is_alive_TI.size(); q++)
      {
        PH_DATA.nominals_sum_TI[q] = PH_DATA.nominals_TI[q].sum();
        PH_DATA.is_alive_TI[q].setConstant(true);
      }
      for(int q=0; q<PH_DATA.is_alive_EI.size(); q++)
      {
        PH_DATA.nominals_sum_EI[q] = PH_DATA.nominals_EI[q].sum();
        PH_DATA.is_alive_EI[q].setConstant(true);
      }


      for(int i=0; i<N; i++)
      { 
        double var_val  = 0.5 * pi.dot(cov_mat*pi);
        double mean_val = 0.5 * pi.dot(mu_mod);

        for(int l=0;l<dim;l++) G(l) = normcdfinv_as241(U(gen));

        //Carbon intensity evolution
        NV_CIR(Y, C0, C1, C2, C3, G);
        // Computation of portfolio weigths
        e_CI = Y * vols_sq_penalised; //carbon penalty = carbon transation cost
        d_cov_mat.diagonal().array() += e_CI;
        Eigen::LLT<Eigen::MatrixXd> llt_solver(d_cov_mat);
        pi = llt_solver.solve(mu_mod);
        d_cov_mat.diagonal().array() -= e_CI;
        // computation of the mean and variance of the logstock
        var_val  = h * ( var_val  + 0.5 * pi.dot(cov_mat*pi) );
        mean_val = h * ( mean_val + 0.5 * pi.dot(mu_mod) + r ) - 0.5 * var_val;
        //evolution of fund price
        double X2 = X*exp( mean_val + sqrt(var_val)*normcdfinv_as241(U(gen)) );

        //calculating the hedge gains
        trading_gains[j] += (X2 - X) * fund_shares[j];
        //std::cout << "trading_gains[j] = " << trading_gains[j]   << std::endl;

        //updating the value of the fund
        X = X2;

        
        // updating status of the Policyholders and calculating the claims to be paid
        double PRICE_DEAD = 0.0;
        for(int q=0; q<PH_DATA.nominals_PE.size(); q++)
        { 
          double survival_proba = GM_survival_proba(PH_DATA.age_PE[q], phi_GM, m_GM, b_GM, h);
          for(int l=0; l<PH_DATA.is_alive_PE[q].size(); l++)
          {
            if(PH_DATA.is_alive_PE[q](l)==true)
            { 
              if(U(gen) > survival_proba)
              {
                PH_DATA.is_alive_PE[q](l) = false;
                PH_DATA.nominals_sum_PE[q] -= PH_DATA.nominals_PE[q](l);
              } 
            }
          }
        }
        for(int q=0; q<PH_DATA.nominals_TI.size(); q++)
        {
          double survival_proba = GM_survival_proba(PH_DATA.age_TI[q], phi_GM, m_GM, b_GM, h);
          for(int l=0; l<PH_DATA.is_alive_TI[q].size(); l++)
          {
            if(PH_DATA.is_alive_TI[q](l)==true)
            {
              if(U(gen) > survival_proba)
              {
                PH_DATA.is_alive_TI[q](l) = false;
                PH_DATA.nominals_sum_TI[q] -= PH_DATA.nominals_TI[q](l);
                //TI_toPay.push(std::make_pair(q,l)); // add the (q,l) pair ph to the list to be payed
                PRICE_DEAD                 += PH_DATA.nominals_TI[q](l);
              } 
            }
          }
        }

        for(int q=0; q<PH_DATA.nominals_EI.size(); q++)
        {
          double survival_proba = GM_survival_proba(PH_DATA.age_EI[q], phi_GM, m_GM, b_GM, h);
          for(int l=0; l<PH_DATA.is_alive_EI[q].size(); l++)
          {
            if(PH_DATA.is_alive_EI[q](l)==true)
            {
              if(U(gen) > survival_proba)
              {
                PH_DATA.is_alive_EI[q](l) = false;
                PH_DATA.nominals_sum_EI[q] -= PH_DATA.nominals_EI[q](l);
                //EI_toPay.push(std::make_pair(q,l)); // add the (q,l) pair ph to the list to be payed
                PRICE_DEAD += perc_prem_death*PH_DATA.nominals_EI[q](l);
              } 
            }
          }
        }

        TtoMat         -= h;
        PH_DATA.age_PE += h;
        PH_DATA.age_TI += h;
        PH_DATA.age_EI += h;
        double min_gua = exp(min_rate*(T-TtoMat)); //X0=1
        double max_gua = exp(max_rate*(T-TtoMat)); //X0=1

        
        double market_payoff;
        if      (X < min_gua) market_payoff = min_gua;
        else if (X > max_gua) market_payoff = max_gua;
        else                  market_payoff = X;
        /* while (!TI_toPay.empty()) { // Paying all the TI policy holder dead
          PRICE_DEAD += PH_DATA.nominals_TI[TI_toPay.top().first](TI_toPay.top().second); // calculating the nom_sum of phs dead
          TI_toPay.pop(); // Remove the last client from the list "toPay"
        }
        while (!EI_toPay.empty()) { // Paying all the EI policy holder dead
          PRICE_DEAD += perc_prem_death*PH_DATA.nominals_EI[EI_toPay.top().first](EI_toPay.top().second); // calculating the nom_sum of phs dead
          EI_toPay.pop(); // Remove the last client from the list "toPay"
        } */
        PRICE_DEAD *= market_payoff;

        if(i<N-1)
        { 
          PETIEI_min_max_GM_MMM_mc_MLMC(CRRA_delta, X, r, mu, vols, cov_mat, alpha_penalty, Y, beta, kappa, s,
                        PH_DATA.nominals_sum_PE, PH_DATA.age_PE, 
                        PH_DATA.nominals_sum_TI, PH_DATA.age_TI,
                        PH_DATA.nominals_sum_EI, PH_DATA.age_EI,
                        phi_GM, m_GM, b_GM,
                        min_gua, max_gua, min_rate, max_rate, perc_prem_death, 
                        TtoMat, N_delta_lvl, M_delta,
                        price_PE, CI_price_PE, delta_PE, CI_delta_PE,
                        price_TI, CI_price_TI, delta_TI, CI_delta_TI,
                        price_EI, CI_price_EI, delta_EI, CI_delta_EI);
          
          //one could optimize n_delta_lvl and M_delta going forward in time too much precision at the end
          double PRICE_ALIVE = price_PE + price_TI + price_EI;
          

          COST[j]       += PRICE_ALIVE + PRICE_DEAD - (bonds[j]*exp(r*h) + fund_shares[j]*X);
          fund_shares[j] = delta_PE + delta_TI + delta_EI;
          bonds[j]       = PRICE_ALIVE - fund_shares[j]*X;
          no_hedge[j]    = no_hedge[j]*exp(r*h) - PRICE_DEAD;

          
          static_hedge[j] = stock_position_static_hedge*X + bonds_static_hedge[j]*exp(r*h) - PRICE_DEAD;
          stock_position_static_hedge = sum(PH_DATA.nominals_sum_PE)*delta_start_PE 
                                      + sum(PH_DATA.nominals_sum_TI)*delta_start_TI
                                      + sum(PH_DATA.nominals_sum_EI)*delta_start_EI;     
          bonds_static_hedge[j] = static_hedge[j] - stock_position_static_hedge*X;

          //std::cout << "PRICE_ALIVE = " << PRICE_ALIVE << std::endl;
          //std::cout << "PRICE_DEAD  = " << PRICE_DEAD << std::endl;
          //std::cout << "COST =" << COST[j] << std::endl;
        }
        else
        { // at expiration T we need to pay the TI and EI dead in [T-h,T] and the PE and EI ALIVE at T
          double nominals_alive = sum(PH_DATA.nominals_sum_PE) + sum(PH_DATA.nominals_sum_EI);
          COST[j]       += PRICE_DEAD + nominals_alive*market_payoff - (bonds[j]*exp(r*h) + fund_shares[j]*X);
          no_hedge[j]    = PRICE_DEAD + nominals_alive*market_payoff - no_hedge[j]*exp(r*h);

          static_hedge[j] = stock_position_static_hedge*X + bonds_static_hedge[j]*exp(r*h) 
                            - PRICE_DEAD  - nominals_alive*market_payoff;
          static_hedge[j] = -static_hedge[j];
          //bonds[j]       = 0.0; //price - delta*X;
          //fund_shares[j] = 0.0; //delta;
        }

      }
      X_vec[j] = X; //X a time T (maturity)

    }
  } //end parallel zone
  //std::cout << "\ntrading_gains:\n" << trading_gains << std::endl;
  //std::cout << "\nfund_shares:\n" << fund_shares << std::endl;
  //double mean_hedging_gains = trading_gains.mean();
  //double std_dev_hedging_gains = sqrt( ((trading_gains -mean_hedging_gains).square()).mean() );
  //std::cout << "expected trading gains = " << mean_hedging_gains    << std::endl;
  //std::cout << "CI95                   = " << 1.96*std_dev_hedging_gains/sqrt(M) << std::endl;
  //std::cout << "std_dev trading gains  = " << std_dev_hedging_gains << std::endl;

  double mean_cost = mean(COST);
  std::pair<double,double> output_var = empirical_var_CI95(COST);
  std::cout << "expected value of cost = " << mean_cost    << std::endl;
  std::cout << "CI95  expected value   = " << 1.96*sqrt(output_var.first/M) << std::endl;
  std::cout << "     var cost = " << output_var.first << std::endl;
  std::cout << "CI95 var cost = " << output_var.second << std::endl;

  Output_hedging output;
  output.no_hedge = no_hedge;
  output.static_hedge = static_hedge;
  output.trading_gains = trading_gains;
  output.hedging_cost = COST;
  output.start_price = price_start_PE+price_start_TI+price_start_EI;
  output.X = X_vec;
  return output;
}



Output_hedging
PETIEI_hedge_cost_paral_mc_MLMC_scenarios(double CRRA_delta, double r, ArrayXd mu, ArrayXd vols, MatrixXd cov_mat, 
									double alpha_penalty, ArrayXd beta, ArrayXd kappa, ArrayXd s,
                  PETIEI_polyciholders PH_DATA, double phi_GM, double m_GM, double b_GM, 
                  double min_rate, double max_rate, double  perc_prem_death, 
                  double T, int N_delta_lvl, int M_delta, Scenarios &XY_scenarios)
{
  // different N for number of times of rebalancing portfolio and computing deltas
  // compute of prices and deltas are all in the MMM
  // In output 3 strategies:
  // 1) no hedging investment on the cash account
  // 2) static hedge
  // 3) continous hedging

  // several classe ages
  // real life scenarios precomputed

  int M = XY_scenarios.X.size();
  int N = XY_scenarios.X[0].size() -1;

  //std::cout << "M = " << M << std::endl;
  //std::cout << "N = " << N << std::endl;

  double price_start_PE, CI_price_start_PE, delta_start_PE, CI_delta_start_PE;
  double price_start_TI, CI_price_start_TI, delta_start_TI, CI_delta_start_TI;
  double price_start_EI, CI_price_start_EI, delta_start_EI, CI_delta_start_EI;

  //X0 = 1.0
  PETIEI_min_max_GM_MMM_mc_MLMC(CRRA_delta, 1.0, r, mu, vols, cov_mat, alpha_penalty, 
                  XY_scenarios.Y[0][0], beta, kappa, s,
                  PH_DATA.nominals_sum_PE, PH_DATA.age_PE, 
                  PH_DATA.nominals_sum_TI, PH_DATA.age_TI,
                  PH_DATA.nominals_sum_EI, PH_DATA.age_EI,
                  phi_GM, m_GM, b_GM,
                  1.0, 1.0, min_rate, max_rate, perc_prem_death, 
                  T, N_delta_lvl, M_delta*100,  // we demand extra precision at the start to fix the price
                  price_start_PE, CI_price_start_PE, delta_start_PE, CI_delta_start_PE,
                  price_start_TI, CI_price_start_TI, delta_start_TI, CI_delta_start_TI,
                  price_start_EI, CI_price_start_EI, delta_start_EI, CI_delta_start_EI);
  
  std::cout << "price_PE in MMM = " << price_start_PE  << std::endl;
  std::cout << "           CI95 = " << CI_price_start_PE  << std::endl;
  std::cout << "price_TI in MMM = " << price_start_TI  << std::endl;
  std::cout << "           CI95 = " << CI_price_start_TI  << std::endl;
  std::cout << "price_EI in MMM = " << price_start_EI  << std::endl;
  std::cout << "           CI95 = " << CI_price_start_EI  << std::endl;
  std::cout << "delta_PE at time 0 = " << delta_start_PE  << std::endl;
  std::cout << "              CI95 = " << CI_delta_start_PE  << std::endl;
  std::cout << "delta_TI at time 0 = " << delta_start_TI  << std::endl;
  std::cout << "              CI95 = " << CI_delta_start_TI  << std::endl;
  std::cout << "delta_EI at time 0 = " << delta_start_EI  << std::endl;
  std::cout << "              CI95 = " << CI_delta_start_EI  << std::endl;
  

  double price_start = price_start_PE + price_start_TI + price_start_EI;
  double delta_start = delta_start_PE + delta_start_TI + delta_start_EI;

  std::cout << "price_start = " << price_start   << std::endl;
  std::cout << "delta_start = " << delta_start   << std::endl;


  double bond_start = price_start  - delta_start; //X0=1.0
  std::vector<double> bonds(M, bond_start);

  std::vector<double> fund_shares(M, delta_start);

  std::vector<double> trading_gains(M, 0.0);
  std::vector<double> X_vec(M, 0.0);
  std::vector<double> COST(M, 0.0);

  std::vector<double> no_hedge(M, price_start);
  std::vector<double> static_hedge(M);
  std::vector<double> bonds_static_hedge(M,bond_start);

  
  #pragma omp parallel num_threads(8) firstprivate(PH_DATA) //16 threads possible but bottlenecked by memory bandwith
  { 
    double h = T / double(N);

    auto& gen = parallel_thread_rng<__LINE__>();
    boost::random::uniform_real_distribution<double> U(0, 1); // uni dist over (0,1)

    double price_PE, CI_price_PE, delta_PE, CI_delta_PE,
           price_TI, CI_price_TI, delta_TI, CI_delta_TI,
           price_EI, CI_price_EI, delta_EI, CI_delta_EI;

    std::vector<double> start_age_PE = PH_DATA.age_PE;
    std::vector<double> start_age_TI = PH_DATA.age_TI;
    std::vector<double> start_age_EI = PH_DATA.age_EI;

    #pragma omp for
    for(int j=0; j<M; j++)
    { 
      double stock_position_static_hedge = delta_start;
      double TtoMat = T;
      
      // resetting for each simulations the good starting values for the ph
      // ages, nominals_sum and live status
      PH_DATA.age_PE = start_age_PE;
      PH_DATA.age_TI = start_age_TI;
      PH_DATA.age_EI = start_age_EI;
      for(int q=0; q<PH_DATA.is_alive_PE.size(); q++)
      { 
        PH_DATA.nominals_sum_PE[q] = PH_DATA.nominals_PE[q].sum();
        PH_DATA.is_alive_PE[q].setConstant(true);
      } 
      for(int q=0; q<PH_DATA.is_alive_TI.size(); q++)
      {
        PH_DATA.nominals_sum_TI[q] = PH_DATA.nominals_TI[q].sum();
        PH_DATA.is_alive_TI[q].setConstant(true);
      }
      for(int q=0; q<PH_DATA.is_alive_EI.size(); q++)
      {
        PH_DATA.nominals_sum_EI[q] = PH_DATA.nominals_EI[q].sum();
        PH_DATA.is_alive_EI[q].setConstant(true);
      }


      for(int i=0; i<N; i++)
      { 
        //calculating the hedge gains
        trading_gains[j] += (XY_scenarios.X[j][i+1] - XY_scenarios.X[j][i]) * fund_shares[j];
        //std::cout << "trading_gains[j] = " << trading_gains[j]   << std::endl;


        
        // updating status of the Policyholders and calculating the claims to be paid
        double PRICE_DEAD = 0.0;
        for(int q=0; q<PH_DATA.nominals_PE.size(); q++)
        { 
          double survival_proba = GM_survival_proba(PH_DATA.age_PE[q], phi_GM, m_GM, b_GM, h);
          for(int l=0; l<PH_DATA.is_alive_PE[q].size(); l++)
          {
            if(PH_DATA.is_alive_PE[q](l)==true)
            { 
              if(U(gen) > survival_proba)
              {
                PH_DATA.is_alive_PE[q](l) = false;
                PH_DATA.nominals_sum_PE[q] -= PH_DATA.nominals_PE[q](l);
              } 
            }
          }
        }
        for(int q=0; q<PH_DATA.nominals_TI.size(); q++)
        {
          double survival_proba = GM_survival_proba(PH_DATA.age_TI[q], phi_GM, m_GM, b_GM, h);
          for(int l=0; l<PH_DATA.is_alive_TI[q].size(); l++)
          {
            if(PH_DATA.is_alive_TI[q](l)==true)
            {
              if(U(gen) > survival_proba)
              {
                PH_DATA.is_alive_TI[q](l) = false;
                PH_DATA.nominals_sum_TI[q] -= PH_DATA.nominals_TI[q](l);
                //TI_toPay.push(std::make_pair(q,l)); // add the (q,l) pair ph to the list to be payed
                PRICE_DEAD                 += PH_DATA.nominals_TI[q](l);
              } 
            }
          }
        }

        for(int q=0; q<PH_DATA.nominals_EI.size(); q++)
        {
          double survival_proba = GM_survival_proba(PH_DATA.age_EI[q], phi_GM, m_GM, b_GM, h);
          for(int l=0; l<PH_DATA.is_alive_EI[q].size(); l++)
          {
            if(PH_DATA.is_alive_EI[q](l)==true)
            {
              if(U(gen) > survival_proba)
              {
                PH_DATA.is_alive_EI[q](l) = false;
                PH_DATA.nominals_sum_EI[q] -= PH_DATA.nominals_EI[q](l);
                //EI_toPay.push(std::make_pair(q,l)); // add the (q,l) pair ph to the list to be payed
                PRICE_DEAD += perc_prem_death*PH_DATA.nominals_EI[q](l);
              } 
            }
          }
        }

        TtoMat         -= h;
        PH_DATA.age_PE += h;
        PH_DATA.age_TI += h;
        PH_DATA.age_EI += h;
        double min_gua = exp(min_rate*(T-TtoMat)); //X0=1 starting value
        double max_gua = exp(max_rate*(T-TtoMat)); //X0=1

        
        double market_payoff;
        if      (XY_scenarios.X[j][i+1] < min_gua) market_payoff = min_gua;
        else if (XY_scenarios.X[j][i+1] > max_gua) market_payoff = max_gua;
        else                  market_payoff = XY_scenarios.X[j][i+1];
        PRICE_DEAD *= market_payoff;

        if(i<N-1)
        { 
          PETIEI_min_max_GM_MMM_mc_MLMC(CRRA_delta, XY_scenarios.X[j][i+1], r, mu, vols, cov_mat, 
                        alpha_penalty, XY_scenarios.Y[j][i+1], beta, kappa, s,
                        PH_DATA.nominals_sum_PE, PH_DATA.age_PE, 
                        PH_DATA.nominals_sum_TI, PH_DATA.age_TI,
                        PH_DATA.nominals_sum_EI, PH_DATA.age_EI,
                        phi_GM, m_GM, b_GM,
                        min_gua, max_gua, min_rate, max_rate, perc_prem_death, 
                        TtoMat, N_delta_lvl, M_delta,
                        price_PE, CI_price_PE, delta_PE, CI_delta_PE,
                        price_TI, CI_price_TI, delta_TI, CI_delta_TI,
                        price_EI, CI_price_EI, delta_EI, CI_delta_EI);
          
          //one could optimize n_delta_lvl and M_delta going forward in time too much precision at the end
          double PRICE_ALIVE = price_PE + price_TI + price_EI;
          

          COST[j]       += PRICE_ALIVE + PRICE_DEAD - (bonds[j]*exp(r*h) + fund_shares[j]*XY_scenarios.X[j][i+1]);
          fund_shares[j] = delta_PE + delta_TI + delta_EI;
          bonds[j]       = PRICE_ALIVE - fund_shares[j]*XY_scenarios.X[j][i+1];
          no_hedge[j]    = no_hedge[j]*exp(r*h) - PRICE_DEAD;

          
          static_hedge[j] = stock_position_static_hedge*XY_scenarios.X[j][i+1] + bonds_static_hedge[j]*exp(r*h) - PRICE_DEAD;
          stock_position_static_hedge = sum(PH_DATA.nominals_sum_PE)*delta_start_PE 
                                      + sum(PH_DATA.nominals_sum_TI)*delta_start_TI
                                      + sum(PH_DATA.nominals_sum_EI)*delta_start_EI;     
          bonds_static_hedge[j] = static_hedge[j] - stock_position_static_hedge*XY_scenarios.X[j][i+1];

          //std::cout << "PRICE_ALIVE = " << PRICE_ALIVE << std::endl;
          //std::cout << "PRICE_DEAD  = " << PRICE_DEAD << std::endl;
          //std::cout << "COST =" << COST[j] << std::endl;
        }
        else
        { // at expiration T we need to pay the TI and EI dead in [T-h,T] and the PE and EI ALIVE at T
          double nominals_alive = sum(PH_DATA.nominals_sum_PE) + sum(PH_DATA.nominals_sum_EI);
          COST[j]       += PRICE_DEAD + nominals_alive*market_payoff - (bonds[j]*exp(r*h) + fund_shares[j]*XY_scenarios.X[j][i+1]);
          no_hedge[j]    = PRICE_DEAD + nominals_alive*market_payoff - no_hedge[j]*exp(r*h);

          static_hedge[j] = stock_position_static_hedge*XY_scenarios.X[j][i+1] + bonds_static_hedge[j]*exp(r*h) 
                            - PRICE_DEAD  - nominals_alive*market_payoff;
          static_hedge[j] = -static_hedge[j];
          //bonds[j]       = 0.0; //price - delta*XY_scenarios.X[j][i+1];
          //fund_shares[j] = 0.0; //delta;
        }

      }
      X_vec[j] = XY_scenarios.X[j][N]; //X a time T (maturity)

    }
  } //end parallel zone
  //std::cout << "\ntrading_gains:\n" << trading_gains << std::endl;
  //std::cout << "\nfund_shares:\n" << fund_shares << std::endl;
  //double mean_hedging_gains = trading_gains.mean();
  //double std_dev_hedging_gains = sqrt( ((trading_gains -mean_hedging_gains).square()).mean() );
  //std::cout << "expected trading gains = " << mean_hedging_gains    << std::endl;
  //std::cout << "CI95                   = " << 1.96*std_dev_hedging_gains/sqrt(M) << std::endl;
  //std::cout << "std_dev trading gains  = " << std_dev_hedging_gains << std::endl;

  double mean_cost = mean(COST);
  std::pair<double,double> output_var = empirical_var_CI95(COST);
  std::cout << "expected value of cost = " << mean_cost    << std::endl;
  std::cout << "CI95  expected value   = " << 1.96*sqrt(output_var.first/M) << std::endl;
  std::cout << "     var cost = " << output_var.first << std::endl;
  std::cout << "CI95 var cost = " << output_var.second << std::endl;

  Output_hedging output;
  output.no_hedge = no_hedge;
  output.static_hedge = static_hedge;
  output.trading_gains = trading_gains;
  output.hedging_cost = COST;
  output.start_price = price_start_PE+price_start_TI+price_start_EI;
  output.X = X_vec;
  return output;
}


// General-model scenario hedging with ordinary Monte Carlo valuations only.
// The comparison benchmark can hold the grid fixed to match MLMC.
// N_delta is the initial pricing interval count, reduced with remaining duration.
template <class CarbonModel, class NormalSampler = As241NormalSampler>
Output_hedging
PETIEI_hedge_cost_paral_mc_scenarios_general_model(
                  double CRRA_delta, double r, ArrayXd mu, ArrayXd vols, MatrixXd cov_mat, 
									double alpha_penalty, ArrayXd beta, ArrayXd kappa, ArrayXd s,
                  PETIEI_polyciholders PH_DATA, double phi_GM, double m_GM, double b_GM, 
                  double min_rate, double max_rate, double  perc_prem_death, 
                  double T, int N_delta, int M_delta, Scenarios &XY_scenarios,
                  bool fixed_pricing_grid = false)
{
  // different N for number of times of rebalancing portfolio and computing deltas
  // compute of prices and deltas are all in the MMM
  // In output 3 strategies:
  // 1) no hedging investment on the cash account
  // 2) static hedge
  // 3) continous hedging

  // several classe ages
  // real life scenarios precomputed

  if (mu.size() != 4 || vols.size() != 4 || beta.size() != 4 ||
      kappa.size() != 4 || s.size() != 4 ||
      cov_mat.rows() != 4 || cov_mat.cols() != 4) {
    throw std::invalid_argument("Four-factor hedge requires four-dimensional model inputs");
  }
  if (XY_scenarios.X.empty() || XY_scenarios.Y.size() != XY_scenarios.X.size()) {
    throw std::invalid_argument("Scenario paths are missing or inconsistent");
  }
  if (XY_scenarios.X[0].size() < 2 || N_delta < 1 || M_delta < 2) {
    throw std::invalid_argument("Hedging requires a nonempty time grid, N_delta >= 1, and M_delta >= 2");
  }
  const int M = XY_scenarios.X.size();
  const int N = XY_scenarios.X[0].size() - 1;
  for (int j = 0; j < M; ++j) {
    if (XY_scenarios.X[j].size() != static_cast<size_t>(N + 1) ||
        XY_scenarios.Y[j].size() != static_cast<size_t>(N + 1)) {
      throw std::invalid_argument("Scenario paths have inconsistent lengths");
    }
    for (const auto &carbon_state : XY_scenarios.Y[j]) {
      if (carbon_state.size() != 4) {
        throw std::invalid_argument("Four-factor hedge requires four-dimensional scenario states");
      }
    }
  }

  const Eigen::Vector4d mu4 = mu.matrix();
  const Eigen::Vector4d vols4 = vols.matrix();
  const Eigen::Matrix4d cov4 = cov_mat;
  const Eigen::Vector4d beta4 = beta.matrix();
  const Eigen::Vector4d kappa4 = kappa.matrix();
  const Eigen::Vector4d s4 = s.matrix();
  const Eigen::Vector4d Y_start = XY_scenarios.Y[0][0].matrix();


  double price_start_PE, var_price_start_PE, delta_start_PE, var_delta_start_PE;
  double price_start_TI, var_price_start_TI, delta_start_TI, var_delta_start_TI;
  double price_start_EI, var_price_start_EI, delta_start_EI, var_delta_start_EI;

  //X0 = 1.0
  PETIEI_min_max_GM_MMM_mc_general_model<CarbonModel, NormalSampler>(
                  CRRA_delta, 1.0, r, mu4, vols4, cov4, alpha_penalty,
                  Y_start, beta4, kappa4, s4,
                  PH_DATA.nominals_sum_PE, PH_DATA.age_PE, 
                  PH_DATA.nominals_sum_TI, PH_DATA.age_TI,
                  PH_DATA.nominals_sum_EI, PH_DATA.age_EI,
                  phi_GM, m_GM, b_GM,
                  1.0, 1.0, min_rate, max_rate, perc_prem_death, 
                  T, N_delta, M_delta*100,  // we demand extra precision at the start to fix the price
                  price_start_PE, var_price_start_PE, delta_start_PE, var_delta_start_PE,
                  price_start_TI, var_price_start_TI, delta_start_TI, var_delta_start_TI,
                  price_start_EI, var_price_start_EI, delta_start_EI, var_delta_start_EI);
  
  std::cout << "price_PE in MMM = " << price_start_PE  << std::endl;
  std::cout << "           CI95 = " << 1.96*sqrt(var_price_start_PE/(100.0*M_delta)) << std::endl;
  std::cout << "price_TI in MMM = " << price_start_TI  << std::endl;
  std::cout << "           CI95 = " << 1.96*sqrt(var_price_start_TI/(100.0*M_delta)) << std::endl;
  std::cout << "price_EI in MMM = " << price_start_EI  << std::endl;
  std::cout << "           CI95 = " << 1.96*sqrt(var_price_start_EI/(100.0*M_delta)) << std::endl;
  std::cout << "delta_PE at time 0 = " << delta_start_PE  << std::endl;
  std::cout << "              CI95 = " << 1.96*sqrt(var_delta_start_PE/(100.0*M_delta)) << std::endl;
  std::cout << "delta_TI at time 0 = " << delta_start_TI  << std::endl;
  std::cout << "              CI95 = " << 1.96*sqrt(var_delta_start_TI/(100.0*M_delta)) << std::endl;
  std::cout << "delta_EI at time 0 = " << delta_start_EI  << std::endl;
  std::cout << "              CI95 = " << 1.96*sqrt(var_delta_start_EI/(100.0*M_delta)) << std::endl;
  

  double price_start = price_start_PE + price_start_TI + price_start_EI;
  double delta_start = delta_start_PE + delta_start_TI + delta_start_EI;

  std::cout << "price_start = " << price_start   << std::endl;
  std::cout << "delta_start = " << delta_start   << std::endl;


  double bond_start = price_start  - delta_start; //X0=1.0
  std::vector<double> bonds(M, bond_start);

  std::vector<double> fund_shares(M, delta_start);

  std::vector<double> trading_gains(M, 0.0);
  std::vector<double> X_vec(M, 0.0);
  std::vector<double> COST(M, 0.0);

  std::vector<double> no_hedge(M, price_start);
  std::vector<double> static_hedge(M);
  std::vector<double> bonds_static_hedge(M,bond_start);

  
  #pragma omp parallel num_threads(16) firstprivate(PH_DATA) 
  { //16 threads possible but bottlenecked by memory bandwith

    double h = T / double(N);
    double ratio_N = N / double(N_delta);

    auto& gen = parallel_thread_rng<__LINE__>();
    boost::random::uniform_real_distribution<double> U(0, 1); // uni dist over (0,1)

    double price_PE, var_price_PE, delta_PE, var_delta_PE,
           price_TI, var_price_TI, delta_TI, var_delta_TI,
           price_EI, var_price_EI, delta_EI, var_delta_EI;

    std::vector<double> start_age_PE = PH_DATA.age_PE;
    std::vector<double> start_age_TI = PH_DATA.age_TI;
    std::vector<double> start_age_EI = PH_DATA.age_EI;

    #pragma omp for
    for(int j=0; j<M; j++)
    { 
      double stock_position_static_hedge = delta_start;
      double TtoMat = T;
      
      // resetting for each simulations the good starting values for the ph
      // ages, nominals_sum and live status
      PH_DATA.age_PE = start_age_PE;
      PH_DATA.age_TI = start_age_TI;
      PH_DATA.age_EI = start_age_EI;
      for(int q=0; q<PH_DATA.is_alive_PE.size(); q++)
      { 
        PH_DATA.nominals_sum_PE[q] = PH_DATA.nominals_PE[q].sum();
        PH_DATA.is_alive_PE[q].setConstant(true);
      } 
      for(int q=0; q<PH_DATA.is_alive_TI.size(); q++)
      {
        PH_DATA.nominals_sum_TI[q] = PH_DATA.nominals_TI[q].sum();
        PH_DATA.is_alive_TI[q].setConstant(true);
      }
      for(int q=0; q<PH_DATA.is_alive_EI.size(); q++)
      {
        PH_DATA.nominals_sum_EI[q] = PH_DATA.nominals_EI[q].sum();
        PH_DATA.is_alive_EI[q].setConstant(true);
      }


      for(int i=0; i<N; i++)
      { 
        //calculating the hedge gains
        trading_gains[j] += (XY_scenarios.X[j][i+1] - XY_scenarios.X[j][i]) * fund_shares[j];


        // updating status of the Policyholders and calculating the claims to be paid
        double PRICE_DEAD = 0.0;
        for(int q=0; q<PH_DATA.nominals_PE.size(); q++)
        { 
          double survival_proba = GM_survival_proba(PH_DATA.age_PE[q], phi_GM, m_GM, b_GM, h);
          for(int l=0; l<PH_DATA.is_alive_PE[q].size(); l++)
          {
            if(PH_DATA.is_alive_PE[q](l)==true)
            { 
              if(U(gen) > survival_proba)
              {
                PH_DATA.is_alive_PE[q](l) = false;
                PH_DATA.nominals_sum_PE[q] -= PH_DATA.nominals_PE[q](l);
              } 
            }
          }
        }
        for(int q=0; q<PH_DATA.nominals_TI.size(); q++)
        {
          double survival_proba = GM_survival_proba(PH_DATA.age_TI[q], phi_GM, m_GM, b_GM, h);
          for(int l=0; l<PH_DATA.is_alive_TI[q].size(); l++)
          {
            if(PH_DATA.is_alive_TI[q](l)==true)
            {
              if(U(gen) > survival_proba)
              {
                PH_DATA.is_alive_TI[q](l) = false;
                PH_DATA.nominals_sum_TI[q] -= PH_DATA.nominals_TI[q](l);
                //TI_toPay.push(std::make_pair(q,l)); // add the (q,l) pair ph to the list to be payed
                PRICE_DEAD                 += PH_DATA.nominals_TI[q](l);
              } 
            }
          }
        }

        for(int q=0; q<PH_DATA.nominals_EI.size(); q++)
        {
          double survival_proba = GM_survival_proba(PH_DATA.age_EI[q], phi_GM, m_GM, b_GM, h);
          for(int l=0; l<PH_DATA.is_alive_EI[q].size(); l++)
          {
            if(PH_DATA.is_alive_EI[q](l)==true)
            {
              if(U(gen) > survival_proba)
              {
                PH_DATA.is_alive_EI[q](l) = false;
                PH_DATA.nominals_sum_EI[q] -= PH_DATA.nominals_EI[q](l);
                //EI_toPay.push(std::make_pair(q,l)); // add the (q,l) pair ph to the list to be payed
                PRICE_DEAD += perc_prem_death*PH_DATA.nominals_EI[q](l);
              } 
            }
          }
        }

        TtoMat         -= h;
        PH_DATA.age_PE += h;
        PH_DATA.age_TI += h;
        PH_DATA.age_EI += h;
        double min_gua = exp(min_rate*(T-TtoMat)); //X0=1 starting value
        double max_gua = exp(max_rate*(T-TtoMat)); //X0=1

        
        double market_payoff;
        if      (XY_scenarios.X[j][i+1] < min_gua) market_payoff = min_gua;
        else if (XY_scenarios.X[j][i+1] > max_gua) market_payoff = max_gua;
        else                  market_payoff = XY_scenarios.X[j][i+1];
        PRICE_DEAD *= market_payoff;

        if(i<N-1)
        { 
          const Eigen::Vector4d Y_current = XY_scenarios.Y[j][i+1].matrix();
          PETIEI_min_max_GM_MMM_mc_general_model<CarbonModel, NormalSampler>(
                        CRRA_delta, XY_scenarios.X[j][i+1], r, mu4, vols4, cov4,
                        alpha_penalty, Y_current, beta4, kappa4, s4,
                        PH_DATA.nominals_sum_PE, PH_DATA.age_PE, 
                        PH_DATA.nominals_sum_TI, PH_DATA.age_TI,
                        PH_DATA.nominals_sum_EI, PH_DATA.age_EI,
                        phi_GM, m_GM, b_GM,
                        min_gua, max_gua, min_rate, max_rate, perc_prem_death, 
                        TtoMat,
                        fixed_pricing_grid ? N_delta : std::max(8, static_cast<int>(std::ceil((N-i-1)/ratio_N))),
                        M_delta,
                        price_PE, var_price_PE, delta_PE, var_delta_PE,
                        price_TI, var_price_TI, delta_TI, var_delta_TI,
                        price_EI, var_price_EI, delta_EI, var_delta_EI);
          
          double PRICE_ALIVE = price_PE + price_TI + price_EI;
          

          COST[j]       += PRICE_ALIVE + PRICE_DEAD - (bonds[j]*exp(r*h) + fund_shares[j]*XY_scenarios.X[j][i+1]);
          fund_shares[j] = delta_PE + delta_TI + delta_EI;
          bonds[j]       = PRICE_ALIVE - fund_shares[j]*XY_scenarios.X[j][i+1];
          
          no_hedge[j]    = no_hedge[j]*exp(r*h) - PRICE_DEAD;

          
          static_hedge[j] = stock_position_static_hedge*XY_scenarios.X[j][i+1] + bonds_static_hedge[j]*exp(r*h) - PRICE_DEAD;
          stock_position_static_hedge = sum(PH_DATA.nominals_sum_PE)*delta_start_PE 
                                      + sum(PH_DATA.nominals_sum_TI)*delta_start_TI
                                      + sum(PH_DATA.nominals_sum_EI)*delta_start_EI;     
          bonds_static_hedge[j] = static_hedge[j] - stock_position_static_hedge*XY_scenarios.X[j][i+1];

        }
        else
        { // at expiration T we need to pay the TI and EI dead in [T-h,T] and the PE and EI ALIVE at T
          double nominals_alive = sum(PH_DATA.nominals_sum_PE) + sum(PH_DATA.nominals_sum_EI);
          COST[j]       += PRICE_DEAD + nominals_alive*market_payoff - (bonds[j]*exp(r*h) + fund_shares[j]*XY_scenarios.X[j][i+1]);
          no_hedge[j]    = PRICE_DEAD + nominals_alive*market_payoff - no_hedge[j]*exp(r*h);

          static_hedge[j] = stock_position_static_hedge*XY_scenarios.X[j][i+1] + bonds_static_hedge[j]*exp(r*h) 
                            - PRICE_DEAD  - nominals_alive*market_payoff;
          static_hedge[j] = -static_hedge[j];
        }

      }
      X_vec[j] = XY_scenarios.X[j][N]; //X a time T (maturity)

    }
  } //end parallel zone

  double mean_cost = mean(COST);
  std::pair<double,double> output_var = empirical_var_CI95(COST);
  std::cout << "expected value of cost = " << mean_cost    << std::endl;
  std::cout << "CI95  expected value   = " << 1.96*sqrt(output_var.first/M) << std::endl;
  std::cout << "     var cost = " << output_var.first << std::endl;
  std::cout << "CI95 var cost = " << output_var.second << std::endl;

  Output_hedging output;
  output.no_hedge = no_hedge;
  output.static_hedge = static_hedge;
  output.trading_gains = trading_gains;
  output.hedging_cost = COST;
  output.start_price = price_start_PE+price_start_TI+price_start_EI;
  output.X = X_vec;
  return output;
}




std::pair< std::vector<std::vector<double> >, std::vector<std::vector<double> > >
weights_evolution(double CRRA_delta, double r, ArrayXd mu, ArrayXd vols, MatrixXd cov_mat, 
									double alpha_penalty, ArrayXd Y0, ArrayXd beta, ArrayXd kappa, ArrayXd s,
                  double T, int N)
{ 
  auto dim = (beta.size()); 
  std::vector<double> weight(N+1,0.0); 
  std::vector<std::vector<double> > weights(dim,weight);
  std::vector<std::vector<double> > C_vec(dim,weight);
  VectorXd mu_mod = (mu-r).matrix();
	ArrayXd vols_sq_penalised = alpha_penalty*vols.square();
	MatrixXd d_cov_mat = CRRA_delta * cov_mat;
	double h = T/double(N);
	ArrayXd C0(dim), C1(dim), C2(dim), C3(dim);
	prm_NV_CIR(h, beta, kappa, s, C0, C1, C2, C3);

  boost::random::xoshiro256pp gen(boost::random::random_device{}());    
  boost::random::uniform_real_distribution<double> U(0, 1); // uni dist over (0,1)

  ArrayXd e_CI = Y0 * vols_sq_penalised;

  ArrayXd Y = Y0;
	ArrayXd G(dim); 

  d_cov_mat.diagonal().array() += e_CI;
	Eigen::LLT<Eigen::MatrixXd> llt_solver_start(d_cov_mat);
	VectorXd pi_start = llt_solver_start.solve(mu_mod);
	d_cov_mat.diagonal().array() -= e_CI;
  for(int l=0;l<dim;l++)
  {
    weights[l][0] = pi_start(l);
    C_vec[l][0]   = Y0(l);
  } 
  for(int i=0;i<N;i++)
  {
    for(int l=0;l<dim;l++) G(l) = normcdfinv_as241(U(gen));
    NV_CIR(Y, C0, C1, C2, C3, G);

    e_CI = Y * vols_sq_penalised;
    d_cov_mat.diagonal().array() += e_CI;
    Eigen::LLT<Eigen::MatrixXd> llt_solver(d_cov_mat);
    VectorXd pi = llt_solver.solve(mu_mod);
    d_cov_mat.diagonal().array() -= e_CI;
    for(int l=0;l<dim;l++)
    {
      weights[l][i+1] = pi(l);
      C_vec[l][i+1]   =  Y(l);
    } 
  }

  return std::make_pair(weights, C_vec);
}



#ifndef HEDGE_COST_NO_MAIN
int main()
{   
  //set with 4 risky assets
  ArrayXd mu(4); 
  mu <<    0.25, 0.15, 0.1,  0.08;
  ArrayXd vols(4);
  vols <<  0.30, 0.25, 0.20, 0.16;
  ArrayXd beta(4);  // long-run mean of the carbon intensities
  beta  << 2500, 2000, 1500 , 500;
  //beta  << 0.5, 0.4, 0.3, 0.4;
  ArrayXd kappa(4);
  //kappa << 0.2, 0.2,  0.2, 0.2;
  kappa << 0.05, 0.05,  0.05, 0.05;
  ArrayXd sigma(4);
  //sigma << 0.2, 0.2, 0.2, 0.2;
  sigma << 3.0, 3.0, 3.0, 3.0;
  /* sigma << sqrt(2*beta(0)*kappa(0)-1e-5), 
           sqrt(2*beta(1)*kappa(1)-1e-5),
           sqrt(2*beta(2)*kappa(2)-1e-5), 
           sqrt(2*beta(3)*kappa(3)-1e-5); */
  ArrayXd Y0(4); // starting value for the carbon intensities
  Y0     << 5000, 4000, 3000, 1000;
  MatrixXd corr_mat(4,4);
  corr_mat << 1.,	    0.4397,	0.3900,	0.3168,
              0.4397,	1.,	    0.2954,	0.3261,
              0.3900,	0.2954,	1.,	    0.3134,
              0.3168,	0.3261,	0.3134,	1.    ;

  //set with 8 risky assets
  /* ArrayXd mu(8); 
  mu << 0.25, 0.15, 0.1, 0.08, 0.25, 0.15, 0.1, 0.08;
  ArrayXd vols(8);
  vols <<  0.35, 0.25, 0.20, 0.15, 0.35, 0.25, 0.20, 0.15;
  ArrayXd beta(8);
  beta  << 0.3, 0.25, 0.2 , 0.3, 0.3, 0.25, 0.2 , 0.3;
  ArrayXd kappa(8);
  kappa << 0.1, 0.1,  0.1, 0.1, 0.1, 0.1,  0.1, 0.1;
  //kappa << 0.3, 0.25, 0.2 , 0.3, 0.3, 0.25, 0.2 , 0.3;
  ArrayXd sigma(8);
  sigma << 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05;
  //sigma << 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0;
  ArrayXd Y0(8); // starting value for the carbon intensities
  Y0     << 0.5, 0.4, 0.3, 0.4, 0.5, 0.4, 0.3, 0.4;
  MatrixXd corr_mat(8,8);
  corr_mat << 1.,	    0.4397,	0.3900,	0.3168, 0.3761,	0.3552,	0.3023,	0.2779,
              0.4397,	1.,	    0.2954,	0.3261, 0.1881,	0.3664,	0.1647,	0.2569,
              0.3900,	0.2954,	1.,	    0.3134, 0.2994,	0.3403,	0.3040,	0.2991,
              0.3168,	0.3261,	0.3134,	1.    , 0.2134,	0.2926,	0.2292,	0.3633,
              0.3761,	0.1881,	0.2994,	0.2134,	1.    ,	0.3194,	0.2642,	0.2980,
              0.3552,	0.3664,	0.3403,	0.2926,	0.3194,	1.,	    0.2585,	0.3626,
              0.3023,	0.1647,	0.3040,	0.2292,	0.2642,	0.2585,	1.    ,	0.3201,
              0.2779,	0.2569,	0.2991,	0.3633,	0.2980,	0.3626,	0.3201,	1.    ; */

  //computing covariance matrix
  MatrixXd cov_mat =  corr_mat.array() * (vols.matrix() * vols.matrix().transpose()).array();

  //std::cout << cov_mat << std::endl;

  double T  = 20;

  double  r = 0.05;
  double X0 = 1.0;


  //METTERE MIN E MAX RATE COME STANDARD OVUNQUE

  //double min_gua = X0;
  double min_gua = exp(r*T)*X0;
  double max_gua = exp(10*r*T)*X0;

  double min_rate = r;
  double max_rate = 10*r;

  double perc_prem_death = 1.0; // For Ei new contracts

  //std::cout << "max_rate = " << max_rate << std::endl;                                

  double CRRA_delta    = 1.0;
  double alpha_penalty = 0.0025;
  //double alpha_penalty = 0.0;

  double start_age = 60;
  //male
  double phi_GM  =  0.0027900;
  double m_GM    = 79.6921211;
  double b_GM    = 1.0 / 11.5818911; // new b_GM is the reciprocal of the old b_GM

  //female
  /* double phi_GM =  0.0041959;
  double m_GM    = 86.2438644;
  double b_GM    =  1.0 / 9.6511506; */

  double price, var_price, delta, var_delta;



  std::chrono::_V2::system_clock::time_point start, finish;
  std::chrono::duration<double> elapsed;

  //////////////////////////////////////////////////////////
  ////////////////////// MLMC TEST ////////////////////////
  ////////////////////////////////////////////////////////

  // ** PARAMETERS ** 
  /* const int T_MLMC = 20;
  const int N_lvl_MLMC  = 8;
  const int start_lvl = 0;

  const int N_final = int(pow(2,start_lvl+N_lvl_MLMC-1));

  int M_tun_MLMC = 100000;

  double min_gua_MLMC = exp(min_rate*T_MLMC)*X0;
  double max_gua_MLMC = exp(max_rate*T_MLMC)*X0;

  std::vector<double> RES_MLMC( N_lvl_MLMC);
  std::vector<double> DELTA_MLMC(N_lvl_MLMC);
  std::vector<double> RES_VAR_MLMC( N_lvl_MLMC);
  std::vector<double> DELTA_VAR_MLMC(N_lvl_MLMC);



  std::vector<double> _1overN_MLMC(N_lvl_MLMC);

  std::vector<double> log1overN_MLMC(N_lvl_MLMC-1);
  std::vector<double> log_PRICE_ERR_MLMC(N_lvl_MLMC-1);
  std::vector<double> log_DELTA_ERR_MLMC(N_lvl_MLMC-1);

  int N_MLMC = 1.0;
  for(int q = 0; q < start_lvl; ++q) N_MLMC *= 2;
  for(int l=0; l<N_lvl_MLMC; l++) 
  { 
    _1overN_MLMC[l]   =     1.0 / N_MLMC;
    if(l<N_lvl_MLMC-1)  log1overN_MLMC[l] = log(1.0 / N_MLMC);
    N_MLMC *= 2;
  }

  double price_ref, var_price_ref, delta_ref, var_delta_ref; */

  //PE MLMC
  /* std::cout << "N_final = " << N_final  << std::endl;
  start = std::chrono::high_resolution_clock::now();
  PE_min_max_GM_MMM(CRRA_delta, X0, r, mu, vols, cov_mat, 
              alpha_penalty, Y0, beta, kappa, sigma,
              start_age, phi_GM, m_GM, b_GM,  
              min_gua_MLMC, max_gua_MLMC,
              T_MLMC, N_final*4, M_tun_MLMC,
              price_ref, var_price_ref, delta_ref, var_delta_ref);
  finish = std::chrono::high_resolution_clock::now();
  elapsed = finish - start;
  std::cout << "execution time     = " << elapsed.count() << std::endl;


  start = std::chrono::high_resolution_clock::now();
  PE_min_max_GM_MMM_MLMC(CRRA_delta, X0, r, mu, vols, cov_mat, 
              alpha_penalty, Y0, beta, kappa, sigma,
              start_age, phi_GM, m_GM, b_GM,  
              min_gua_MLMC, max_gua_MLMC,
              T_MLMC, start_lvl, N_lvl_MLMC, M_tun_MLMC,
              RES_MLMC, RES_VAR_MLMC, DELTA_MLMC, DELTA_VAR_MLMC);
  finish = std::chrono::high_resolution_clock::now();
  elapsed = finish - start;
  std::cout << "execution time MLMC = " << elapsed.count() << std::endl; */ 



  //TI MLCM
  /* std::cout << "N_final = " << N_final  << std::endl;
  start = std::chrono::high_resolution_clock::now();
  TI_min_max_GM_MMM(CRRA_delta, X0, r, mu, vols, cov_mat, 
              alpha_penalty, Y0, beta, kappa, sigma,
              start_age, phi_GM, m_GM, b_GM,  
              X0, X0, min_rate, max_rate,
              T_MLMC, N_final, M_tun_MLMC,
              price_ref, var_price_ref, delta_ref, var_delta_ref);
  finish = std::chrono::high_resolution_clock::now();
  elapsed = finish - start;
  std::cout << "execution time     = " << elapsed.count() << std::endl; 


  start = std::chrono::high_resolution_clock::now();
  TI_min_max_GM_MMM_MLMC(CRRA_delta, X0, r, mu, vols, cov_mat, 
              alpha_penalty, Y0, beta, kappa, sigma,
              start_age, phi_GM, m_GM, b_GM,  
              X0, X0, min_rate, max_rate,
              T_MLMC, start_lvl, N_lvl_MLMC, M_tun_MLMC*2,
              RES_MLMC, RES_VAR_MLMC, DELTA_MLMC, DELTA_VAR_MLMC);
  finish = std::chrono::high_resolution_clock::now();
  elapsed = finish - start;
  std::cout << "execution time MLMC = " << elapsed.count() << std::endl; */


  //EI MLCM
  /* std::cout << "N_final = " << N_final  << std::endl;
  start = std::chrono::high_resolution_clock::now();
  EI_min_max_GM_MMM(CRRA_delta, X0, r, mu, vols, cov_mat, 
              alpha_penalty, Y0, beta, kappa, sigma,
              start_age, phi_GM, m_GM, b_GM,  
              X0, X0, min_rate, max_rate, perc_prem_death,
              T_MLMC, N_final, M_tun_MLMC,
              price_ref, var_price_ref, delta_ref, var_delta_ref);
  finish = std::chrono::high_resolution_clock::now();
  elapsed = finish - start;
  std::cout << "execution time     = " << elapsed.count() << std::endl; 


  start = std::chrono::high_resolution_clock::now();
  EI_min_max_GM_MMM_MLMC(CRRA_delta, X0, r, mu, vols, cov_mat, 
              alpha_penalty, Y0, beta, kappa, sigma,
              start_age, phi_GM, m_GM, b_GM,  
              X0, X0, min_rate, max_rate, perc_prem_death,
              T_MLMC, start_lvl, N_lvl_MLMC, M_tun_MLMC*2,
              RES_MLMC, RES_VAR_MLMC, DELTA_MLMC, DELTA_VAR_MLMC);
  finish = std::chrono::high_resolution_clock::now();
  elapsed = finish - start;
  std::cout << "execution time MLMC = " << elapsed.count() << std::endl;


  std::cout << "\nPrice ref = " << price_ref << std::endl;
  std::cout << "CI95  ref = " << 1.96*sqrt(var_price_ref/M_tun_MLMC) << std::endl;
  
  std::vector<double> result(RES_MLMC.size());
  std::cout << "\nMLMC results:\n";
  std::partial_sum(RES_MLMC.begin(), RES_MLMC.end(),result.begin());
  print_vector(result);
  
  for(int i=0;i<N_lvl_MLMC-1; i++) log_PRICE_ERR_MLMC[i] = log(abs(result[i+1]-result[i]));
  
  std::pair<double,double> regr_line_price_err_mlmc = lreg(log1overN_MLMC, log_PRICE_ERR_MLMC);
  std::cout << "\nRegressed slope price errors MLMC = " << regr_line_price_err_mlmc.first << std::endl;

  std::cout << "\ndelta ref = " << delta_ref << std::endl;
  std::cout << "CI95  ref = " << 1.96*sqrt(var_delta_ref/M_tun_MLMC) << std::endl;
  std::cout << "\nMLMC delta:\n";
  std::partial_sum(DELTA_MLMC.begin(), DELTA_MLMC.end(), result.begin());
  print_vector(result);

  for(int i=0;i<N_lvl_MLMC-1; i++) log_DELTA_ERR_MLMC[i] = log(abs(result[i+1]-result[i]));
  

  std::pair<double,double> regr_line_delta_err_mlmc = lreg(log1overN_MLMC, log_DELTA_ERR_MLMC);
  std::cout << "\nRegressed slope delta errors MLMC = " << regr_line_delta_err_mlmc.first << std::endl;


  print_vector(RES_VAR_MLMC);
  std::pair<double,double> regr_line_var_corr_mlmc = lreg(_1overN_MLMC, RES_VAR_MLMC, true);
  std::cout << "\nRegressed slope variance corrections MLMC = " << regr_line_var_corr_mlmc.first << std::endl;

  std::pair<double,double> regr_line_var_delta_corr_mlmc = lreg(_1overN_MLMC, DELTA_VAR_MLMC, true);
  std::cout << "\nRegressed slope variance delta corrections MLMC = " << regr_line_var_delta_corr_mlmc.first << std::endl; */
  


  

  ///////////////////////////////////
  //////// CONVERGENCE TEST ////////
  /////////////////////////////////

  // ** PARAMETERS ** 
  /* const double start_age_conv = 40;
  const int l_conv = 5;
  const int T_conv = 20;
  const int N_conv[ l_conv] = {1,2,4,  8, 16};
  const int N2_conv[l_conv] = {2,4,8, 16, 32};

  int M_conv = 10000000;

  double min_gua_conv = exp(r*T_conv)*X0;
  double max_gua_conv = exp(10*r*T_conv)*X0;

  std::vector<double> RESN_conv( l_conv);
  std::vector<double> RES2N_conv(l_conv);

  std::vector<double> CI95N( l_conv);
  std::vector<double> CI952N(l_conv);


  std::vector<double> logabserr(l_conv);

  //std::vector<double> abserr(l_conv);

  std::vector<double> _1overN_conv(l_conv);
  std::vector<double> log1overN_conv(l_conv);
  for(int l=0; l<l_conv; l++) 
  {
    _1overN_conv[l] = 1.0 / N_conv[l];
    log1overN_conv[l] = log(1.0/N_conv[l]);
  } */



  //double BC = 0.4184019533; //age=40, T=30, alpha_penalty=  0.005


  // RR SEEMS DOES NOT WORK IN THIS FRAMEWORK! EVEN IN PE and i do not understand why
  //Convergence test PE
  /* std::cout << "*** Convergence Test PE ***\n" << std::endl;
  for(int l=0; l<l_conv; l++)
  { 
    double pri, var_pri, del, var_del;
    std::cout << "\n\n\nFor n = " << N_conv[l] <<  std::endl;
    PE_min_max_GM_MMM(CRRA_delta, X0, r, mu, vols, cov_mat, 
              alpha_penalty, Y0, beta, kappa, sigma,
              start_age_conv, phi_GM, m_GM, b_GM,  
              min_gua_conv, max_gua_conv,
              T_conv, N_conv[l], M_conv,
              pri, var_pri, del, var_del);

    RESN_conv[l] = del;
    CI95N[l] = 1.96*sqrt(var_del/M_conv);

    
    PE_min_max_GM_MMM(CRRA_delta, X0, r, mu, vols, cov_mat, 
              alpha_penalty, Y0, beta, kappa, sigma,
              start_age_conv, phi_GM, m_GM, b_GM,  
              min_gua_conv, max_gua_conv,
              T_conv, N2_conv[l], M_conv,
              pri, var_pri, del, var_del);

    RES2N_conv[l] = del;       
    CI952N[l] = 1.96*sqrt(var_del/M_conv);
    double temp = var_del;


    logabserr[l]  = log(abs(RES2N_conv[l]- RESN_conv[l]));


    std::cout << std::setprecision(10);
    std::cout << "RES2N    = " << RES2N_conv[l] <<  std::endl;
    std::cout << "CI952N   = " << CI952N[l] <<  std::endl;
    //std::cout << "VAR_correction 2N = " << temp  <<  std::endl;

  }

  std::pair<double,double> regr_line_conv = lreg(log1overN_conv, logabserr);
  std::cout << "\nRegressed slope     = " << regr_line_conv.first << std::endl; */


  //Convergence test TI
  /* std::vector<double> RESN_delta( l_conv);
  std::vector<double> RES2N_delta(l_conv);
  std::vector<double> CI95N_delta( l_conv);
  std::vector<double> CI952N_delta(l_conv);
  std::vector<double> logabserr_delta(l_conv);

  std::cout << "*** Convergence Test TI ***\n" << std::endl;
  for(int l=0; l<l_conv; l++)
  { 
    double pri, var_pri, del, var_del;
    std::cout << "\n\n\nFor n = " << N_conv[l] <<  std::endl;
    TI_min_max_GM_MMM(CRRA_delta, X0, r, mu, vols, cov_mat, 
              alpha_penalty, Y0, beta, kappa, sigma,
              start_age_conv, phi_GM, m_GM, b_GM,  
              X0, X0, min_rate, max_rate,
              T_conv, N_conv[l], M_conv,
              pri, var_pri, del, var_del);

    RESN_delta[l] = del;
    CI95N_delta[l] = 1.96*sqrt(var_del/M_conv);
    RESN_conv[l] = pri;
    CI95N[l] = 1.96*sqrt(var_pri/M_conv);

    
    TI_min_max_GM_MMM(CRRA_delta, X0, r, mu, vols, cov_mat, 
              alpha_penalty, Y0, beta, kappa, sigma,
              start_age_conv, phi_GM, m_GM, b_GM,  
              X0, X0, min_rate, max_rate,
              T_conv, N2_conv[l], M_conv,
              pri, var_pri, del, var_del);

    RES2N_delta[l] = del;       
    CI952N_delta[l] = 1.96*sqrt(var_del/M_conv);
    RES2N_conv[l] = pri;       
    CI952N[l] = 1.96*sqrt(var_pri/M_conv);

    logabserr[l]       = log(abs(RES2N_conv[l]  - RESN_conv[l]));
    logabserr_delta[l] = log(abs(RES2N_delta[l] - RESN_delta[l]));

    std::cout << std::setprecision(10);
    std::cout << "RES2N        = " << RES2N_conv[l] <<  std::endl;
    std::cout << "CI952N       = " << CI952N[l] <<  std::endl;
    std::cout << "RES2N_delta  = " << RES2N_delta[l] <<  std::endl;
    std::cout << "CI952N_delta = " << CI952N_delta[l] <<  std::endl;
  } 

	std::pair<double,double> regr_line_conv = lreg(log1overN_conv, logabserr);
  std::cout << "\nRegressed slope price = " << regr_line_conv.first << std::endl;

  std::pair<double,double> regr_line_delta = lreg(log1overN_conv, logabserr_delta);
  std::cout << "\nRegressed slope delta = " << regr_line_delta.first << std::endl; */

 


  //////////////////////////////////////////////
  ///////// VARIANCE REDUCTION TEST ///////////
  ////////////////////////////////////////////

  // ** VR TEST PARAMETERS **
  /* const int l_var = 4;
  const int T_VEC[l_var] = {5, 10, 20, 30};
  const int  n_py = 5;
  const int N_var[l_var] = {T_VEC[0]*n_py, T_VEC[1]*n_py, T_VEC[2]*n_py, T_VEC[3]*n_py};

  int M_var = 1000000;
  double min_gua_var[l_var];// = exp(r*T_VEC)*X0;
  double max_gua_var[l_var];// = exp(10*r*T_VEC)*X0;

  for(int l=0; l<l_var; l++)
  { 
    min_gua_var[l] = X0*exp(   r*T_VEC[l]);
    max_gua_var[l] = X0*exp(10*r*T_VEC[l]);
  }

  double CRRA_delta_VR    = 1.0;
  double alpha_penalty_VR = 0.0025;


  //double CRRA_delta_VR    = 0.5;    //parameters that gives a huge variance reduction
  //double alpha_penalty_VR = 0.0005; //parameters that gives a huge variance reduction
  //double alpha_penalty_VR = 0.0002; //even bigger variance reduction

  std::vector<double> RES(    l_var);
  std::vector<double> VAR(    l_var);
  std::vector<double> CI95(   l_var);
  std::vector<double> RES_VR( l_var);
  std::vector<double> VAR_VR( l_var);
  std::vector<double> CI95_VR(l_var);

  std::vector<double> _1overN(l_var);
  for(int l=0; l<l_var; l++) _1overN[l] = 1.0 / N_var[l]; */

  
  
  // PE VR TEST
  /* 
  std::cout << "*** Variance Reduction Test" << std::endl;
  std::cout << "*** PE Test\n" << std::endl;
  for(int l=0; l<l_var; l++)
  { 
    std::cout << "\n\n\nFor n = " << N_var[l] <<  std::endl;
    double price, var_price, delta, var_delta;
    PE_NOVR_min_max_GM_MMM(CRRA_delta_VR, X0, r, mu, vols, cov_mat, 
              alpha_penalty_VR, Y0, beta, kappa, sigma,
              start_age, phi_GM, m_GM, b_GM,  
              min_gua_var[l], max_gua_var[l], T_VEC[l], N_var[l], M_var,
              price, var_price, delta, var_delta);
    
    RES[l] = price;
    VAR[l] = var_price;
    CI95[l] = 1.96*sqrt(var_price/M_var);

    PE_min_max_GM_MMM(CRRA_delta_VR, X0, r, mu, vols, cov_mat, 
              alpha_penalty_VR, Y0, beta, kappa, sigma,
              start_age, phi_GM, m_GM, b_GM,  
              min_gua_var[l], max_gua_var[l], T_VEC[l], N_var[l], M_var,
              price, var_price, delta, var_delta);
    RES_VR[l] = price;
    VAR_VR[l] = var_price;
    CI95_VR[l] = 1.96*sqrt(var_price/M_var);
    std::cout << std::setprecision(10);
    std::cout << "RES     = " << RES[l] <<  std::endl;
    std::cout << "CI95    = " << CI95[l] <<  std::endl;
    std::cout << "RES_VR  = " << RES_VR[l] <<  std::endl;
    std::cout << "CI95_VR = " << CI95_VR[l] <<  std::endl;
    std::cout << "VAR    = " << std::scientific << VAR[l] <<  std::endl;
    std::cout << "VAR_VR = " << std::scientific << VAR_VR[l] <<  std::endl;
    std::cout << "Reduction of variance = " << std::fixed << (VAR[l]-VAR_VR[l])/VAR[l]*100 << "%" <<  std::endl;
  }  */
 

  //TI VR TEST
  /* std::cout << "*** TI Test\n" << std::endl;
  for(int l=0; l<l_var; l++)
  { 
    std::cout << "\n\n\nFor n = " << N_var[l] <<  std::endl;
    std::cout << "T = " << T_VEC[l] <<  std::endl;
    double price, var_price, delta, var_delta;
    TI_NOVR_min_max_GM_MMM(CRRA_delta_VR, X0, r, mu, vols, cov_mat, 
									alpha_penalty_VR, Y0, beta, kappa, sigma,
                  start_age, phi_GM, m_GM, b_GM,  
                  X0, X0, min_rate, max_rate,
                  T_VEC[l], N_var[l], M_var,
                  price, var_price, delta, var_delta);
    
    RES[l] = price;
    VAR[l] = var_price;
    CI95[l] = 1.96*sqrt(var_price/M_var);

    TI_min_max_GM_MMM(CRRA_delta_VR, X0, r, mu, vols, cov_mat, 
									alpha_penalty_VR, Y0, beta, kappa, sigma,
                  start_age, phi_GM, m_GM, b_GM,
                  X0, X0, min_rate, max_rate,
                  T_VEC[l], N_var[l], M_var,
                  price, var_price, delta, var_delta);
    RES_VR[l] = price;
    VAR_VR[l] = var_price;
    CI95_VR[l] = 1.96*sqrt(var_price/M_var);
    std::cout << std::setprecision(10);
    std::cout << "RES     = " << RES[l] <<  std::endl;
    std::cout << "CI95    = " << CI95[l] <<  std::endl;
    std::cout << "RES_VR  = " << RES_VR[l] <<  std::endl;
    std::cout << "CI95_VR = " << CI95_VR[l] <<  std::endl;
    std::cout << "VAR    = " << std::scientific << VAR[l] <<  std::endl;
    std::cout << "VAR_VR = " << std::scientific << VAR_VR[l] <<  std::endl;
    std::cout << "Reduction of variance = " << std::fixed << (VAR[l]-VAR_VR[l])/VAR[l]*100 << "%" <<  std::endl;
  }  */

  //EI VR TEST
  /* std::cout << "*** EI Test\n" << std::endl;
  for(int l=0; l<l_var; l++)
  { 
    std::cout << "\n\n\nFor n = " << N_var[l] <<  std::endl;
    std::cout << "T = " << T_VEC[l] <<  std::endl;
    double price, var_price, delta, var_delta;
    EI_NOVR_min_max_GM_MMM(CRRA_delta_VR, X0, r, mu, vols, cov_mat, 
									alpha_penalty_VR, Y0, beta, kappa, sigma,
                  start_age, phi_GM, m_GM, b_GM,  
                  X0, X0, min_rate, max_rate, perc_prem_death,
                  T_VEC[l], N_var[l], M_var,
                  price, var_price, delta, var_delta);
    
    RES[l] = price;
    VAR[l] = var_price;
    CI95[l] = 1.96*sqrt(var_price/M_var);

    EI_min_max_GM_MMM(CRRA_delta_VR, X0, r, mu, vols, cov_mat, 
									alpha_penalty_VR, Y0, beta, kappa, sigma,
                  start_age, phi_GM, m_GM, b_GM,  
                  X0, X0, min_rate, max_rate, perc_prem_death,
                  T_VEC[l], N_var[l], M_var,
                  price, var_price, delta, var_delta);
    RES_VR[l] = price;
    VAR_VR[l] = var_price;
    CI95_VR[l] = 1.96*sqrt(var_price/M_var);
    std::cout << std::setprecision(10);
    std::cout << "RES     = " << RES[l] <<  std::endl;
    std::cout << "CI95    = " << CI95[l] <<  std::endl;
    std::cout << "RES_VR  = " << RES_VR[l] <<  std::endl;
    std::cout << "CI95_VR = " << CI95_VR[l] <<  std::endl;
    std::cout << "VAR    = " << std::scientific << VAR[l] <<  std::endl;
    std::cout << "VAR_VR = " << std::scientific << VAR_VR[l] <<  std::endl;
    std::cout << "Reduction of variance = " << std::fixed << (VAR[l]-VAR_VR[l])/VAR[l]*100 << "%" <<  std::endl;
  }  */

  
  /////////////////////////////////
  ///// Hedging Parameters ///////
  ///////////////////////////////

  int N1      = 5;
  int N2      = 52;
  int N3      = 1040; //=52*20

  int N_delta_lvl = 8;

  int M       = 1000; //10000

  int M_delta = 200;  //400


  ///////////////////////////////////////////////////////
  ///////// 1 Classes VS N Classes test PETIEI /////////
  /////////////////////////////////////////////////////
  
  /* double mean_age = start_age;

  //double mean_age = 75;

  
  std::vector<double> AGES_PE_multi_class  = {55.5, 56.5, 57.5, 58.5, 59.5, 60.5, 61.5, 62.5, 63.5, 64.5};
  std::vector<double> AGES_PE_mean         = {mean_age};
  std::vector<double> AGES_PE_0;
  int n_PE_multi_class = 100;
  int n_PE_mean        = AGES_PE_multi_class.size()*n_PE_multi_class;
  int n_PE_0 = 0;
  double nom_PE  = 1.0;


  std::vector<double> AGES_TI_multi_class = {55.5, 56.5, 57.5, 58.5, 59.5, 60.5, 61.5, 62.5, 63.5, 64.5};
  std::vector<double> AGES_TI_mean        = {mean_age};
  std::vector<double> AGES_TI_0;
  int n_TI_multi_class = 100;
  int n_TI_mean        = AGES_TI_multi_class.size()*n_TI_multi_class;
  int n_TI_0 = 0;
  double nom_TI = 1.0;


  std::vector<double> AGES_EI_multi_class = {55.5, 56.5, 57.5, 58.5, 59.5, 60.5, 61.5, 62.5, 63.5, 64.5};
  std::vector<double> AGES_EI_mean        = {mean_age};
  std::vector<double> AGES_EI_0;
  int n_EI_multi_class = 100;
  int n_EI_mean        = AGES_EI_multi_class.size()*n_EI_multi_class;
  int n_EI_0 = 0;
  double nom_EI  = 1.0;
  
  PETIEI_polyciholders PH_DATA_1( AGES_PE_0, AGES_TI_mean, AGES_EI_0, 
                                      nom_PE, nom_TI,  nom_EI,
                                      n_PE_0,    1,    n_EI_0);
  PH_DATA_1.print_summary();


  PETIEI_polyciholders PH_DATA_mean( AGES_PE_0, AGES_TI_mean, AGES_EI_0, 
                                      nom_PE,       nom_TI,  nom_EI,
                                      n_PE_0,    n_TI_mean,    n_EI_0);
  PH_DATA_mean.print_summary();

  PETIEI_polyciholders PH_DATA_mc( AGES_PE_0, AGES_TI_multi_class, AGES_EI_0, 
                                      nom_PE,  nom_TI, nom_EI,
                                      n_PE_0,    n_TI_multi_class,    n_EI_0);
  PH_DATA_mc.print_summary();
  

  start = std::chrono::high_resolution_clock::now();
  Scenarios XY_scenarios =
                fund_ci_scenario_generator(CRRA_delta, r, mu, vols, cov_mat, 
									                    alpha_penalty, Y0, beta, kappa, sigma,
                                                              T, N3, M);
  finish = std::chrono::high_resolution_clock::now();
	elapsed = finish - start;
  std::cout << "\n" << M << " scenarios " << "with N = " << N3;
  std::cout << " and dimY = " << Y0.size() << " generated in : " << elapsed.count() << std::endl;
	std::cout << "execution time  = " << elapsed.count() << std::endl;

  //XY_scenarios.print();


  std::cout << "\nRebalancing " << N3 <<  " times:" << std::endl;
  std::cout << "\n1 ph class results\n";
	start = std::chrono::high_resolution_clock::now();
  Output_hedging output_1 =
    PETIEI_hedge_cost_paral_mc_MLMC_scenarios(CRRA_delta, r, mu, vols, cov_mat, 
									alpha_penalty, beta, kappa, sigma,
                  PH_DATA_1, phi_GM, m_GM, b_GM, 
                  min_rate, max_rate, perc_prem_death, 
                  T, N_delta_lvl, M_delta, XY_scenarios);
  finish = std::chrono::high_resolution_clock::now();
	elapsed = finish - start;
	std::cout << "execution time  = " << elapsed.count() << std::endl;
  writeToCSV(output_1.no_hedge, output_1.static_hedge, output_1.hedging_cost, "hist_1.csv");

  std::cout << "\nmean_class results\n";
	start = std::chrono::high_resolution_clock::now();
  Output_hedging output_mean =
    PETIEI_hedge_cost_paral_mc_MLMC_scenarios(CRRA_delta, r, mu, vols, cov_mat, 
									alpha_penalty, beta, kappa, sigma,
                  PH_DATA_mean, phi_GM, m_GM, b_GM, 
                  min_rate, max_rate, perc_prem_death, 
                  T, N_delta_lvl, M_delta, XY_scenarios);
  finish = std::chrono::high_resolution_clock::now();
	elapsed = finish - start;
	std::cout << "execution time  = " << elapsed.count() << std::endl;
  writeToCSV(output_mean.no_hedge, output_mean.static_hedge, output_mean.hedging_cost, "hist_mean.csv");
  
  std::cout << "\nmulti_class results\n";
	start = std::chrono::high_resolution_clock::now();
  Output_hedging output_mc =
    PETIEI_hedge_cost_paral_mc_MLMC_scenarios(CRRA_delta, r, mu, vols, cov_mat, 
									alpha_penalty, beta, kappa, sigma,
                  PH_DATA_mc, phi_GM, m_GM, b_GM, 
                  min_rate, max_rate, perc_prem_death, 
                  T, N_delta_lvl, M_delta, XY_scenarios);
  finish = std::chrono::high_resolution_clock::now();
	elapsed = finish - start;
	std::cout << "execution time  = " << elapsed.count() << std::endl;
  writeToCSV(output_mc.no_hedge, output_mc.static_hedge, output_mc.hedging_cost, "hist_mc.csv");
 */

  ////////////////////////////////////////////////////////////////
  //// CIR general-model scenario test without MLMC            ////
  //// Same policyholder data and outer scenarios as above.     ////
  ////////////////////////////////////////////////////////////////


  double mean_age = start_age;

  //double mean_age = 75;

  
  std::vector<double> AGES_PE_multi_class  = {55.5, 56.5, 57.5, 58.5, 59.5, 60.5, 61.5, 62.5, 63.5, 64.5};
  std::vector<double> AGES_PE_mean         = {mean_age};
  std::vector<double> AGES_PE_0;
  int n_PE_multi_class = 100;
  int n_PE_mean        = AGES_PE_multi_class.size()*n_PE_multi_class;
  int n_PE_0 = 0;
  double nom_PE  = 1.0;


  std::vector<double> AGES_TI_multi_class = {55.5, 56.5, 57.5, 58.5, 59.5, 60.5, 61.5, 62.5, 63.5, 64.5};
  std::vector<double> AGES_TI_mean        = {mean_age};
  std::vector<double> AGES_TI_0;
  int n_TI_multi_class = 100;
  int n_TI_mean        = AGES_TI_multi_class.size()*n_TI_multi_class;
  int n_TI_0 = 0;
  double nom_TI = 1.0;


  std::vector<double> AGES_EI_multi_class = {55.5, 56.5, 57.5, 58.5, 59.5, 60.5, 61.5, 62.5, 63.5, 64.5};
  std::vector<double> AGES_EI_mean        = {mean_age};
  std::vector<double> AGES_EI_0;
  int n_EI_multi_class = 100;
  int n_EI_mean        = AGES_EI_multi_class.size()*n_EI_multi_class;
  int n_EI_0 = 0;
  double nom_EI  = 1.0;
  
  PETIEI_polyciholders PH_DATA_1( AGES_PE_0, AGES_TI_mean, AGES_EI_0, 
                                      nom_PE, nom_TI,  nom_EI,
                                      n_PE_0,    1,    n_EI_0);
  PH_DATA_1.print_summary();


  PETIEI_polyciholders PH_DATA_mean( AGES_PE_0, AGES_TI_mean, AGES_EI_0, 
                                      nom_PE,       nom_TI,  nom_EI,
                                      n_PE_0,    n_TI_mean,    n_EI_0);
  PH_DATA_mean.print_summary();

  PETIEI_polyciholders PH_DATA_mc( AGES_PE_0, AGES_TI_multi_class, AGES_EI_0, 
                                      nom_PE,  nom_TI, nom_EI,
                                      n_PE_0,    n_TI_multi_class,    n_EI_0);
  PH_DATA_mc.print_summary();

  // The ordinary-Monte-Carlo hedger expects a number of time intervals,
  // whereas N_delta_lvl above is an MLMC level count. Use N2 (52 intervals)
  // for the new routine while keeping the same M, N3 and M_delta values.
  const int N_delta_general = N2;

  start = std::chrono::high_resolution_clock::now();
  Scenarios XY_scenarios_general_CIR =
                fund_ci_scenario_generator_new<ZigguratNormalSampler>(CRRA_delta, r, mu, vols, cov_mat,
                                               alpha_penalty, Y0, beta, kappa, sigma,
                                               T, N3, M, prm_NV_CIR, NV_CIR);
  finish = std::chrono::high_resolution_clock::now();
  elapsed = finish - start;
  std::cout << "\nCIR general-model Ziggurat scenarios " << M
            << " with N = " << N3
            << " and dimY = " << Y0.size()
            << " generated in: " << elapsed.count() << std::endl;

  std::cout << "\nGeneral-model Ziggurat ordinary-MC hedging, 1 ph class\n";
  start = std::chrono::high_resolution_clock::now();
  Output_hedging output_general_1 =
    PETIEI_hedge_cost_paral_mc_scenarios_general_model<CIRModel4, ZigguratNormalSampler>(
                  CRRA_delta, r, mu, vols, cov_mat,
                  alpha_penalty, beta, kappa, sigma,
                  PH_DATA_1, phi_GM, m_GM, b_GM,
                  min_rate, max_rate, perc_prem_death,
                  T, N_delta_general, M_delta,
                  XY_scenarios_general_CIR);
  finish = std::chrono::high_resolution_clock::now();
  elapsed = finish - start;
  std::cout << "execution time = " << elapsed.count() << std::endl;
  writeToCSV(output_general_1.no_hedge,
             output_general_1.static_hedge,
             output_general_1.hedging_cost,
             "hist_general_CIR_1.csv");

  std::cout << "\nGeneral-model Ziggurat ordinary-MC hedging, mean class\n";
  start = std::chrono::high_resolution_clock::now();
  Output_hedging output_general_mean =
    PETIEI_hedge_cost_paral_mc_scenarios_general_model<CIRModel4, ZigguratNormalSampler>(
                  CRRA_delta, r, mu, vols, cov_mat,
                  alpha_penalty, beta, kappa, sigma,
                  PH_DATA_mean, phi_GM, m_GM, b_GM,
                  min_rate, max_rate, perc_prem_death,
                  T, N_delta_general, M_delta,
                  XY_scenarios_general_CIR);
  finish = std::chrono::high_resolution_clock::now();
  elapsed = finish - start;
  std::cout << "execution time = " << elapsed.count() << std::endl;
  writeToCSV(output_general_mean.no_hedge,
             output_general_mean.static_hedge,
             output_general_mean.hedging_cost,
             "hist_general_CIR_mean.csv");

  std::cout << "\nGeneral-model Ziggurat ordinary-MC hedging, multi class\n";
  start = std::chrono::high_resolution_clock::now();
  Output_hedging output_general_mc =
    PETIEI_hedge_cost_paral_mc_scenarios_general_model<CIRModel4, ZigguratNormalSampler>(
                  CRRA_delta, r, mu, vols, cov_mat,
                  alpha_penalty, beta, kappa, sigma,
                  PH_DATA_mc, phi_GM, m_GM, b_GM,
                  min_rate, max_rate, perc_prem_death,
                  T, N_delta_general, M_delta,
                  XY_scenarios_general_CIR);
  finish = std::chrono::high_resolution_clock::now();
  elapsed = finish - start;
  std::cout << "execution time = " << elapsed.count() << std::endl;
  writeToCSV(output_general_mc.no_hedge,
             output_general_mc.static_hedge,
             output_general_mc.hedging_cost,
             "hist_general_CIR_mc.csv");





  //////////////////////////////////////////////////////
  /////// DEBUG OF PETIEIMLMC PRICING ROUTINE /////////
  ////////////////////////////////////////////////////

  /* const int N_lvl_PETIEI = 8;

  const int M_debug = 200;
  double    T_debug = 20;

  double min_gua_t_debug = X0*exp(min_rate*T_debug);
  double max_gua_t_debug = X0*exp(max_rate*T_debug);

  std::vector<std::vector<double>> prices_petiei(    3, std::vector<double>(N_lvl_PETIEI, 0.0));
  std::vector<std::vector<double>> deltas_petiei(    3, std::vector<double>(N_lvl_PETIEI, 0.0));
  std::vector<std::vector<double>> prices_var_petiei(3, std::vector<double>(N_lvl_PETIEI, 0.0));
  std::vector<std::vector<double>> deltas_var_petiei(3, std::vector<double>(N_lvl_PETIEI, 0.0));

  double  pri_PE, var_pri_PE, del_PE, var_del_PE,
          pri_TI, var_pri_TI, del_TI, var_del_TI,
          pri_EI, var_pri_EI, del_EI, var_del_EI;

  double  pri_PE_MLMC, CI_pri_PE_MLMC, del_PE_MLMC, CI_del_PE_MLMC,
          pri_TI_MLMC, CI_pri_TI_MLMC, del_TI_MLMC, CI_del_TI_MLMC,
          pri_EI_MLMC, CI_pri_EI_MLMC, del_EI_MLMC, CI_del_EI_MLMC;

  start = std::chrono::high_resolution_clock::now();
  PETIEI_min_max_GM_MMM(CRRA_delta, X0, r, mu, vols, cov_mat, 
            alpha_penalty, Y0, beta, kappa, sigma,
            start_age, phi_GM, m_GM, b_GM,  
            X0, X0, min_rate, max_rate, perc_prem_death,
            T_debug, 128, M_debug,
            pri_PE, var_pri_PE, del_PE, var_del_PE,
            pri_TI, var_pri_TI, del_TI, var_del_TI,
            pri_EI, var_pri_EI, del_EI, var_del_EI);
  finish = std::chrono::high_resolution_clock::now();
  elapsed = finish - start;
  std::cout << "execution time      = " << elapsed.count() << std::endl; 

  start = std::chrono::high_resolution_clock::now();
  PETIEI_min_max_GM_MMM_MLMC(CRRA_delta, X0, r, mu, vols, cov_mat, 
            alpha_penalty, Y0, beta, kappa, sigma,
            start_age, phi_GM, m_GM, b_GM,  
            X0, X0, min_rate, max_rate, perc_prem_death,
            T_debug, N_lvl_PETIEI, M_debug,
            pri_PE_MLMC, CI_pri_PE_MLMC, del_PE_MLMC, CI_del_PE_MLMC,
            pri_TI_MLMC, CI_pri_TI_MLMC, del_TI_MLMC, CI_del_TI_MLMC,
            pri_EI_MLMC, CI_pri_EI_MLMC, del_EI_MLMC, CI_del_EI_MLMC);
  finish = std::chrono::high_resolution_clock::now();
  elapsed = finish - start;
  std::cout << "execution time MLMC = " << elapsed.count() << std::endl; 
  
  std::vector<double> result(prices_petiei[0].size());
  std::cout << "\nprice_multi_PE      = " << pri_PE  <<  std::endl;
  std::cout << "CI95_multi_PE   = " << 1.96*sqrt(var_pri_PE/M_debug) <<  std::endl;
  std::cout << "price_multi_PE_MLMC = "<< pri_PE_MLMC  <<  std::endl;
  std::cout << "CI95_multi_PE_MLMC  = " << CI_pri_PE_MLMC <<  std::endl;


  std::cout << "\nprice_multi_TI      = " << pri_TI  <<  std::endl;
  std::cout << "CI95_multi_TI   = " << 1.96*sqrt(var_pri_TI/M_debug) <<  std::endl;
  std::cout << "price_multi_TI_MLMC = "<< pri_TI_MLMC  <<  std::endl;
  std::cout << "CI95_multi_TI_MLMC  = " << CI_pri_TI_MLMC <<  std::endl;


  std::cout << "\nprice_multi_EI      = " << pri_EI  <<  std::endl;
  std::cout << "CI95_multi_EI   = " << 1.96*sqrt(var_pri_EI/M_debug) <<  std::endl;
  std::cout << "price_multi_EI_MLMC = "<< pri_EI_MLMC  <<  std::endl;
  std::cout << "CI95_multi_EI_MLMC  = " << CI_pri_EI_MLMC <<  std::endl;


  
  std::cout << "\ndelta_multi_PE      = " << del_PE  <<  std::endl;
  std::cout << "CI95_multi_PE   = " << 1.96*sqrt(var_del_PE/M_debug) <<  std::endl;
  std::cout << "delta_multi_PE_MLMC = "<< del_PE_MLMC  <<  std::endl;
  std::cout << "CI95_multi_PE_MLMC  = " << CI_del_PE_MLMC <<  std::endl;


  std::cout << "\ndelta_multi_TI      = " << del_TI  <<  std::endl;
  std::cout << "CI95_multi_TI   = " << 1.96*sqrt(var_del_TI/M_debug) <<  std::endl;
  std::cout << "delta_multi_TI_MLMC = "<< del_TI_MLMC  <<  std::endl;
  std::cout << "CI95_multi_TI_MLMC  = " << CI_del_TI_MLMC <<  std::endl;


  std::cout << "\ndelta_multi_EI      = " << del_EI  <<  std::endl;
  std::cout << "CI95_multi_EI   = " << 1.96*sqrt(var_del_EI/M_debug) <<  std::endl;
  std::cout << "delta_multi_EI_MLMC = "<< del_EI_MLMC  <<  std::endl;
  std::cout << "CI95_multi_EI_MLMC  = " << CI_del_EI_MLMC <<  std::endl; */


  // PLOTS
  using namespace matplot;

  //////////////////////////////////////////////////
  /////// PLOT of DEBUG OF PETIEI PRICING /////////
  ////////////////////////////////////////////////

  /* figure();
  errorbar(_1overN_debug, RES_PE,       CI95_PE);
  hold(on);
  errorbar(_1overN_debug, RES_multi_PE, CI95_multi_PE);


  title("PETIEI DEBUG: PE");
  xlabel("N");
  ylabel("Values");

  figure();
  errorbar(_1overN_debug, RES_TI,       CI95_TI);
  hold(on);
  errorbar(_1overN_debug, RES_multi_TI, CI95_multi_TI);


  title("PETIEI DEBUG: TI");
  xlabel("N");
  ylabel("Values");

  figure();
  errorbar(_1overN_debug, RES_EI,       CI95_EI);
  hold(on);
  errorbar(_1overN_debug, RES_multi_EI, CI95_multi_EI);


  title("PETIEI DEBUG: EI");
  xlabel("N");
  ylabel("Values"); */

  


  //////////////////////////////////////
  ////// Weights evolutions plots /////
  ////////////////////////////////////

  /* double T_weight_ev = 20;
  int N_disc = 1000;
  std::vector<double> Time = matplot::linspace(0, T_weight_ev, N_disc+1);

  std::pair<std::vector<std::vector<double> >,std::vector<std::vector<double> > > 
                  WEIGHTS_CI = weights_evolution(CRRA_delta, r, mu, vols, cov_mat, 
                        alpha_penalty, Y0, beta, kappa, sigma, T_weight_ev, N_disc);


  figure();
  plot(Time, WEIGHTS_CI.second[0]);
  hold(on);
  plot(Time, WEIGHTS_CI.second[1]);
  plot(Time, WEIGHTS_CI.second[2]);
  plot(Time, WEIGHTS_CI.second[3]);
  title("Carbon Intensity evolution");
  xlabel("Time");
  ylabel("Values");

  figure();
  plot(Time, WEIGHTS_CI.first[0]);
  hold(on);
  plot(Time, WEIGHTS_CI.first[1]);
  plot(Time, WEIGHTS_CI.first[2]);
  plot(Time, WEIGHTS_CI.first[3]);
  plot(Time, WEIGHTS_CI.first[0]+WEIGHTS_CI.first[1]+WEIGHTS_CI.first[2]+WEIGHTS_CI.first[3]);
  title("Weights evolution");
  xlabel("Time");
  ylabel("Values"); 
  
  // write a scenario of weights and CIs over time
  writeToCSV_CIs_Weights(WEIGHTS_CI.first, WEIGHTS_CI.second, "CIW.csv");*/


  //////////////////////////////////////
  ////// Convergence test   plots /////
  ////////////////////////////////////
  /* figure();
  errorbar(_1overN_conv, RESN_conv,  CI95N);


  title("Convergence TEST");
  xlabel("1/n");
  ylabel("Values");

  figure(); 
  plot(log1overN_conv, logabserr);
  title("Convergence TEST slope");
  xlabel("log(1/n)");
  ylabel("logabserr"); */



  //////////////////////////////////////
  ////// COST HISTOGRAMS PETIEI ///////
  ////////////////////////////////////

  /* auto rule = histogram::binning_algorithm::automatic;

  figure(); 
  hist(output_1.hedging_cost, rule, histogram::normalization::probability);
  hold(on);
  hist(output_mean.hedging_cost, rule, histogram::normalization::probability);
  hist(output_mc.hedging_cost, rule, histogram::normalization::probability);
  title("1 Vs multi and different number of classes");
  xlabel("COST");
  ylabel("Proba");

  legend({"Continous hedging 1phclass","Continous hedging mean","Continous hedging mc"});

  figure(); 
  hist(output_mean.hedging_cost, rule, histogram::normalization::probability);
  hold(on);
  hist(output_mc.hedging_cost, rule, histogram::normalization::probability);
  title("Different number of classes");
  xlabel("COST");
  ylabel("Proba");

  legend({"Continous hedging mean","Continous hedging mc"}); */

  

  ///////////////////////////////
  ////// COST HISTOGRAMS ///////
  /////////////////////////////

  /* auto rule = histogram::binning_algorithm::automatic;
  figure(); 
  hist(output3.trading_gains, rule, histogram::normalization::probability);
  xlabel("Trading Gains");
  ylabel("Proba");

  figure(); 
  hist(output3.hedging_cost, rule, histogram::normalization::probability);
  xlabel("Hedge Cost");
  ylabel("Proba");


  figure(); 
  hist(output3.X, rule, histogram::normalization::probability);
  xlabel("Fund Value");
  ylabel("Proba");

  figure(); 
  hist(output3.no_hedge, rule, histogram::normalization::probability);
  hold(on);
  hist(output3.static_hedge, rule, histogram::normalization::probability);
  hist(output3.hedging_cost, rule, histogram::normalization::probability);
  title("Different Hedging");
  xlabel("COST");
  ylabel("Proba");

  legend({"No hedging","Static hedging","Continous hedging"});


  writeToCSV(output3.no_hedge, output3.static_hedge, output3.hedging_cost, "hist.csv");  */

  

  show();

  

	return 1;
}
#endif // HEDGE_COST_NO_MAIN

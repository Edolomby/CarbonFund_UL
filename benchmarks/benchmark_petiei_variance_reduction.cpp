#include <carbonfund/pricing/petiei_monte_carlo.hpp>
#include <carbonfund/random/simulation_rng.hpp>
#include <carbonfund/utils/TablePrinter.hpp>
#include <boost/math/distributions/students_t.hpp>
#include <algorithm>
#include <array>
#include <charconv>
#include <cmath>
#include <cstdint>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {
namespace cf = carbonfund;
using namespace cf::pricing;
using Pricer = PetieiMonteCarloPricer4<cf::models::CIRNVStepper4>;
constexpr std::array<int,4> maturities{5,10,20,30};
constexpr std::array<const char*,3> contract_names{"PE","TI","EI"};
constexpr std::array<const char*,4> mode_names{
    "standard","standard_antithetic","conditioned","conditioned_antithetic"};
constexpr std::array<PetieiContract,3> contracts{
    PetieiContract::pure_endowment,PetieiContract::term_insurance,PetieiContract::endowment_insurance};
constexpr std::array<std::pair<int,int>,6> contrasts{
    std::pair{0,1},{0,2},{0,3},{1,2},{1,3},{2,3}};
// Two declared fund constructions, four maturities, three contracts, two
// quantities and six contrasts. Bonferroni also covers a single-construction run.
constexpr std::size_t confidence_family=2*4*3*2*6;
constexpr std::array<std::array<double,4>,3> paper_standard{{
    {{2.3325e-2,4.2705e-2,3.7802e-2,4.6313e-3}},
    {{6.4640e-6,4.5424e-5,4.2306e-4,1.1124e-3}},
    {{2.3378e-2,4.2908e-2,3.9107e-2,5.9768e-3}}}};
constexpr std::array<std::array<double,4>,3> paper_conditioned{{
    {{7.3157e-6,2.0879e-5,2.1528e-5,2.2617e-6}},
    {{2.5549e-8,5.7513e-7,1.3889e-5,5.7844e-5}},
    {{8.1701e-6,2.8083e-5,6.8198e-5,8.1028e-5}}}};
volatile double checksum=0;
struct Options {
    std::size_t paths=1000000,repetitions=3,batches=100,batch_paths=10000;
    std::uint64_t seed=20261015;
    PetieiFundSampling fund=PetieiFundSampling::path;
    std::string prefix="petiei_variance_reduction";
};
std::uint64_t mix(std::uint64_t x)
{
    x+=UINT64_C(0x9e3779b97f4a7c15);
    x=(x^(x>>30))*UINT64_C(0xbf58476d1ce4e5b9);
    x=(x^(x>>27))*UINT64_C(0x94d049bb133111eb);
    return x^(x>>31);
}
std::uint64_t integer(std::string_view s)
{
    std::uint64_t value=0;
    const auto r=std::from_chars(s.data(),s.data()+s.size(),value);
    if(r.ec!=std::errc{} || r.ptr!=s.data()+s.size()) throw std::invalid_argument("Invalid integer argument.");
    return value;
}
std::size_t even_paths(std::size_t n)
{
    if(n<4 || n==std::numeric_limits<std::size_t>::max())
        throw std::invalid_argument("Trajectory requests must be at least four and round without overflow.");
    return n+n%2;
}
const char* fund_name(PetieiFundSampling f)
{
    return f==PetieiFundSampling::path?"path":"independent_marginals";
}
Options options(int argc,char** argv)
{
    Options o;
    for(int i=1;i<argc;i+=2) {
        if(i+1==argc) throw std::invalid_argument("Each option requires a value; use --help for usage.");
        const std::string name=argv[i],value=argv[i+1];
        if(name=="--paths") o.paths=integer(value);
        else if(name=="--repetitions") o.repetitions=integer(value);
        else if(name=="--batches") o.batches=integer(value);
        else if(name=="--batch-paths") o.batch_paths=integer(value);
        else if(name=="--seed") o.seed=integer(value);
        else if(name=="--output-prefix") {
            if(value.empty() || value.find_first_of("\"\\\n\r")!=std::string::npos)
                throw std::invalid_argument("Output prefix must be a nonempty simple path.");
            o.prefix=value;
        }
        else if(name=="--fund-sampling") {
            if(value=="path") o.fund=PetieiFundSampling::path;
            else if(value=="marginals") o.fund=PetieiFundSampling::independent_marginals;
            else throw std::invalid_argument("Fund sampling must be path or marginals.");
        }
        else throw std::invalid_argument("Unknown benchmark option: "+name);
    }
    if(o.repetitions==0 || o.batches==1) throw std::invalid_argument("Repetitions must be positive; batches must be zero or at least two.");
    static_cast<void>(even_paths(o.paths));
    if(o.batches) static_cast<void>(even_paths(o.batch_paths));
    return o;
}
struct Inputs {
    PetieiMonteCarloParameters contract;
    cf::models::CIRNVStepper4::Parameters carbon;
    cf::portfolio::CarbonAllocationParameters4 allocation;
    Eigen::Vector4d state{5000,4000,3000,1000};
    cf::contracts::CohortBook book{{{60,1}}};
    cf::mortality::GompertzMakeham mortality{{0.0041959,79.6921211,1.0/11.5818911}};
    explicit Inputs(int t)
    {
        contract.fund_value=1;contract.minimum_guarantee=1;contract.maximum_guarantee=1;
        contract.minimum_guarantee_growth_rate=.05;contract.maximum_guarantee_growth_rate=.5;
        contract.time_to_maturity=t;contract.time_intervals=5*t;contract.endowment_death_benefit_fraction=1;
        carbon.long_run_mean << 2500,2000,1500,500;
        carbon.mean_reversion.setConstant(.05);carbon.volatility.setConstant(3);
        allocation.risk_free_rate=.05;allocation.risk_aversion=1;allocation.carbon_penalty=.0025;
        allocation.expected_returns << .25,.15,.10,.08;
        const Eigen::Vector4d sigma{.30,.25,.20,.16};
        Eigen::Matrix4d rho;
        rho << 1,.44,.39,.32, .44,1,.30,.33, .39,.30,1,.31, .32,.33,.31,1;
        allocation.covariance=sigma.asDiagonal()*rho*sigma.asDiagonal();
    }
    auto prepare() const {
        return Pricer::prepare(contract,state,carbon,allocation,
            {book.cohorts(),book.cohorts(),book.cohorts()},mortality);
    }
};
struct Measurement { PetieiContractEstimate estimate; double cpu; std::size_t paths,observations; };
Measurement run(const Pricer::Prepared& p,std::size_t paths,std::uint64_t seed,
                int mode,int contract,PetieiFundSampling fund)
{
    cf::random::SimulationRng carbon{mix(seed^UINT64_C(0x434152424f4e))};
    cf::random::SimulationRng financial{mix(seed^UINT64_C(0x46494e414e4345))};
    const auto sampling=(mode%2)?PetieiSampling::antithetic:PetieiSampling::independent;
    const auto begin=std::clock();
    const auto r=mode<2?Pricer::evaluate(p,paths,carbon,financial,PetieiEstimator::standard,sampling,contracts[contract],fund)
                       :Pricer::evaluate(p,paths,carbon,sampling,contracts[contract]);
    const auto e=contract==0?r.pure_endowment:contract==1?r.term_insurance:r.endowment_insurance;
    checksum=e.price.mean+e.delta.mean;
    const auto end=std::clock();
    if(begin==static_cast<std::clock_t>(-1)||end==static_cast<std::clock_t>(-1)) throw std::runtime_error("CPU timing unavailable.");
    if(r.simulations!=paths || r.independent_samples!=paths/((mode%2)?2:1))
        throw std::runtime_error("Unexpected benchmark trajectory/observation counts.");
    return {e,static_cast<double>(end-begin)/CLOCKS_PER_SEC,r.simulations,r.independent_samples};
}
double mean(const std::vector<double>& x)
{
    return static_cast<double>(std::accumulate(x.begin(),x.end(),0.0L)/x.size());
}
double variance(const std::vector<double>& x)
{
    const long double m=std::accumulate(x.begin(),x.end(),0.0L)/x.size();
    long double sum=0;for(double v:x) sum+=(v-m)*(v-m);
    return static_cast<double>(sum/(x.size()-1));
}
double rms(const std::vector<double>& x)
{
    long double sum=0;for(double v:x) sum+=static_cast<long double>(v)*v;
    return std::sqrt(static_cast<double>(sum/x.size()));
}
double median(std::vector<double> x)
{
    std::sort(x.begin(),x.end());const auto n=x.size();
    return n%2?x[n/2]:.5*(x[n/2-1]+x[n/2]);
}
void print_summary_table(
    const std::array<Measurement, 4>& results,
    const std::array<std::vector<double>, 4>& timings)
{
    std::vector<std::string> configurations;
    std::vector<std::string> price_reductions, delta_reductions;
    std::vector<double> prices, price_variances, price_errors;
    std::vector<double> deltas, delta_variances, delta_errors, cpu_seconds;

    // Equal trajectory counts: compare variances of the estimated means.
    // This accounts for the smaller number of independent antithetic pairs.
    const auto reduction_percent = [](double se, double reference_se, bool baseline) {
        if (baseline) return std::string{"0%"};
        const double reference_variance = reference_se * reference_se;
        if (reference_variance == 0.0) return std::string{"n/a"};
        const double reduction = 100.0 * (1.0 - se * se / reference_variance);
        return cf::utils::toString(reduction,
            cf::utils::TableConfig{cf::utils::FloatFormat::Fixed, 6}) + "%";
    };

    for (std::size_t mode = 0; mode < results.size(); ++mode) {
        const auto& estimate = results[mode].estimate;
        configurations.emplace_back(mode_names[mode]);
        prices.push_back(estimate.price.mean);
        price_variances.push_back(estimate.price.sample_variance);
        price_reductions.push_back(reduction_percent(
            estimate.price.standard_error,
            results[0].estimate.price.standard_error, mode == 0));
        price_errors.push_back(estimate.price.standard_error);
        deltas.push_back(estimate.delta.mean);
        delta_variances.push_back(estimate.delta.sample_variance);
        delta_reductions.push_back(reduction_percent(
            estimate.delta.standard_error,
            results[0].estimate.delta.standard_error, mode == 0));
        delta_errors.push_back(estimate.delta.standard_error);
        cpu_seconds.push_back(median(timings[mode]));
    }

    cf::utils::printVectors(
        cf::utils::TableConfig{cf::utils::FloatFormat::Default, 10},
        {"Configuration", "Price", "Price variance", "Price VR (%)", "Price SE",
         "Delta", "Delta variance", "Delta VR (%)", "Delta SE", "CPU (s)"},
        configurations, prices, price_variances, price_reductions, price_errors,
        deltas, delta_variances, delta_reductions, delta_errors, cpu_seconds);
}
struct Outputs {
    std::ofstream summary,timings,reference,batches,agreement,main_agreement,calibration;
    explicit Outputs(const std::string& p):summary(p+"_summary.csv"),timings(p+"_timings.csv"),
        reference(p+"_table_e1.csv"),batches(p+"_batches.csv"),agreement(p+"_agreement.csv"),
        main_agreement(p+"_main_agreement.csv"),
        calibration(p+"_calibration.csv")
    {
        for(auto* s:{&summary,&timings,&reference,&batches,&agreement,&main_agreement,&calibration}) {
            if(!*s) throw std::runtime_error("Cannot create benchmark output files.");
            *s<<std::setprecision(17);
        }
        summary<<"fund_sampling,T,N,contract,configuration,quantity,requested_paths,actual_paths,independent_samples,mean,sample_variance,standard_error,ci95_lower,ci95_upper,effective_variance,variance_reduction_pct,se_gain,cpu_seconds,cpu_times_se_squared,efficiency_vs_standard,seed\n";
        timings<<"fund_sampling,T,contract,repetition,configuration,paths,cpu_seconds,seed\n";
        reference<<"fund_sampling,T,contract,estimator,observed_variance,table_e1_variance,relative_difference_pct\n";
        batches<<"fund_sampling,T,contract,batch,configuration,quantity,paths,mean,standard_error,seed\n";
        agreement<<"fund_sampling,T,contract,quantity,comparison,batches,paths,mean_a,mean_b,mean_difference,difference_se,t_statistic,ci95_lower,ci95_upper,zero_in_ci95,simultaneous_family_size,simultaneous95_lower,simultaneous95_upper,zero_in_simultaneous95\n";
        main_agreement<<"fund_sampling,T,contract,quantity,comparison,actual_paths,calibration_batches,calibration_paths,mean_a,mean_b,mean_difference,difference_se,t_statistic,ci95_lower,ci95_upper,zero_in_ci95,simultaneous_family_size,simultaneous95_lower,simultaneous95_upper,zero_in_simultaneous95\n";
        calibration<<"fund_sampling,T,contract,configuration,quantity,batches,paths,grand_mean,rms_reported_se,observed_batch_sd,observed_over_reported\n";
    }
};
struct Counts {std::size_t total=0,individual_misses=0,simultaneous_misses=0;};
void validate(const Options& o,const Pricer::Prepared& p,int ti,int ci,std::uint64_t tag,
              const std::array<Measurement,4>& main_results,
              Outputs& output,Counts& counts,Counts& main_counts)
{
    if(!o.batches) return;
    std::array<std::array<std::vector<double>,2>,4> values,errors;
    const auto paths=even_paths(o.batch_paths);
    for(std::size_t batch=0;batch<o.batches;++batch) {
        const auto seed=mix(tag^UINT64_C(0x56414c4944415445)^static_cast<std::uint64_t>(batch));
        for(int mode=0;mode<4;++mode) {
            const auto r=run(p,paths,seed,mode,ci,o.fund);
            const std::array<MonteCarloEstimate,2> e{r.estimate.price,r.estimate.delta};
            for(int q=0;q<2;++q) {
                values[mode][q].push_back(e[q].mean);errors[mode][q].push_back(e[q].standard_error);
                output.batches<<fund_name(o.fund)<<','<<maturities[ti]<<','<<contract_names[ci]<<','<<batch<<','
                    <<mode_names[mode]<<','<<(q?"delta":"price")<<','<<paths<<','<<e[q].mean<<','<<e[q].standard_error<<','<<seed<<'\n';
            }
        }
    }
    const boost::math::students_t_distribution<double> t(o.batches-1);
    const double critical=boost::math::quantile(boost::math::complement(t,.025));
    const double simultaneous=boost::math::quantile(boost::math::complement(t,.05/(2*confidence_family)));
    for(int q=0;q<2;++q) {
        for(int mode=0;mode<4;++mode) {
            const double reported=rms(errors[mode][q]),observed=std::sqrt(variance(values[mode][q]));
            output.calibration<<fund_name(o.fund)<<','<<maturities[ti]<<','<<contract_names[ci]<<','<<mode_names[mode]<<','
                <<(q?"delta":"price")<<','<<o.batches<<','<<paths<<','<<mean(values[mode][q])<<','
                <<reported<<','<<observed<<','<<(reported>0?observed/reported:0)<<'\n';
        }
        for(auto [a,b]:contrasts) {
            std::vector<double> d(o.batches);
            for(std::size_t k=0;k<o.batches;++k) d[k]=values[a][q][k]-values[b][q][k];
            const double difference=mean(d),se=std::sqrt(variance(d)/o.batches);
            const double lo=difference-critical*se,hi=difference+critical*se;
            const double slo=difference-simultaneous*se,shi=difference+simultaneous*se;
            const bool zero=lo<=0 && 0<=hi,szero=slo<=0 && 0<=shi;
            ++counts.total;counts.individual_misses+=!zero;counts.simultaneous_misses+=!szero;
            output.agreement<<fund_name(o.fund)<<','<<maturities[ti]<<','<<contract_names[ci]<<','<<(q?"delta":"price")<<','
                <<mode_names[a]<<'-'<<mode_names[b]<<','<<o.batches<<','<<paths<<','<<mean(values[a][q])<<','<<mean(values[b][q])<<','
                <<difference<<','<<se<<','<<(se>0?difference/se:0)<<','<<lo<<','<<hi<<','<<zero<<','<<confidence_family<<','<<slo<<','<<shi<<','<<szero<<'\n';
            // Covariance-aware CI for the main equal-budget estimates. The
            // independently seeded calibration differences estimate variance
            // including shared RNG inputs. Variance scales as 1/path budget.
            const double main_a=q?main_results[a].estimate.delta.mean:main_results[a].estimate.price.mean;
            const double main_b=q?main_results[b].estimate.delta.mean:main_results[b].estimate.price.mean;
            const double main_difference=main_a-main_b;
            const double main_se=std::sqrt(variance(d)*static_cast<double>(paths)/main_results[a].paths);
            const double ml=main_difference-critical*main_se,mh=main_difference+critical*main_se;
            const double msl=main_difference-simultaneous*main_se,msh=main_difference+simultaneous*main_se;
            const bool mz=ml<=0 && 0<=mh,msz=msl<=0 && 0<=msh;
            ++main_counts.total;main_counts.individual_misses+=!mz;main_counts.simultaneous_misses+=!msz;
            output.main_agreement<<fund_name(o.fund)<<','<<maturities[ti]<<','<<contract_names[ci]<<','<<(q?"delta":"price")<<','
                <<mode_names[a]<<'-'<<mode_names[b]<<','<<main_results[a].paths<<','<<o.batches<<','<<paths<<','
                <<main_a<<','<<main_b<<','<<main_difference<<','<<main_se<<','<<(main_se>0?main_difference/main_se:0)<<','
                <<ml<<','<<mh<<','<<mz<<','<<confidence_family<<','<<msl<<','<<msh<<','<<msz<<'\n';
            if(!szero) std::cout<<"Simultaneous CI excludes zero: T="<<maturities[ti]<<' '<<contract_names[ci]
                <<' '<<(q?"delta":"price")<<' '<<mode_names[a]<<'-'<<mode_names[b]<<" ["<<slo<<", "<<shi<<"]\n";
            if(!msz) std::cout<<"Main equal-budget simultaneous CI excludes zero: T="<<maturities[ti]<<' '<<contract_names[ci]
                <<' '<<(q?"delta":"price")<<' '<<mode_names[a]<<'-'<<mode_names[b]<<" ["<<msl<<", "<<msh<<"]\n";
        }
    }
    output.batches.flush();output.agreement.flush();output.main_agreement.flush();output.calibration.flush();
}
void metadata(const Options& o)
{
    std::ofstream out(o.prefix+"_parameters.json");
    if(!out) throw std::runtime_error("Cannot write parameter metadata.");
    out<<std::setprecision(17)<<"{\n  \"makeham_intensity\": 0.0041959,\n  \"gompertz_scale\": 11.5818911,\n"
       <<"  \"gompertz_location\": 79.6921211,\n  \"initial_age\": 60,\n  \"gompertz_exponent\": \"(age+t-m)/b\",\n"
       <<"  \"r\": 0.05,\n  \"risk_aversion\": 1,\n  \"carbon_penalty\": 0.0025,\n"
       <<"  \"initial_fund\": 1,\n  \"notional\": 1,\n  \"death_benefit_fraction\": 1,\n"
       <<"  \"guarantee_growth_rates\": [0.05,0.5],\n  \"guarantees_fixed_in_delta\": true,\n"
       <<"  \"initial_equal_bounds_delta\": 0,\n  \"maturities\": [5,10,20,30],\n"
       <<"  \"intervals_per_year\": 5,\n  \"initial_carbon\": [5000,4000,3000,1000],\n"
       <<"  \"long_run_carbon\": [2500,2000,1500,500],\n  \"carbon_mean_reversion\": [0.05,0.05,0.05,0.05],\n"
       <<"  \"carbon_volatility\": [3,3,3,3],\n  \"asset_returns\": [0.25,0.15,0.1,0.08],\n"
       <<"  \"asset_volatility\": [0.3,0.25,0.2,0.16],\n"
       <<"  \"correlation\": [[1,0.44,0.39,0.32],[0.44,1,0.3,0.33],[0.39,0.3,1,0.31],[0.32,0.33,0.31,1]],\n"
       <<"  \"fund_sampling\": \""<<fund_name(o.fund)<<"\",\n  \"rng\": \"Boost xoshiro256pp + Ziggurat\",\n"
       <<"  \"requested_paths\": "<<o.paths<<",\n  \"actual_paths\": "<<even_paths(o.paths)<<",\n"
       <<"  \"timing_repetitions\": "<<o.repetitions<<",\n  \"validation_batches\": "<<o.batches<<",\n"
       <<"  \"validation_paths\": "<<(o.batches?even_paths(o.batch_paths):0)<<",\n  \"seed\": "<<o.seed<<",\n"
       <<"  \"confidence_family\": "<<confidence_family<<",\n  \"compiler\": \""<<__VERSION__<<"\",\n"
       <<"  \"openmp_parallel_simulation\": false\n}\n";
}
}
int main(int argc,char** argv)
{
    try {
        if(argc==2 && std::string_view(argv[1])=="--help") {
            std::cout<<"Options (each requires a value): --paths 1000000 --repetitions 3 --seed 20261015 "
                       "--batches 100 --batch-paths 10000 --fund-sampling path|marginals --output-prefix PATH\n";
            return 0;
        }
        const auto o=options(argc,argv);const auto paths=even_paths(o.paths);
        Outputs output(o.prefix);metadata(o);Counts counts,main_counts;
        std::cout<<"Paper experiment: xi=0.0041959, exponent=(age+t-m)/b, CIR NV, N=5T, "
                 <<paths<<" actual trajectories per configuration; fund="<<fund_name(o.fund)<<".\n"
                 <<"Individual 95% and simultaneous 95% intervals use a declared family of "<<confidence_family<<" comparisons.\n"<<std::flush;
        const boost::math::students_t_distribution<double> individual_t(paths-1),paired_t(paths/2-1);
        for(int ti=0;ti<4;++ti) {
            const Inputs inputs(maturities[ti]);const auto prepared=inputs.prepare();
            for(int ci=0;ci<3;++ci) {
                const auto tag=mix(o.seed^(static_cast<std::uint64_t>(ti+1)<<48)^
                    (static_cast<std::uint64_t>(ci+1)<<40));
                for(int mode=0;mode<4;++mode) static_cast<void>(run(prepared,std::min(paths,std::size_t{1000}),tag,mode,ci,o.fund));
                std::array<Measurement,4> first{};
                std::array<std::vector<double>,4> times;
                for(std::size_t repetition=0;repetition<o.repetitions;++repetition) {
                    const auto seed=mix(tag^static_cast<std::uint64_t>(repetition));
                    for(int order=0;order<4;++order) {
                        const int mode=static_cast<int>((order+repetition)%4);
                        const auto result=run(prepared,paths,seed,mode,ci,o.fund);
                        times[mode].push_back(result.cpu);
                        if(repetition==0) first[mode]=result;
                        output.timings<<fund_name(o.fund)<<','<<maturities[ti]<<','<<contract_names[ci]<<','<<repetition<<','
                            <<mode_names[mode]<<','<<paths<<','<<result.cpu<<','<<seed<<'\n';
                    }
                }
                const double baseline_cpu=median(times[0]);
                std::cout<<"T="<<maturities[ti]<<" N="<<5*maturities[ti]<<' '<<contract_names[ci]<<'\n';
                for(int mode=0;mode<4;++mode) {
                    const std::array<MonteCarloEstimate,2> e{first[mode].estimate.price,first[mode].estimate.delta};
                    const std::array<MonteCarloEstimate,2> base{first[0].estimate.price,first[0].estimate.delta};
                    const double cpu=median(times[mode]);
                    const auto& t=(mode%2)?paired_t:individual_t;
                    const double critical=boost::math::quantile(boost::math::complement(t,.025));
                    for(int q=0;q<2;++q) {
                        const double se2=e[q].standard_error*e[q].standard_error;
                        const double base_se2=base[q].standard_error*base[q].standard_error;
                        output.summary<<fund_name(o.fund)<<','<<maturities[ti]<<','<<5*maturities[ti]<<','<<contract_names[ci]<<','
                            <<mode_names[mode]<<','<<(q?"delta":"price")<<','<<o.paths<<','<<paths<<','<<first[mode].observations<<','
                            <<e[q].mean<<','<<e[q].sample_variance<<','<<e[q].standard_error<<','
                            <<e[q].mean-critical*e[q].standard_error<<','<<e[q].mean+critical*e[q].standard_error<<','
                            <<paths*se2<<',';
                        if(mode==0) output.summary<<0;
                        else if(base_se2>0) output.summary<<100*(1-se2/base_se2);
                        output.summary<<',';
                        if(e[q].standard_error>0) output.summary<<base[q].standard_error/e[q].standard_error;
                        output.summary<<','<<cpu<<','<<cpu*se2<<',';
                        if(cpu>0 && se2>0) output.summary<<baseline_cpu*base_se2/(cpu*se2);
                        output.summary<<','<<mix(tag)<<'\n';
                    }
                    if(mode==0 || mode==2) {
                        const double reference=mode==0?paper_standard[ci][ti]:paper_conditioned[ci][ti];
                        output.reference<<fund_name(o.fund)<<','<<maturities[ti]<<','<<contract_names[ci]<<','<<mode_names[mode]<<','
                            <<e[0].sample_variance<<','<<reference<<','<<100*(e[0].sample_variance/reference-1)<<'\n';
                    }
                }
                print_summary_table(first, times);
                output.summary.flush();output.timings.flush();output.reference.flush();std::cout<<std::flush;
                validate(o,prepared,ti,ci,tag,first,output,counts,main_counts);
                std::cout<<"  Validation batches completed: "<<o.batches<<'\n'<<std::flush;
            }
        }
        std::cout<<"Individual 95% intervals containing zero: "<<counts.total-counts.individual_misses<<'/'<<counts.total<<'\n'
                 <<"Simultaneous 95% intervals containing zero: "<<counts.total-counts.simultaneous_misses<<'/'<<counts.total<<'\n';
        std::cout<<"Main equal-budget individual 95% intervals containing zero: "<<main_counts.total-main_counts.individual_misses<<'/'<<main_counts.total<<'\n'
                 <<"Main equal-budget simultaneous 95% intervals containing zero: "<<main_counts.total-main_counts.simultaneous_misses<<'/'<<main_counts.total<<'\n';
        // A completed stochastic diagnostic need not have every interval include zero.
        return 0;
    } catch(const std::exception& e) {std::cerr<<"Benchmark error: "<<e.what()<<'\n';return 1;}
}

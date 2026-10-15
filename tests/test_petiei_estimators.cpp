#include <carbonfund/pricing/petiei_monte_carlo.hpp>
#include <boost/math/quadrature/gauss.hpp>
#include <Eigen/Cholesky>
#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>
#include <limits>
#include <numeric>
#include <span>
#include <stdexcept>
#include <type_traits>
#include <vector>

namespace {
namespace cf = carbonfund;
using namespace cf::pricing;
using Values = std::array<double, 6>;
void require(bool ok, const char* message) { if (!ok) throw std::runtime_error(message); }
void close(double a, double b, const char* message, double tolerance = 2e-11)
{
    require(std::isfinite(a) && std::isfinite(b) &&
            std::abs(a-b) <= tolerance * std::max(1.0, std::abs(b)), message);
}
Values means(const PetieiMonteCarloResult& r)
{
    return {r.pure_endowment.price.mean, r.pure_endowment.delta.mean,
            r.term_insurance.price.mean, r.term_insurance.delta.mean,
            r.endowment_insurance.price.mean, r.endowment_insurance.delta.mean};
}
std::array<MonteCarloEstimate, 6> estimates(const PetieiMonteCarloResult& r)
{
    return {r.pure_endowment.price, r.pure_endowment.delta,
            r.term_insurance.price, r.term_insurance.delta,
            r.endowment_insurance.price, r.endowment_insurance.delta};
}
struct CarbonTape {
    std::span<const Eigen::Vector4d> draws;
    std::size_t position = 0;
    void fill_std_normals(Eigen::Vector4d& out)
    {
        require(position < draws.size(), "Carbon tape exhausted.");
        out = draws[position++];
    }
};
struct FinancialTape {
    std::span<const double> draws;
    std::size_t position = 0;
    double std_normal()
    {
        require(position < draws.size(), "Financial tape exhausted.");
        return draws[position++];
    }
};
struct Fixture {
    PetieiMonteCarloParameters p;
    cf::portfolio::CarbonAllocationParameters4 a;
    Eigen::Vector4d state{0.8,1.1,1.3,0.6};
    Eigen::Vector4d mean{1.0,1.2,0.9,0.8};
    Eigen::Vector4d kappa = Eigen::Vector4d::Constant(0.5);
    Eigen::Vector4d sigma = Eigen::Vector4d::Constant(0.2);
    cf::contracts::CohortBook book{{{60.0,1.0}}};
    cf::mortality::GompertzMakeham mortality{{0.0041959,79.6921211,1.0/11.5818911}};
    Fixture()
    {
        p.fund_value=1.07; p.minimum_guarantee=0.8; p.maximum_guarantee=1.4;
        p.minimum_guarantee_growth_rate=0.01; p.maximum_guarantee_growth_rate=0.03;
        p.endowment_death_benefit_fraction=0.7; p.time_to_maturity=1.2; p.time_intervals=3;
        a.risk_free_rate=0.04; a.risk_aversion=2.0; a.carbon_penalty=0.3;
        a.expected_returns << 0.20,0.18,0.15,0.12;
        a.covariance << .09,.01,.005,.002, .01,.0625,.003,.001,
                        .005,.003,.04,.004, .002,.001,.004,.0256;
    }
    template<class Stepper> auto carbon() const
    {
        typename Stepper::Parameters c;
        c.mean_reversion=kappa; c.volatility=sigma;
        if constexpr (std::is_same_v<Stepper,cf::models::GeoOUExactStepper4>) c.long_run_log_mean=mean.array().log();
        else c.long_run_mean=mean;
        return c;
    }
    template<class Stepper> auto prepare() const
    {
        return PetieiMonteCarloPricer4<Stepper>::prepare(p,state,carbon<Stepper>(),a,
            {book.cohorts(),book.cohorts(),book.cohorts()},mortality);
    }
};
PayoffDelta direct(double fund, double initial, double lo, double hi)
{
    if (lo==hi) return {lo,0.0};
    if (fund<lo) return {lo,0.0};
    if (fund>hi) return {hi,0.0};
    return {fund,fund/initial*((fund==lo || fund==hi)?0.5:1.0)};
}
double survival(double t)
{
    const double b=1.0/11.5818911;
    return std::exp(-0.0041959*t-std::exp(b*(60.0-79.6921211))*std::expm1(b*t));
}
double death_weight(double t, double h, double r, bool endpoint)
{
    const double b=1.0/11.5818911;
    return h*(endpoint?0.5:1.0)*std::exp(-r*t)*survival(t)*
        (0.0041959+b*std::exp(b*(60+t-79.6921211)));
}
// Independent scalar transitions, a general matrix solve, and direct payoff
// equations provide a reference separate from the production hot loops.
template<class Stepper>
std::vector<Values> reference(const Fixture& x, std::span<const Eigen::Vector4d> carbon,
                            std::span<const double> financial, PetieiContract contract,
                            PetieiFundSampling coupling)
{
    const auto& p=x.p; const double h=p.time_to_maturity/p.time_intervals;
    const bool pe_only=contract==PetieiContract::pure_endowment;
    const std::size_t count=carbon.size()/p.time_intervals;
    std::vector<Values> out;
    std::size_t fpos=0;
    const auto q=[&](const Eigen::Vector4d& c) {
        Eigen::Matrix4d matrix=x.a.risk_aversion*x.a.covariance;
        matrix.diagonal().array() += x.a.carbon_penalty*x.a.covariance.diagonal().array()*c.array().max(0.0);
        const Eigen::Vector4d pi=matrix.ldlt().solve((x.a.expected_returns.array()-x.a.risk_free_rate).matrix());
        return pi.dot(x.a.covariance*pi);
    };
    for(std::size_t path=0;path<count;++path) {
        Eigen::Vector4d c=x.state;
        double prev=q(c),v=0,log_factor=0;
        const auto initial=direct(p.fund_value,p.fund_value,p.minimum_guarantee,p.maximum_guarantee);
        PayoffDelta death{initial.expected_payoff*death_weight(0,h,x.a.risk_free_rate,true),
                          initial.delta*death_weight(0,h,x.a.risk_free_rate,true)};
        PayoffDelta payoff{};
        for(std::size_t j=0;j<p.time_intervals;++j) {
            for(int k=0;k<4;++k) {
                const double kk=x.kappa[k],s=x.sigma[k],g=carbon[path*p.time_intervals+j][k];
                if constexpr(std::is_same_v<Stepper,cf::models::CIRNVStepper4>) {
                    const double drift=(kk*x.mean[k]-.25*s*s)*(-std::expm1(-.5*kk*h))/kk;
                    const double root=std::sqrt(drift*std::exp(-.5*kk*h)+std::exp(-kk*h)*c[k]);
                    c[k]=drift+std::pow(root+std::exp(-.25*kk*h)*.5*std::sqrt(h)*s*g,2);
                } else {
                    const double decay=std::exp(-kk*h);
                    const double noise=s*std::sqrt(-std::expm1(-2*kk*h)/(2*kk));
                    if constexpr(std::is_same_v<Stepper,cf::models::GeoOUExactStepper4>)
                        c[k]=std::exp(std::log(x.mean[k])*(1-decay)+decay*std::log(c[k])+noise*g);
                    else c[k]=x.mean[k]*(1-decay)+decay*c[k]+noise*g;
                }
            }
            const double current=q(c),increment=.5*h*(prev+current);
            v+=increment; prev=current;
            if(!pe_only) {
                require(fpos<financial.size(),"Reference financial tape exhausted.");
                const double f=financial[fpos++];
                if(coupling==PetieiFundSampling::path) log_factor+=-.5*increment+std::sqrt(increment)*f;
                else log_factor=-.5*v+std::sqrt(v)*f;
                const double t=(j+1)*h;
                payoff=direct(p.fund_value*std::exp(x.a.risk_free_rate*t+log_factor),p.fund_value,
                    p.minimum_guarantee*std::exp(p.minimum_guarantee_growth_rate*t),
                    p.maximum_guarantee*std::exp(p.maximum_guarantee_growth_rate*t));
                const double w=death_weight(t,h,x.a.risk_free_rate,j+1==p.time_intervals);
                death.expected_payoff+=payoff.expected_payoff*w;
                death.delta+=payoff.delta*w;
            }
        }
        if(pe_only) {
            log_factor=-.5*v+std::sqrt(v)*financial[fpos++];
            payoff=direct(p.fund_value*std::exp(x.a.risk_free_rate*p.time_to_maturity+log_factor),p.fund_value,
                p.minimum_guarantee*std::exp(p.minimum_guarantee_growth_rate*p.time_to_maturity),
                p.maximum_guarantee*std::exp(p.maximum_guarantee_growth_rate*p.time_to_maturity));
        }
        const double w=std::exp(-x.a.risk_free_rate*p.time_to_maturity)*survival(p.time_to_maturity);
        Values y{payoff.expected_payoff*w,payoff.delta*w,death.expected_payoff,death.delta,
                 payoff.expected_payoff*w+p.endowment_death_benefit_fraction*death.expected_payoff,
                 payoff.delta*w+p.endowment_death_benefit_fraction*death.delta};
        for(std::size_t k=0;k<6;++k) {
            if(contract!=PetieiContract::all && static_cast<int>(contract)!=1+static_cast<int>(k/2)) y[k]=0;
        }
        out.push_back(y);
    }
    return out;
}
void compare(const PetieiMonteCarloResult& actual, const std::vector<Values>& samples)
{
    require(actual.independent_samples==samples.size(),"Wrong independent sample count.");
    const auto stats=estimates(actual);
    for(std::size_t k=0;k<6;++k) {
        long double sum=0; for(const auto& s:samples) sum+=s[k];
        const long double mean=sum/samples.size();
        long double sq=0;for(const auto& s:samples) sq+=(s[k]-mean)*(s[k]-mean);
        const double var=static_cast<double>(sq/(samples.size()-1));
        close(stats[k].mean,static_cast<double>(mean),"Standard mean differs from scalar reference.");
        close(stats[k].sample_variance,var,"Standard sample variance mismatch.");
        close(stats[k].standard_error,std::sqrt(var/samples.size()),"Standard SE mismatch.");
    }
}
std::vector<Eigen::Vector4d> carbon_draws()
{
    return {{.2,-.4,.7,-.5},{-.6,.1,.3,.8},{.5,.9,-.2,.4},
            {-.8,.4,-.1,.3},{.6,-.7,.5,-.2},{.1,.3,-.4,.9}};
}
template<class Stepper> void check_reference()
{
    Fixture x; const auto prepared=x.prepare<Stepper>();
    const auto c=carbon_draws(); const std::vector<double> f{-.9,.2,.7,1.1,-.5,.4};
    using P=PetieiMonteCarloPricer4<Stepper>;
    for(auto contract:{PetieiContract::all,PetieiContract::pure_endowment,
                       PetieiContract::term_insurance,PetieiContract::endowment_insurance})
    for(auto coupling:{PetieiFundSampling::path,PetieiFundSampling::independent_marginals}) {
        CarbonTape cs{c};FinancialTape fs{f};
        const auto actual=P::evaluate(prepared,2,cs,fs,PetieiEstimator::standard,
            PetieiSampling::independent,contract,coupling);
        compare(actual,reference<Stepper>(x,c,f,contract,coupling));
        require(cs.position==6 && fs.position==(contract==PetieiContract::pure_endowment?2u:6u),
                "Wrong standard RNG consumption.");
        std::vector<Eigen::Vector4d> expanded_c;
        std::vector<double> expanded_f;
        const std::size_t nf=contract==PetieiContract::pure_endowment?1:3;
        for(std::size_t pair=0;pair<2;++pair) for(double sign:{1.0,-1.0}) {
            for(std::size_t j=0;j<3;++j) expanded_c.push_back(sign*c[3*pair+j]);
            for(std::size_t j=0;j<nf;++j) expanded_f.push_back(sign*f[nf*pair+j]);
        }
        const auto paths=reference<Stepper>(x,expanded_c,expanded_f,contract,coupling);
        std::vector<Values> pairs(2);
        for(std::size_t i=0;i<2;++i) for(std::size_t k=0;k<6;++k)
            pairs[i][k]=.5*(paths[2*i][k]+paths[2*i+1][k]);
        CarbonTape ca{c};FinancialTape fa{f};
        const auto anti=P::evaluate(prepared,4,ca,fa,PetieiEstimator::standard,
            PetieiSampling::antithetic,contract,coupling);
        require(anti.simulations==4,"Wrong actual antithetic trajectory count.");
        compare(anti,pairs);
    }
}
void check_conditioning_kernel()
{
    using Gauss=boost::math::quadrature::gauss<double,30>;
    for(double lo:{0.0,0.8}) for(double v:{.01,.2,1.3}) {
        MinMaxPayoffParameters p{1.1,1.08,lo,1.4};
        const auto prepared=MinMaxPayoff::prepare(p);
        const double root=std::sqrt(v),forward=p.fund_value*p.risk_free_growth;
        std::vector<double> cuts{-12,12};
        for(double bound:{lo,p.maximum_guarantee}) if(bound>0) {
            const double z=(std::log(bound/forward)+.5*v)/root;
            if(z>-12 && z<12) cuts.push_back(z);
        }
        std::sort(cuts.begin(),cuts.end());
        PayoffDelta integral{};
        for(std::size_t j=1;j<cuts.size();++j) for(int component=0;component<2;++component) {
            const auto integrand=[&](double z) {
                const auto value=MinMaxPayoff::evaluate_sample(prepared,-.5*v+root*z);
                return (component?value.delta:value.expected_payoff)*std::exp(-.5*z*z)/std::sqrt(2*std::acos(-1.0));
            };
            const double value=Gauss::integrate(integrand,cuts[j-1],cuts[j]);
            (component?integral.delta:integral.expected_payoff)+=value;
        }
        const auto conditioned=MinMaxPayoff::evaluate(prepared,v);
        close(conditioned.expected_payoff,integral.expected_payoff,"Conditioned payoff is not the financial expectation.");
        close(conditioned.delta,integral.delta,"Conditioned delta is not the financial expectation.");
    }
}
void check_delta_and_endpoint()
{
    Fixture x; x.p.fund_value=1; x.p.minimum_guarantee=1; x.p.maximum_guarantee=1;
    x.p.minimum_guarantee_growth_rate=.04; x.p.maximum_guarantee_growth_rate=.5;
    using P=PetieiMonteCarloPricer4<cf::models::OUExactStepper4>;
    const auto c=carbon_draws();const std::vector<double> f{-.9,.2,.7,1.1,-.5,.4};
    for(auto estimator:{PetieiEstimator::standard,PetieiEstimator::conditioned})
    for(auto sampling:{PetieiSampling::independent,PetieiSampling::antithetic}) {
        const std::size_t m=sampling==PetieiSampling::antithetic?4:2;
        const auto evaluate=[&](const Fixture& test) {
            CarbonTape cs{c};FinancialTape fs{f};
            return P::evaluate(test.prepare<cf::models::OUExactStepper4>(),m,cs,fs,estimator,sampling);
        };
        const auto actual=means(evaluate(x));
        auto up=x,down=x;const double bump=1e-6;
        up.p.fund_value+=bump;down.p.fund_value-=bump;
        const auto a=means(evaluate(up)),b=means(evaluate(down));
        for(std::size_t k=0;k<6;k+=2) close(actual[k+1],(a[k]-b[k])/(2*bump),
            "Fund delta differs from fixed-guarantee finite difference at equal initial bounds.",2e-8);
    }
    const auto p=MinMaxPayoff::prepare({1,1,1,2});
    close(MinMaxPayoff::evaluate_sample(p,0).delta,.5,"Single-kink symmetric derivative mismatch.");
}
void check_selection_and_validation()
{
    Fixture x;using P=PetieiMonteCarloPricer4<cf::models::OUExactStepper4>;
    const auto p=x.prepare<cf::models::OUExactStepper4>();const auto c=carbon_draws();
    const std::vector<double> f{-.9,.2,.7,1.1,-.5,.4};
    for(auto contract:{PetieiContract::pure_endowment,PetieiContract::term_insurance,PetieiContract::endowment_insurance}) {
        CarbonTape c1{c},c2{c};FinancialTape never{{}};
        const auto all=P::evaluate(p,4,c1);
        const auto one=P::evaluate(p,4,c2,never,PetieiEstimator::conditioned,PetieiSampling::antithetic,contract);
        require(never.position==0,"Conditioned estimator consumed financial normals.");
        const auto a=means(all),b=means(one);const std::size_t i=2*(static_cast<std::size_t>(contract)-1);
        close(a[i],b[i],"Selected conditioned price differs from joint evaluation.");
        close(a[i+1],b[i+1],"Selected conditioned delta differs from joint evaluation.");
    }
    const auto rejects=[&](auto action) { bool threw=false;try{action();}catch(const std::invalid_argument&){threw=true;}require(threw,"Invalid mode accepted."); };
    rejects([&]{CarbonTape cs{c};FinancialTape fs{f};static_cast<void>(P::evaluate(p,4,cs,fs,static_cast<PetieiEstimator>(9)));});
    rejects([&]{CarbonTape cs{c};FinancialTape fs{f};static_cast<void>(P::evaluate(p,4,cs,fs,PetieiEstimator::standard,PetieiSampling::antithetic,PetieiContract::all,static_cast<PetieiFundSampling>(9)));});
    rejects([&]{CarbonTape cs{c};static_cast<void>(P::evaluate(p,4,cs,PetieiSampling::antithetic,static_cast<PetieiContract>(9)));});
}
}
int main()
{
    try {
        check_conditioning_kernel();
        check_reference<cf::models::OUExactStepper4>();
        check_reference<cf::models::GeoOUExactStepper4>();
        check_reference<cf::models::CIRNVStepper4>();
        check_delta_and_endpoint();check_selection_and_validation();
        std::cout<<"Standard/conditioned estimators, E2/E3, datewise marginals, antithetic means/SEs and deltas passed.\n";
        return 0;
    } catch(const std::exception& e) {std::cerr<<e.what()<<'\n';return 1;}
}

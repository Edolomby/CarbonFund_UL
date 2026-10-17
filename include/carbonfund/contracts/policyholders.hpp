#pragma once

#include <carbonfund/contracts/cohorts.hpp>
#include <carbonfund/mortality/gompertz_makeham.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <map>
#include <span>
#include <stdexcept>
#include <utility>
#include <vector>

namespace carbonfund::contracts {

enum class PolicyType {
    pure_endowment,
    term_insurance,
    endowment_insurance
};

// One row represents one independent insured life, alive at scenario time zero.
// initial_age is the age at that date, not the age when the policy was issued.
// Policies of the same type share the pricing terms supplied to the pricer.
struct Policyholder {
    double initial_age = 0.0;
    double notional = 0.0;
    PolicyType type = PolicyType::pure_endowment;
};

struct PolicyholderTotals {
    std::size_t count = 0;
    double notional = 0.0;
};

struct PetieiPopulationTotals {
    PolicyholderTotals pure_endowment{};
    PolicyholderTotals term_insurance{};
    PolicyholderTotals endowment_insurance{};
};

// Owns both the initial policies and the mutable state of one mortality scenario.
// Copy once per worker if needed; reset() reuses all storage between scenarios.
// Empty populations and zero-notional policies are allowed. A zero-notional
// policy still represents a life and is included in the count and death draws.
class PolicyholderPopulation {
public:
    explicit PolicyholderPopulation(std::vector<Policyholder> policyholders = {})
        : policyholders_(std::move(policyholders)),
          cohort_indices_(policyholders_.size()),
          alive_(policyholders_.size(), 1)
    {
        std::array<std::map<double, std::size_t>, 3> indices;
        double total_notional = 0.0;

        for (std::size_t i = 0; i < policyholders_.size(); ++i) {
            const auto& policy = policyholders_[i];
            const std::size_t type = type_index(policy.type);
            if (!std::isfinite(policy.initial_age) || policy.initial_age < 0.0 ||
                !std::isfinite(policy.notional) || policy.notional < 0.0) {
                throw std::invalid_argument(
                    "Policyholder ages and notionals must be finite and nonnegative.");
            }
            total_notional += policy.notional;
            if (!std::isfinite(total_notional)) {
                throw std::invalid_argument("Total policyholder notional overflows.");
            }

            auto& book = books_[type];
            const auto [position, inserted] = indices[type].try_emplace(
                policy.initial_age, book.initial.size());
            if (inserted) {
                book.initial.push_back({policy.initial_age, 0.0});
            }
            cohort_indices_[i] = position->second;
            book.initial[position->second].total_notional += policy.notional;
            add_policy(initial_totals_[type], policy.notional);
        }

        for (auto& book : books_) {
            book.living = book.initial;
            book.death_probabilities.resize(book.initial.size());
        }
        reset();
    }

    // Stable policy identifiers are their original input indices.
    [[nodiscard]] std::span<const Policyholder> policyholders() const noexcept
    {
        return policyholders_;
    }

    [[nodiscard]] std::size_t size() const noexcept
    {
        return policyholders_.size();
    }

    [[nodiscard]] bool is_alive(std::size_t policy_index) const
    {
        return alive_.at(policy_index) != 0;
    }

    [[nodiscard]] double elapsed_time() const noexcept
    {
        return elapsed_time_;
    }

    [[nodiscard]] PetieiPopulationTotals initial_totals() const noexcept
    {
        return named_totals(initial_totals_);
    }

    [[nodiscard]] PetieiPopulationTotals living_totals() const noexcept
    {
        return named_totals(living_totals_);
    }

    // Direct input to the existing PETIEI pricer. Rows are grouped by type and
    // initial age, in first-occurrence order. Depleted rows remain with zero
    // notional. No allocation or copying; contents change on advance/reset.
    // The owner must outlive the view and must not be moved or assigned meanwhile.
    [[nodiscard]] PetieiPortfolioView living_portfolio() const noexcept
    {
        return {books_[0].living, books_[1].living, books_[2].living};
    }

    // Restore every initial policy, age and notional. Does not reset the RNG.
    void reset() noexcept
    {
        std::fill(alive_.begin(), alive_.end(), std::uint8_t{1});
        for (auto& book : books_) {
            std::copy(book.initial.begin(), book.initial.end(), book.living.begin());
        }
        living_totals_ = initial_totals_;
        elapsed_time_ = 0.0;
    }

    // Advance to an absolute elapsed time in years, never backwards.
    // Each surviving life dies with probability
    //   1 - S(initial_age + elapsed_time_, target_time - elapsed_time_).
    // Compute it once per cohort, then sample individuals in input order.
    // Rng must supply independent uniform01() values in [0, 1).
    // Consume one draw per living life when 0 < probability < 1; otherwise none.
    // Return the deaths in THIS interval, including their actual notionals.
    // Fund payoffs and death-claim settlement belong to the hedging layer.
    template<class Rng>
    [[nodiscard]] PetieiPopulationTotals advance_to(
        double target_time,
        const mortality::GompertzMakeham& mortality,
        Rng& rng)
    {
        if (!std::isfinite(target_time) || target_time < elapsed_time_) {
            throw std::invalid_argument(
                "Mortality target time must be finite and cannot go backwards.");
        }
        if (target_time == elapsed_time_) {
            return {};
        }

        const double horizon = target_time - elapsed_time_;
        // Prepare all probabilities and validate new ages before changing lives.
        for (auto& book : books_) {
            for (std::size_t j = 0; j < book.initial.size(); ++j) {
                if (!std::isfinite(book.initial[j].current_age + target_time)) {
                    throw std::invalid_argument("Updated policyholder age overflows.");
                }
                book.death_probabilities[j] = mortality.death_probability(
                    book.living[j].current_age, horizon);
            }
        }

        for (auto& book : books_) {
            for (std::size_t j = 0; j < book.initial.size(); ++j) {
                book.living[j] = {
                    book.initial[j].current_age + target_time, 0.0};
            }
        }
        living_totals_ = {};
        std::array<PolicyholderTotals, 3> deaths{};

        for (std::size_t i = 0; i < policyholders_.size(); ++i) {
            if (alive_[i] == 0) {
                continue;
            }
            const auto& policy = policyholders_[i];
            const std::size_t type = static_cast<std::size_t>(policy.type);
            auto& book = books_[type];
            const std::size_t cohort = cohort_indices_[i];
            const double probability = book.death_probabilities[cohort];
            const bool dies = probability == 1.0 ||
                (probability > 0.0 && rng.uniform01() < probability);

            if (dies) {
                alive_[i] = 0;
                add_policy(deaths[type], policy.notional);
            } else {
                // Sum actual survivors instead of subtracting large deaths from
                // a cohort total: a small surviving notional must not disappear.
                book.living[cohort].total_notional += policy.notional;
                add_policy(living_totals_[type], policy.notional);
            }
        }
        elapsed_time_ = target_time;
        return named_totals(deaths);
    }

private:
    struct ContractBook {
        std::vector<Cohort> initial;
        std::vector<Cohort> living;
        std::vector<double> death_probabilities;
    };

    std::vector<Policyholder> policyholders_;
    std::vector<std::size_t> cohort_indices_;
    std::vector<std::uint8_t> alive_;
    std::array<ContractBook, 3> books_;
    std::array<PolicyholderTotals, 3> initial_totals_{};
    std::array<PolicyholderTotals, 3> living_totals_{};
    double elapsed_time_ = 0.0;

    static std::size_t type_index(PolicyType type)
    {
        switch (type) {
        case PolicyType::pure_endowment: return 0;
        case PolicyType::term_insurance: return 1;
        case PolicyType::endowment_insurance: return 2;
        }
        throw std::invalid_argument("Unknown policyholder contract type.");
    }

    static void add_policy(PolicyholderTotals& totals, double notional) noexcept
    {
        ++totals.count;
        totals.notional += notional;
    }

    static PetieiPopulationTotals named_totals(
        const std::array<PolicyholderTotals, 3>& totals) noexcept
    {
        return {totals[0], totals[1], totals[2]};
    }
};

} // namespace carbonfund::contracts

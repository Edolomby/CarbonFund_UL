#pragma once

#include <cmath>
#include <span>
#include <stdexcept>
#include <utility>
#include <vector>

namespace carbonfund::contracts {

struct Cohort {
    // Age in years at the current valuation date.
    double current_age = 0.0;

    // Aggregate notional of the living policyholders in this row.
    // All policies in a row share the same pricing terms and mortality law.
    double total_notional = 0.0;
};

// Validated pricing input for one contract type.
// Empty books and zero-notional rows are allowed.
// Ages in every row must refer to the same valuation date.
class CohortBook {
public:
    explicit CohortBook(std::vector<Cohort> cohorts = {})
        : cohorts_(std::move(cohorts))
    {
        double total = 0.0;

        for (const auto& cohort : cohorts_) {
            if (!std::isfinite(cohort.current_age) ||
                !std::isfinite(cohort.total_notional) ||
                cohort.current_age < 0.0 ||
                cohort.total_notional < 0.0) {
                throw std::invalid_argument(
                    "Cohort ages and notionals must be finite and nonnegative.");
            }

            total += cohort.total_notional;

            if (!std::isfinite(total)) {
                throw std::invalid_argument(
                    "Total cohort notional overflows.");
            }
        }
    }

    // Read-only view; the book owns the rows.
    // No allocation or copying.
    [[nodiscard]] std::span<const Cohort> cohorts() const noexcept
    {
        return std::span<const Cohort>{cohorts_};
    }

    // For setup and reporting; not needed in the Monte Carlo path loop.
    [[nodiscard]] double total_notional() const noexcept
    {
        double total = 0.0;

        for (const auto& cohort : cohorts_) {
            total += cohort.total_notional;
        }

        return total;
    }

private:
    std::vector<Cohort> cohorts_;
};

// Borrowed pricing input for all three contract types.
// Rows can belong to CohortBooks or reusable hedging buffers.
// Owners must outlive this view; buffers must not be reallocated while in use.
struct PetieiPortfolioView {
    std::span<const Cohort> pure_endowment{};
    std::span<const Cohort> term_insurance{};
    std::span<const Cohort> endowment_insurance{};
};

} // namespace carbonfund::contracts
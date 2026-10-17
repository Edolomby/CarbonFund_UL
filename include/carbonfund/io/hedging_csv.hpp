#pragma once

#include <carbonfund/hedging/petiei_hedging.hpp>

#include <array>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <locale>
#include <span>
#include <stdexcept>

namespace carbonfund::io {

// One complete record per scenario prevents mismatched strategy-vector sizes.
// Files are replaced on rerun. Missing parent directories are created.
inline void write_hedging_csv(const std::filesystem::path& filename,
                             std::span<const hedging::PetieiHedgeOutcome> outcomes)
{
    for (const auto& row : outcomes) {
        const std::array<double, 8> values{row.no_hedge, row.static_hedge, row.dynamic_hedge,
            row.claims_pv, row.static_trading_gains_pv, row.dynamic_trading_gains_pv,
            row.fund_terminal, row.initial_price};
        for (double value : values)
            if (!std::isfinite(value))
                throw std::invalid_argument("Cannot export non-finite hedging results.");
    }
    if (filename.has_parent_path())
        std::filesystem::create_directories(filename.parent_path());
    std::ofstream file;
    file.exceptions(std::ios::badbit | std::ios::failbit);
    file.open(filename);
    file.imbue(std::locale::classic());
    file << std::setprecision(std::numeric_limits<double>::max_digits10);
    file << "scenario_id,no_hedge,static_hedge,dynamic_hedge,claims_pv,"
            "static_trading_gains_pv,dynamic_trading_gains_pv,fund_terminal,initial_price\n";
    for (const auto& row : outcomes) {
        file << row.scenario_id << ',' << row.no_hedge << ',' << row.static_hedge << ','
             << row.dynamic_hedge << ',' << row.claims_pv << ',' << row.static_trading_gains_pv << ','
             << row.dynamic_trading_gains_pv << ',' << row.fund_terminal << ',' << row.initial_price << '\n';
    }
    file.close();
}

} // namespace carbonfund::io

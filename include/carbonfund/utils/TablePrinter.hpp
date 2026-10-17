#pragma once
#include <algorithm>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

namespace carbonfund::utils {

// -------------------------------------------------------------------------
// Float formatting options
// -------------------------------------------------------------------------
enum class FloatFormat { Default, Scientific, Fixed };

struct TableConfig {
  FloatFormat format = FloatFormat::Default;
  int precision =
      6; // significant digits (Default) or mantissa digits (Sci/Fixed)
};

// -------------------------------------------------------------------------
// Helper: Convert any type to string, respecting TableConfig
// -------------------------------------------------------------------------
template <typename T>
std::string toString(const T &value, const TableConfig &cfg = {}) {
  if (cfg.precision < 0) {
    throw std::invalid_argument("Error: Precision must be nonnegative.");
  }

  std::ostringstream oss;
  if constexpr (std::is_floating_point_v<T>) {
    if (cfg.format == FloatFormat::Scientific)
      oss << std::scientific << std::setprecision(cfg.precision) << value;
    else if (cfg.format == FloatFormat::Fixed)
      oss << std::fixed << std::setprecision(cfg.precision) << value;
    else
      oss << std::defaultfloat << std::setprecision(cfg.precision) << value;
  } else {
    oss << value;
  }
  return oss.str();
}

// -------------------------------------------------------------------------
// Generic Table Printer for Vectors
//
// TableConfig is the FIRST argument (before headers) so that the variadic
// pack deduction is unambiguous. A convenience overload without config is
// provided so existing call sites continue to compile unchanged.
//
// Usage:
//   // default formatting (backward-compatible)
//   printVectors({"Col1","Col2"}, vec1, vec2);
//
//   // scientific notation with 4 mantissa digits
//   printVectors(TableConfig{FloatFormat::Scientific, 4},
//                {"Col1","Col2"}, vec1, vec2);
// -------------------------------------------------------------------------
template <typename... Ts>
void printVectors(const TableConfig &cfg,
                  const std::vector<std::string> &headers,
                  const std::vector<Ts> &...vecs) {

  // 1. Validation
  if (headers.size() != sizeof...(vecs)) {
    throw std::invalid_argument(
        "Error: Headers count must match the number of vectors.");
  }

  if (cfg.precision < 0) {
    throw std::invalid_argument("Error: Precision must be nonnegative.");
  }

  // With no columns, validation still applies but nothing is printed.
  // Discard the rest at compile time so std::max never receives an empty list.
  if constexpr (sizeof...(vecs) > 0) {

    auto get_str_val = [&](const auto &val) { return toString(val, cfg); };

    // 2. Calculate Column Widths
    std::vector<size_t> widths;
    for (const auto &header : headers)
      widths.push_back(header.length());

    size_t i = 0;
    auto calculate_widths = [&](const auto &vec) {
      for (const auto &item : vec)
        widths[i] = std::max(widths[i], get_str_val(item).length());
      i++;
    };
    (..., calculate_widths(vecs));

    // 3. Print Table
    // Restore the caller's formatting even if conversion or output throws.
    struct CoutFormatGuard {
      std::ios_base::fmtflags flags = std::cout.flags();
      std::streamsize precision = std::cout.precision();
      std::streamsize width = std::cout.width();
      char fill = std::cout.fill();

      ~CoutFormatGuard() {
        std::cout.flags(flags);
        std::cout.precision(precision);
        std::cout.fill(fill);
        std::cout.width(width);
      }
    } guard;

    std::cout.fill(' ');
    std::cout.width(0);

    auto print_separator = [&]() {
      std::cout << "+";
      for (const auto &w : widths)
        std::cout << std::string(w + 2, '-') << "+";
      std::cout << "\n";
    };

    // Top Border
    print_separator();

    // Headers
    std::cout << "|";
    for (size_t j = 0; j < headers.size(); ++j)
      std::cout << " " << std::left << std::setw(widths[j]) << headers[j] << " |";
    std::cout << "\n";

    // Separator
    print_separator();

    // Data Rows
    const size_t max_rows = std::max({vecs.size()...});
    for (size_t row = 0; row < max_rows; ++row) {
      std::cout << "|";
      size_t current_col = 0;
      auto print_cell = [&](const auto &vec) {
        std::cout << " " << std::left << std::setw(widths[current_col]);
        if (row < vec.size())
          std::cout << get_str_val(vec[row]);
        else
          std::cout << "";
        std::cout << " |";
        current_col++;
      };
      (..., print_cell(vecs));
      std::cout << "\n";
    }

    // Bottom Border
    print_separator();
  }
}

// Backward-compatible overload: no config → default formatting
template <typename... Ts>
void printVectors(const std::vector<std::string> &headers,
                  const std::vector<Ts> &...vecs) {
  printVectors(TableConfig{}, headers, vecs...);
}

} // namespace carbonfund::utils

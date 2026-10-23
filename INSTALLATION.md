# Installation

[README](README.md) · [Experiment guide](EXPERIMENTS.md)

Run project commands from the repository root. The commands below use a Linux shell and an editable Python environment named `.venv-analysis`.

## Requirements

| Component | Requirement |
|---|---|
| C++ compiler | C++20 support |
| CMake | 3.22 or newer |
| Eigen | 3.4 or newer within the Eigen 3 series |
| Boost | 1.88 or newer; Boost 1.90 is used in the validated configuration |
| OpenMP | Required when `CARBONFUND_ENABLE_OPENMP=ON` |
| Python | 3.10 or newer for the experiment interface |

The numerical library is header-only. The project consumes Eigen and Boost through their CMake package configurations and uses `Boost::headers`; no separately compiled Boost library is required by the current examples. The configurable runner compiles the Boost.JSON implementation in its own translation unit.

## Dependencies

On Ubuntu or Debian:

```bash
sudo apt update &&
sudo apt install build-essential cmake git libeigen3-dev libboost-dev \
    python3 python3-venv python3-pip
```

Distribution package versions vary. The installed Eigen and Boost versions must meet the requirements above. Check the selected Boost headers with:

```bash
grep -E '^#define BOOST_(VERSION|LIB_VERSION)' /usr/include/boost/version.hpp
test -f /usr/include/boost/random/xoshiro.hpp
```

CMake reports the Boost version and include directory during configuration. If system packages are too old, install a suitable version in an external prefix and pass that prefix to CMake. An unpacked Boost source directory alone is not equivalent to an installed Boost CMake package.

## Obtain the source

```bash
git clone https://github.com/Edolomby/CarbonFund_UL.git
cd CarbonFund_UL
```

For an existing checkout, run the following sections from that directory.

## Configure, build and test

Use Release mode for simulation timing and hedging experiments:

```bash
cmake -S . -B build-paper-release \
    -DCMAKE_BUILD_TYPE=Release \
    -DCARBONFUND_ENABLE_OPENMP=ON \
    -DBUILD_TESTING=ON &&
cmake --build build-paper-release --parallel &&
ctest --test-dir build-paper-release --output-on-failure --no-tests=error
```

On Linux, add `-DBoost_ROOT=/usr` to explicitly select the system Boost installation. For dependencies in external prefixes, use:

```bash
cmake -S . -B build-external \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_PREFIX_PATH="/path/to/eigen-prefix;/path/to/boost-prefix" \
    -DCARBONFUND_ENABLE_OPENMP=ON \
    -DBUILD_TESTING=ON
```

Replace the example prefixes with actual installation directories. `Boost_DIR` can alternatively identify the directory containing `BoostConfig.cmake`. Use a separate build directory when changing compilers or generators.

OpenMP parallelizes independent outer hedging scenarios in `carbonfund_experiment`. It does not parallelize the inner pricing loop. Set `CARBONFUND_ENABLE_OPENMP=OFF` for a serial build; hedging configurations must then request `threads` equal to `0` or `1`.

A Debug build can use a separate directory:

```bash
cmake -S . -B build-system-boost \
    -DCMAKE_BUILD_TYPE=Debug \
    -DCARBONFUND_ENABLE_OPENMP=ON \
    -DBUILD_TESTING=ON &&
cmake --build build-system-boost --parallel &&
ctest --test-dir build-system-boost --output-on-failure --no-tests=error
```

## Install the Python interface

```bash
python3 -m venv .venv-analysis &&
./.venv-analysis/bin/python -m pip install -e .
```

The package installs NumPy, SciPy and Matplotlib. Python configures experiments and analyses results; the simulation loops remain in C++.

Run the pricing demonstration:

```bash
./.venv-analysis/bin/python -m carbonfund_experiments \
    --config experiments/demo_pricing.json \
    --output results/quickstart_pricing \
    --binary build-paper-release/carbonfund_experiment
```

Run the hedging demonstration:

```bash
./.venv-analysis/bin/python -m carbonfund_experiments \
    --config experiments/demo_hedging.json \
    --output results/quickstart_hedging \
    --binary build-paper-release/carbonfund_experiment
```

Use a new output directory for another run, or explicitly pass `--overwrite`. See [EXPERIMENTS.md](EXPERIMENTS.md) for inputs, statistics and output files.

## Notebook kernel

Install a kernel in the same environment:

```bash
./.venv-analysis/bin/python -m pip install ipykernel
```

In VS Code, open the project folder, open a notebook, select **Select Kernel → Python Environments**, and choose `.venv-analysis/bin/python`. The notebook setup cell prints the selected Python executable and the C++ backend path.

To use JupyterLab instead:

```bash
./.venv-analysis/bin/python -m pip install jupyterlab &&
./.venv-analysis/bin/python -m jupyterlab
```

Start Jupyter from the repository root or a subdirectory. The notebooks locate the repository by its `include/carbonfund/` and `python/carbonfund_experiments.py` paths. The research notebooks expose large simulation budgets; use smaller explicitly chosen budgets for a first workflow check.

## Build diagnostics

- **Boost not found or too old:** inspect the configure output and select the intended installed prefix. Use a fresh build directory if the cache retains an old dependency path.
- **Benchmark source not found:** the benchmark sources belong in `benchmarks/`, and their `add_executable` paths must refer there.
- **TablePrinter header not found:** the header belongs at `include/carbonfund/utils/TablePrinter.hpp`; example sources include `<carbonfund/utils/TablePrinter.hpp>`.
- **Backend executable not found:** build the `carbonfund_experiment` target and pass its path with `--binary` if it is outside the standard build directories.
- **Only one outer worker:** enable OpenMP, rebuild the runner and inspect `metadata.json` fields `run.openmp_enabled` and `run.threads_used`. Python does not manage the worker threads.

The documented execution workflow has been exercised with Linux/GCC. macOS and Windows builds require platform-appropriate dependencies and compiler settings and have not been validated here. Windows also uses `.exe` executables and `.venv-analysis/Scripts/python.exe` for its virtual environment.

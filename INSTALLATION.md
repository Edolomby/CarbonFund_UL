# Installation and build

[Back to README](README.md)

These instructions cover dependencies, platform setup, configuration, testing and Release builds. Project commands run from the repository root unless stated otherwise.

## Requirements

- A C++20 compiler.
- CMake 3.22 or newer.
- A build tool such as Ninja or Make.
- Git to clone the repository.
- Eigen 3.4 or newer within the Eigen 3 series.
- Boost 1.88 or newer; the validated configuration uses Boost 1.90.0.
- Compiler OpenMP support when `CARBONFUND_ENABLE_OPENMP=ON`, the default.

CarbonFund_UL is a header-only library. The Eigen and Boost APIs it currently uses are also header-only, so the application does not link a compiled Boost library.

Dependencies are installed system-wide or in an external prefix. CMake uses the installed Eigen and Boost package configurations:

```cmake
find_package(Eigen3 3.4 REQUIRED NO_MODULE)
find_package(Boost 1.88.0 CONFIG REQUIRED)
```

Boost installation must provide `BoostConfig.cmake` and its headers package. An unpacked source directory alone is not the installed package expected by this configuration.

## Installation

### Ubuntu / Debian

```bash
sudo apt update
sudo apt install build-essential cmake ninja-build git libeigen3-dev libboost-dev
```

The available Eigen and Boost packages must meet the minimum versions above. Older distribution releases may supply an older Boost. Configuration reports the Boost version and include directory it actually selects.

The validated Ubuntu 26.04.1 setup uses system Boost 1.90.0 from `/usr/include`.

### macOS

Install Apple's Command Line Tools and [Homebrew](https://brew.sh/), then install the dependencies:

```bash
xcode-select --install
brew install cmake ninja git gcc@14 eigen@3 boost
```

The macOS configure command below selects Homebrew GCC and locates Eigen and Boost through their installation prefixes.

### Windows

Install [Visual Studio Build Tools](https://visualstudio.microsoft.com/downloads/) or Visual Studio with **Desktop development with C++**, a Windows SDK and **C++ CMake tools for Windows**. Also install [Git for Windows](https://git-scm.com/downloads/win).

Open **x64 Native Tools Command Prompt**, then run `powershell` inside it. This preserves the MSVC environment for the following commands.

Install Eigen 3.4.0 into an external prefix:

```powershell
$eigenWork = Join-Path $env:TEMP "carbonfund-eigen"
$eigenPrefix = Join-Path $env:USERPROFILE "carbonfund-deps/eigen-3.4.0"

New-Item -ItemType Directory -Force -Path $eigenWork | Out-Null

Invoke-WebRequest `
    -Uri "https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.zip" `
    -OutFile "$eigenWork/eigen-3.4.0.zip"

Expand-Archive -Force `
    -Path "$eigenWork/eigen-3.4.0.zip" `
    -DestinationPath $eigenWork

cmake -S "$eigenWork/eigen-3.4.0" -B "$eigenWork/build" -G Ninja `
    -DCMAKE_BUILD_TYPE=Release `
    "-DCMAKE_INSTALL_PREFIX=$eigenPrefix" `
    -DBUILD_TESTING=OFF `
    -DEIGEN_BUILD_DOC=OFF `
    -DCMAKE_POLICY_VERSION_MINIMUM=3.5

cmake --install "$eigenWork/build"
```

The policy minimum option supports Eigen 3.4.0's older configuration with CMake 4. Older CMake versions may report the option as unused.

Install Boost 1.90.0 into an external prefix, including its CMake package files:

```powershell
$boostWork = Join-Path $env:TEMP "carbonfund-boost"
$boostPrefix = Join-Path $env:USERPROFILE "carbonfund-deps/boost-1.90.0"
$boostArchive = Join-Path $boostWork "boost_1_90_0.zip"

New-Item -ItemType Directory -Force -Path $boostWork | Out-Null

Invoke-WebRequest `
    -Uri "https://archives.boost.io/release/1.90.0/source/boost_1_90_0.zip" `
    -OutFile $boostArchive

$expectedHash = "bdc79f179d1a4a60c10fe764172946d0eeafad65e576a8703c4d89d49949973c"
if ((Get-FileHash $boostArchive -Algorithm SHA256).Hash -ne $expectedHash) {
    throw "Boost archive checksum verification failed."
}

Expand-Archive -Force -Path $boostArchive -DestinationPath $boostWork

Push-Location "$boostWork/boost_1_90_0"
.\bootstrap.bat
.\b2.exe --with-system install "--prefix=$boostPrefix"
Pop-Location
```

The installation tool writes the headers and CMake package metadata. The project consumes `Boost::headers`.

The macOS and Windows procedures still await project build validation on those platforms.

## Obtain the project

```bash
git clone https://github.com/Edolomby/CarbonFund_UL.git
cd CarbonFund_UL
```

With GitHub SSH authentication configured:

```bash
git clone git@github.com:Edolomby/CarbonFund_UL.git
```

The following build commands run from the repository root.

## Configure, build and test

### Linux: system dependencies

```bash
cmake -S . -B build/debug -G Ninja \
    -DCMAKE_BUILD_TYPE=Debug \
    -DBUILD_TESTING=ON

cmake --build build/debug --parallel

ctest --test-dir build/debug --output-on-failure --no-tests=error

./build/debug/check_dependencies
./build/debug/example_petiei_monte_carlo
```

To explicitly select system Boost, add `-DBoost_ROOT=/usr` to the configure command.

### macOS

```bash
cmake -S . -B build/debug -G Ninja \
    -DCMAKE_BUILD_TYPE=Debug \
    -DBUILD_TESTING=ON \
    -DCMAKE_CXX_COMPILER="$(brew --prefix gcc@14)/bin/g++-14" \
    -DCMAKE_PREFIX_PATH="$(brew --prefix eigen@3);$(brew --prefix boost)"

cmake --build build/debug --parallel
ctest --test-dir build/debug --output-on-failure --no-tests=error
./build/debug/example_petiei_monte_carlo
```

### Windows: PowerShell

```powershell
$eigenPrefix = Join-Path $env:USERPROFILE "carbonfund-deps/eigen-3.4.0"
$boostPrefix = Join-Path $env:USERPROFILE "carbonfund-deps/boost-1.90.0"

cmake -S . -B build/debug -G Ninja `
    -DCMAKE_BUILD_TYPE=Debug `
    -DBUILD_TESTING=ON `
    -DCMAKE_CXX_COMPILER=cl `
    "-DCMAKE_PREFIX_PATH=$eigenPrefix;$boostPrefix"

cmake --build build/debug --parallel
ctest --test-dir build/debug --output-on-failure --no-tests=error
.\build\debug\example_petiei_monte_carlo.exe
```

For dependencies installed elsewhere, use `CMAKE_PREFIX_PATH` with their installation prefixes. `Boost_DIR` can instead name the directory containing `BoostConfig.cmake`.

### Release build for runtime measurements

On Linux, with system dependencies:

```bash
cmake -S . -B build/release -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DBUILD_TESTING=OFF

cmake --build build/release --parallel
./build/release/example_petiei_monte_carlo
```

Use a separate build directory for a different compiler or generator. `CARBONFUND_ENABLE_OPENMP=OFF` removes the OpenMP dependency. Enabling OpenMP currently makes it available to applications; the inner pricing loop remains serial.

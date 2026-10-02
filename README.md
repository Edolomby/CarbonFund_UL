# CarbonFund_UL
Monte Carlo pricing and hedging of insurance contracts with carbon-sensitive investment strategies

## Prerequisites and installation

### Requirements

- A compiler supporting C++20
- CMake 3.22 or newer
- Ninja
- Git
- Eigen 3.4 or newer within the Eigen 3 series
- Boost 1.88 or newer; the instructions below use Boost 1.90.0
- Compiler OpenMP support when `CARBONFUND_ENABLE_OPENMP=ON` (the default)

Eigen and the Boost components currently used are header-only.
Boost does not need to be compiled.

### Validation status

The dependency executable has been compiled and run with:

| Component | Tested version |
|-----------|----------------|
| Operating system | Ubuntu 24.04 |
| Compiler | GCC 13.3.0 |
| CMake | 3.28.3 |
| Ninja | 1.11.1 |
| Eigen | 3.4.0 |
| Boost | 1.90.0 |

The macOS and Windows instructions are installation procedures awaiting
project build validation. Package-manager versions can change over time.

### Linux: Ubuntu / Debian

Install the compiler, build tools, and Eigen:

```bash
sudo apt update
sudo apt install g++ cmake ninja-build git libeigen3-dev curl bzip2
```

Check that CMake is at least version 3.22 and Eigen is at least version 3.4.
Older distributions may require newer packages.

GCC provides OpenMP support through its compiler and runtime packages.

For other Linux distributions, install the equivalent packages using
their package manager, respecting the versions listed above.

### macOS: Intel and Apple Silicon

Install Apple's Command Line Tools:

```bash
xcode-select --install
```

Install [Homebrew](https://brew.sh/) if it is not already available.
Follow its installation instructions, including the shell configuration
shown by the installer.

Then install:

```bash
brew install cmake ninja git gcc@14 eigen@3
```

The build commands below explicitly select Homebrew GCC, which provides
OpenMP support. They also locate Homebrew directories automatically,
rather than assuming Intel or Apple Silicon paths.

### Windows: native MSVC build

Install:

1. [Visual Studio Build Tools](https://visualstudio.microsoft.com/downloads/)
   or Visual Studio, selecting **Desktop development with C++**.
2. The MSVC x64/x86 tools, a Windows SDK, and
   **C++ CMake tools for Windows** in the Visual Studio Installer.
3. [Git for Windows](https://git-scm.com/downloads/win).

Use a recent MSVC toolset supporting C++20, such as the toolset supplied
with Visual Studio 2022 or newer.

Open **x64 Native Tools Command Prompt** from the Start menu.
Run `powershell` inside that prompt. Use this PowerShell session for
the Windows commands below so that MSVC remains available.

Check that these commands work:

```powershell
cl
cmake --version
ninja --version
git --version
```

MSVC provides OpenMP support; no separate OpenMP installation is needed
for the current dependency check.

## Obtain the project

From the existing directory where you want to place the repository:

```bash
git clone https://github.com/Edolomby/CarbonFund_UL.git
cd CarbonFund_UL
```

These commands also work in PowerShell.

If GitHub SSH authentication is already configured, use this clone
command instead:

```bash
git clone git@github.com:Edolomby/CarbonFund_UL.git
```

## Obtain Boost 1.90.0

Run the following commands from the repository root.

### Linux / macOS

Download the official archive:

```bash
curl -fL \
  https://archives.boost.io/release/1.90.0/source/boost_1_90_0.tar.bz2 \
  -o /tmp/boost_1_90_0.tar.bz2
```

Verify its checksum.

On Linux:

```bash
echo "49551aff3b22cbc5c5a9ed3dbc92f0e23ea50a0f7325b0d198b705e8ee3fc305  /tmp/boost_1_90_0.tar.bz2" | sha256sum --check
```

On macOS:

```bash
echo "49551aff3b22cbc5c5a9ed3dbc92f0e23ea50a0f7325b0d198b705e8ee3fc305  /tmp/boost_1_90_0.tar.bz2" | shasum -a 256 --check
```

After verification reports `OK`, extract:

```bash
tar -xjf /tmp/boost_1_90_0.tar.bz2 -C .
```

### Windows: PowerShell

Download and verify the official ZIP archive:

```powershell
$boostArchive = Join-Path $env:TEMP "boost_1_90_0.zip"

Invoke-WebRequest `
    -Uri "https://archives.boost.io/release/1.90.0/source/boost_1_90_0.zip" `
    -OutFile $boostArchive

$expectedHash = "bdc79f179d1a4a60c10fe764172946d0eeafad65e576a8703c4d89d49949973c"

if ((Get-FileHash $boostArchive -Algorithm SHA256).Hash -ne $expectedHash) {
    throw "Boost archive checksum verification failed."
}

Expand-Archive -Path $boostArchive -DestinationPath .
```

On every platform, the resulting layout must contain:

```text
CarbonFund_UL/boost_1_90_0/boost/version.hpp
```

CMake automatically detects this local Boost directory unless an
external Boost root has been supplied. The directory is excluded from Git.

## Install Eigen on Windows

Linux and macOS users have already installed Eigen above.

In the Windows PowerShell session, download Eigen 3.4.0 into a temporary
directory and install its headers and CMake package into your user directory:

```powershell
$eigenWork = Join-Path $env:TEMP "carbonfund-eigen"
$eigenPrefix = Join-Path $env:USERPROFILE "carbonfund-deps/eigen-3.4.0"

New-Item -ItemType Directory -Force -Path $eigenWork | Out-Null

Invoke-WebRequest `
    -Uri "https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.zip" `
    -OutFile "$eigenWork/eigen-3.4.0.zip"

Expand-Archive `
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

The policy minimum option allows Eigen 3.4.0's older CMake configuration
to be used with CMake 4. Older CMake versions may report that this option
was unused.

## Configure, build, and check dependencies

Run these commands from the repository root.

### Linux

```bash
cmake -S . -B build/debug -G Ninja \
  -DCMAKE_BUILD_TYPE=Debug

cmake --build build/debug --target check_dependencies

./build/debug/check_dependencies
```

### macOS

```bash
cmake -S . -B build/debug -G Ninja \
  -DCMAKE_BUILD_TYPE=Debug \
  -DCMAKE_CXX_COMPILER="$(brew --prefix gcc@14)/bin/g++-14" \
  -DCMAKE_PREFIX_PATH="$(brew --prefix eigen@3)"

cmake --build build/debug --target check_dependencies

./build/debug/check_dependencies
```

### Windows: PowerShell

```powershell
$eigenPrefix = Join-Path $env:USERPROFILE "carbonfund-deps/eigen-3.4.0"

cmake -S . -B build/debug -G Ninja `
    -DCMAKE_BUILD_TYPE=Debug `
    -DCMAKE_CXX_COMPILER=cl `
    "-DCMAKE_PREFIX_PATH=$eigenPrefix"

cmake --build build/debug --target check_dependencies

.\build\debug\check_dependencies.exe
```

The executable reports dependency versions, an Eigen calculation,
a xoshiro sample, and OpenMP availability. This checks dependency
integration; numerical validation of the pricing routines will be
added separately.

Generated files in `build/` are excluded from Git.
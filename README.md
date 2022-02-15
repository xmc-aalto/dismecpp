# DiSMEC

## Building
The code can be build with cmake:
```shell script
mkdir build
cmake -DCMAKE_BUILD_TYPE=Release -DBUILD_DEVEL_DOCS=Off -S . -B ./build
cmake --build build --parallel
```
The resulting executables and python library will be placed in `./build/bin`
and `./build/lib` respectively. 

## Documentation
If you have doxygen installed, then you can build the documentation 
using `cmake --build build --target doxygen` and will find the documentation
in `build/docs/html`.

## Directories
### `src`
Contains the C++ code for the DiSMEC solver and the python bindings.

### `deps`
Contains sources/headers for libraries that we use as part of the implementation,
as git submodules. Currently, these are 
* https://github.com/CLIUtils/CLI11
* https://github.com/gabime/spdlog
* https://github.com/onqtam/doctest
* https://github.com/pybind/pybind11
* https://github.com/nlohmann/json
* https://gitlab.com/libeigen/eigen

Additionally, we expect Boost (>= 1.70) and an implementation of BLAS to be available on 
the system.

### `test_data`
Contains some data files that are needed for the automatic tests. These are e.g. results
generated using liblinear, where we can compare what our code produces.

# Warning Settings
`Dismec++` can be compiled with a large set of warnings enabled and treated as errors. In particular, 
with disabling only a small number of selected warnings, it is possible to compile using 
`-Wall -Wextra -Werror`. 
By default, the CMake script will add `-Wall` and `-Wextra`, but not `-Werror` to the compile flags.
For development, it is recommended to enable warnings as errors, for example by exporting
```shell
export CXXFLAGS=-Werror
```
before the first configuration run of CMake.

The automatic adjustment of warning flags is handled in the `cmake/compile-options.cmake`
file, which defines a an <tt>INTERFACE</tt> target `libdismec_config` which is linked to all
other targets.

## Non-Error Warnings
We know that in some situations the produced warnings are false positives. Therefore, these warnings
are never treated as errors, even if `-Werror` is present. The following warnings are not treated as 
errors:

### Unused Parameters
This warning complains about function parameters not being used in the definition of a function. In many
cases in the code, however, we have virtual functions that take a set of parameters the cover all the data
a *potential* implementation of this function might need. In each concrete case, though, only a subset of
these values are actually used. It would be possible to annotate all these definitions with corresponding
`[[maybe_unused]]` attributes, but for now we decided to make this warning ignorable instead of littering
the code with annotations. 

### Unused Variable
If testing is disabled (`DOCTEST_CONFIG_DISABLE` is defined), then the test case macros ensure that the
test cases do not end up in the binary. Further, the assert-style macros evaluate to nothing. 
However, at least Clang still parses the test cases' contents, which leads to some variables being 
seemingly unused, as their only use would be in an assert macro. 
Therefore, when compiling with Clang and `-DDOCTEST_CONFIG_DISABLE`, the warning configuration is extended
by `-Wno-error=unused-variable` and `-Wno-error=unused-but-set-variable`.

### Others
If the available boost version if less than 1.70, then `boost.histogram` is not available. This is not a
required component of the software, but its lack is diagnosed using a `#warning` preprocessor directive.
Consequently, in such cases these warnings are excluded from being treated as errors, through
`-Wno-error=cpp` (GCC) or `-Wno-error=\#warnings`.

GCC 8 seems to get confused by some part of our code, and reports an "unused static function" problem
for a non-static(!) member function when compiling `objective/dense_and_sparse.cpp`. Therefore, this 
warning is completely disabled for this source file.
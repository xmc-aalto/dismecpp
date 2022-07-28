# This is a helper target that collects the options common to all targets that we want to generate later on
add_library(libdismec_config INTERFACE)
target_compile_features(libdismec_config INTERFACE cxx_std_17)

set(IS_GCC "$<CXX_COMPILER_ID:GNU>")
set(IS_CLANG "$<CXX_COMPILER_ID:Clang>")
set(IS_GCC_8 "$<AND:${IS_GCC},$<AND:$<VERSION_GREATER_EQUAL:$<CXX_COMPILER_VERSION>,8>,$<VERSION_LESS:$<CXX_COMPILER_VERSION>,9>>>")
set(IS_GCC_11 "$<AND:${IS_GCC},$<AND:$<VERSION_GREATER_EQUAL:$<CXX_COMPILER_VERSION>,11>,$<VERSION_LESS:$<CXX_COMPILER_VERSION>,12>>>")
set(IS_CLANG_10 "$<AND:${IS_CLANG},$<AND:$<VERSION_GREATER_EQUAL:$<CXX_COMPILER_VERSION>,10>,$<VERSION_LESS:$<CXX_COMPILER_VERSION>,11>>>")
set(IS_CLANG_13_14 "$<AND:${IS_CLANG},$<AND:$<VERSION_GREATER_EQUAL:$<CXX_COMPILER_VERSION>,13>,$<VERSION_LESS:$<CXX_COMPILER_VERSION>,15>>>")

# enable all warning in GCC and Clang
# but make sure some of them are not treated as errors, even with -Werror
target_compile_options(libdismec_config INTERFACE "$<$<OR:${IS_GCC},${IS_CLANG}>:-Wall;-Wextra;-Wno-error=unused-parameter>")

if (${Boost_VERSION} VERSION_LESS 1.70)
    # Boost version has no histogram support, enable warning-as-error suppression
    target_compile_options(libdismec_config INTERFACE "$<${IS_GCC}:-Wno-error=cpp>")
    target_compile_options(libdismec_config INTERFACE "$<${IS_CLANG}:-Wno-error=\#warnings>")
endif ()

# The `unused-function` diagnostic seems to be a false positive on GCCs side, but I haven't found a way
# to disable it only locally with pragmas. So we set a source file property in the corresponding CMakeLists.txt file.
set(DISMEC_WORKAROUND_GCC8_BUG "$<${IS_GCC_8}:-Wno-unused-function>")

# on GCC-8 and Clang 10, we need to add a link command when using <filesystem>
target_link_libraries(libdismec_config INTERFACE "$<${IS_GCC_8}:-lstdc++fs>")
target_link_libraries(libdismec_config INTERFACE "$<${IS_CLANG_10}:-lstdc++fs>")

define_property(TARGET PROPERTY DoctestsEnabled BRIEF_DOCS "Set to ON if doctests should be compiled."
        FULL_DOCS [=[If this property is not specified, then the `libdismec_config` target will automatically
add a `DOCTEST_CONFIG_DISABLE` definition. Further, the warning suppressions appropriate to compiling with/without
doctest will be given.
]=])

set(HAS_DOCTEST "$<BOOL:$<TARGET_PROPERTY:DoctestsEnabled>>")
set(NO_DOCTEST "$<NOT:${HAS_DOCTEST}>")

target_compile_options(libdismec_config INTERFACE "$<$<AND:${IS_CLANG},${NO_DOCTEST}>:-Wno-error=unused-variable>")
target_compile_options(libdismec_config INTERFACE "$<$<AND:${IS_CLANG_13_14},${NO_DOCTEST}>:-Wno-error=unused-but-set-variable>")
target_compile_definitions(libdismec_config INTERFACE "$<${NO_DOCTEST}:DOCTEST_CONFIG_DISABLE>")

target_compile_definitions(libdismec_config INTERFACE "$<${HAS_DOCTEST}:DOCTEST_CONFIG_VOID_CAST_EXPRESSIONS>")

cmake_minimum_required(VERSION 3.15)
project(DiSMEC)

find_package(Threads REQUIRED)
find_package(Boost REQUIRED)

option(BUILD_DEVEL_DOCS "If this is set to ON, then the doxygen documentation generated will be to help develop
the library itself, i.e. it will contain all the internal documentation. If set to OFF, only the parts that are
visible to the outside will get documentation created." ON)
find_package(Doxygen)
if(${DOXYGEN_FOUND})
    set(DOXYGEN_GENERATE_HTML YES)
    set(DOXYGEN_BUILTIN_STL_SUPPORT YES)
    if(${BUILD_DEVEL_DOCS})
        set(DOXYGEN_EXTRACT_ALL YES)
        set(DOXYGEN_EXTRACT_STATIC YES)
        set(DOXYGEN_EXTRACT_PRIVATE YES)
        set(DOXYGEN_EXTRACT_ANON_NSPACES YES)
        set(DOXYGEN_INTERNAL_DOCS YES)
        set(DOXYGEN_SOURCE_BROWSER YES)
        set(DOXYGEN_REFERENCED_BY_RELATION YES)
        set(DOXYGEN_REFERENCES_RELATION YES)
        set(DOXYGEN_HIDE_UNDOC_RELATIONS NO)
        set(DOXYGEN_COLLABORATION_GRAPH YES)
        set(DOXYGEN_TEMPLATE_RELATIONS YES)
        set(DOXYGEN_CALL_GRAPH YES)
        set(DOXYGEN_CALLER_GRAPH YES)
        set(DOXYGEN_INTERACTIVE_SVG YES)
    else()
    endif()
    set(DOXYGEN_GENERATE_TREEVIEW YES)
    set(DOXYGEN_EXTRACT_PRIV_VIRTUAL YES)       # even in user mode, these may be needed for overriding some behaviour
    set(DOXYGEN_USE_MATHJAX YES)
    # set(DOXYGEN_CLANG_ASSISTED_PARSING YES)
    set(DOXYGEN_HTML_EXTRA_FILES bench/result.html)

    set(DOXYGEN_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/docs")
    doxygen_add_docs(
            doxygen
            ${PROJECT_SOURCE_DIR}/src
            COMMENT "Generate HTML docs"
    )
endif()

# pull in the dependencies which we have included in the deps directory
set(JSON_BuildTests OFF CACHE INTERNAL "")
set(JSON_MultipleHeaders ON)
add_subdirectory(deps/json)
# this is linked with the pybind library, so we need PIC
SET(CMAKE_POSITION_INDEPENDENT_CODE True)
add_subdirectory(deps/spdlog)
unset(CMAKE_POSITION_INDEPENDENT_CODE)

add_subdirectory(deps/CLI11 EXCLUDE_FROM_ALL)
add_subdirectory(deps/pybind11 EXCLUDE_FROM_ALL)
add_subdirectory(deps/nanobench EXCLUDE_FROM_ALL)
set(BUILD_TESTING Off)
add_subdirectory(deps/eigen EXCLUDE_FROM_ALL)

add_subdirectory(src)

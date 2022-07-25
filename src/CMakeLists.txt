set(LIB_SRC
        solver/cg.cpp
        solver/line_search.cpp
        solver/minimizer.cpp
        solver/newton.cpp
        objective/objective.cpp
        objective/reg_sq_hinge.cpp
        model/model.cpp
        model/sparse.cpp
        data/data.cpp
        data/transform.cpp
        io/xmc.cpp
        io/model-io.cpp
        io/prediction.cpp
        io/weights.cpp
        io/slice.cpp
        training/weighting.cpp
        training/init/constant.cpp
        training/init/subset.cpp
        training/init/pretrained.cpp
        training/init/zero.cpp
        training/init/multi_pos.cpp
        training/init/msi.cpp
        training/postproc/sparsify.cpp
        training/postproc/combine.cpp
        training/postproc/reorder.cpp
        utils/hash_vector.cpp
        io/common.cpp
        io/numpy.cpp
        utils/hyperparams.cpp
        prediction/metrics.cpp
        prediction/evaluate.cpp
        training/dismec.cpp
        model/dense.cpp
        parallel/numa.cpp
        parallel/runner.cpp
        utils/test_utils.cpp        # utilities for both testing and benchmarking, thus the currently appear here. TODO fix this
        training/postproc.cpp
        stats/collection.cpp
        stats/statistics.cpp
        training/statistics.cpp
        stats/histogram.cpp
        stats/tracked.cpp
        stats/timer.cpp
        objective/regularizers_imp.cpp
        solver/null.cpp
        objective/linear.cpp
        objective/generic_linear.cpp
        training/init/ova-primal.cpp
        objective/dense_and_sparse.cpp
        training/cascade.cpp
        training/init/numpy.cpp)

set(TESTS_SRC
        io/test.cpp
        io/common_test.cpp
        objective/reg_sg_hinge_test.cpp
        model/dense_test.cpp
        )

set_source_files_properties(objective/dense_and_sparse.cpp PROPERTIES COMPILE_OPTIONS ${DISMEC_WORKAROUND_GCC8_BUG})

target_include_directories(libdismec_config INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}  ${CMAKE_CURRENT_SOURCE_DIR}/../deps/doctest/doctest)
target_link_libraries(libdismec_config INTERFACE Eigen3::Eigen Threads::Threads spdlog::spdlog Boost::boost ${BLAS_LIBRARIES} nlohmann_json::nlohmann_json numa atomic)

# prevent multiple compilation
add_library(libdismec ${LIB_SRC})
target_link_libraries(libdismec PUBLIC libdismec_config)

# with GCC-8, lto causes an internal compiler error, so we disable the optimization
# With clang, we get a "file format not recognized" error during the linking step
#if ((CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS "9.0") OR CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
#    set_property(TARGET libdismec PROPERTY INTERPROCEDURAL_OPTIMIZATION FALSE)
#else()
#    set_property(TARGET libdismec PROPERTY INTERPROCEDURAL_OPTIMIZATION TRUE)
#endif ()


add_executable(unittest test.cpp ${LIB_SRC} ${TESTS_SRC})
add_executable(train train.cpp training/training.cpp app.cpp)
add_executable(cascade cascade.cpp training/training.cpp)
add_executable(predict predict.cpp prediction/prediction.cpp app.cpp)

set(PYBIND_SRC
        python/py_data.cpp
        python/py_train.cpp
        )

pybind11_add_module(pydismec MODULE pybind.cpp training/training.cpp ${LIB_SRC} ${PYBIND_SRC})

set_target_properties( train predict pydismec cascade
    PROPERTIES
    INTERPROCEDURAL_OPTIMIZATION_RELEASE TRUE
)

set_target_properties( unittest train predict pydismec cascade
        PROPERTIES
        ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
        LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
        )

target_link_libraries(predict PRIVATE CLI11::CLI11 libdismec)
target_link_libraries(train PRIVATE CLI11::CLI11 libdismec)
target_link_libraries(cascade PRIVATE CLI11::CLI11 libdismec)
target_link_libraries(unittest PRIVATE libdismec_config)
target_link_libraries(pydismec PRIVATE libdismec_config)

# enable actually compiling tests. The details are handled in `compile-options.cmake`
set_target_properties(unittest PROPERTIES DoctestsEnabled ON)

add_subdirectory(tools)

add_test(NAME unittest COMMAND unittest)

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
        training/init/avg_of_pos.cpp
        training/postproc/sparsify.cpp
        training/postproc/combine.cpp
        training/postproc/reorder.cpp
        hash_vector.cpp
        io/common.cpp
        io/numpy.cpp
        hyperparams.cpp
        prediction/metrics.cpp
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
        training/init/ova-primal.cpp)

set(TESTS_SRC io/test.cpp objective/reg_sg_hinge_test.cpp)

add_library(libdismec_config INTERFACE)
target_compile_features(libdismec_config INTERFACE cxx_std_17)
#target_compile_definitions(libdismec INTERFACE EIGEN_USE_BLAS)
target_include_directories(libdismec_config INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}  ${CMAKE_CURRENT_SOURCE_DIR}/../deps/doctest/doctest)
target_link_libraries(libdismec_config INTERFACE Eigen3::Eigen Threads::Threads spdlog::spdlog Boost::boost ${BLAS_LIBRARIES} nlohmann_json::nlohmann_json numa)

# prevent multiple compilation
add_library(libdismec ${LIB_SRC})
target_link_libraries(libdismec PUBLIC libdismec_config)
target_compile_definitions(libdismec PUBLIC DOCTEST_CONFIG_DISABLE)
set_property(TARGET libdismec PROPERTY INTERPROCEDURAL_OPTIMIZATION TRUE)

add_executable(test test.cpp ${LIB_SRC} ${TESTS_SRC})
add_executable(train train.cpp training/training.cpp)
add_executable(predict predict.cpp prediction/prediction.cpp)

set(PYBIND_SRC
        python/py_data.cpp
        python/py_train.cpp
        )

pybind11_add_module(pydismec MODULE pybind.cpp training/training.cpp ${LIB_SRC} ${PYBIND_SRC})

set_target_properties( test train predict pydismec
        PROPERTIES
        ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
        LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
        )

target_link_libraries(predict PRIVATE CLI11::CLI11 libdismec)
target_link_libraries(train PRIVATE CLI11::CLI11 libdismec)
target_link_libraries(test PRIVATE libdismec_config)
target_link_libraries(pydismec PRIVATE libdismec_config)

# enable or disable tests as desired
target_compile_definitions(pydismec PRIVATE DOCTEST_CONFIG_DISABLE)
target_compile_definitions(test PRIVATE DOCTEST_CONFIG_VOID_CAST_EXPRESSIONS)

add_subdirectory(tools)
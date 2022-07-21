// Copyright (c) 2021, Aalto University, developed by Erik Schultheis
// All rights reserved.
//
// SPDX-License-Identifier: MIT

#include "io/xmc.h"
#include "io/common.h"
#include "data/data.h"
#include <vector>
#include "doctest.h"
using namespace dismec;


constexpr const char* TEST_FILE = \
R"(4 10 5
2,3 4:1.0 5:-0.5 8:0.25
0 2:1.0
 6:-2.0 5:1.5
1, 2 3:-3.0
)";


/*!
 * \test This test case verifies that reading and writing XMC files round-trips.
 */
TEST_CASE("xmc round trip") {
    std::stringstream original_source;
    original_source.str(TEST_FILE);

    SparseFeatures features(4, 10);
    features.coeffRef(0, 4) = 1.0;
    features.coeffRef(0, 5) = -0.5;
    features.coeffRef(0, 8) = 0.25;
    features.coeffRef(1, 2) = 1.0;
    features.coeffRef(2, 6) = -2.0;
    features.coeffRef(2, 5) = 1.5;
    features.coeffRef(3, 3) = -3.0;

    std::vector<std::vector<long>> label_ex(5);
    label_ex[0].push_back(1);
    label_ex[1].push_back(3);
    label_ex[2].push_back(0);
    label_ex[3].push_back(0);

    MultiLabelData data(features, label_ex);

    std::stringstream canonical_save;
    io::save_xmc_dataset(canonical_save, data);

    auto re_read = io::read_xmc_dataset(canonical_save, "test");
    REQUIRE(re_read.get_features()->rows() == features.rows());
    REQUIRE(re_read.get_features()->cols() == features.cols());
    CHECK(types::DenseColMajor<real_t>(re_read.get_features()->sparse()) == types::DenseColMajor<real_t>(features));

    std::stringstream round_trip;
    io::save_xmc_dataset(round_trip, re_read);

    CHECK(round_trip.str() == canonical_save.str());
}
// Copyright (c) 2021, Aalto University, developed by Erik Schultheis
// All rights reserved.
//
// SPDX-License-Identifier: MIT

#include <iostream>
#include <fstream>
#include "slice.h"
#include "data/data.h"
#include "io/numpy.h"
#include "io/common.h"
#include <spdlog/spdlog.h>
#include <spdlog/stopwatch.h>

using namespace dismec;
namespace io = dismec::io;

namespace {
    DenseFeatures load_features(std::istream& features) {
        if(io::is_npy(features)) {
            return io::load_matrix_from_npy(features);
        }

        std::string line_buffer;
        std::getline(features, line_buffer);
        io::MatrixHeader header = io::parse_header(line_buffer);
        DenseFeatures target(header.NumRows, header.NumCols);

        for(int row = 0; row < header.NumRows; ++row) {
            io::read_vector_from_text(features, target.row(row));
        }
        return target;
    }
}

dismec::MultiLabelData io::read_slice_dataset(std::istream& features, std::istream& labels) {
    spdlog::stopwatch timer;
    DenseFeatures feature_matrix = load_features(features);

    auto label_data = read_binary_matrix_as_lil(labels);

    if(label_data.NumRows != feature_matrix.rows()) {
        THROW_ERROR("Mismatch between number of examples in feature file ({}) and in label file ({})",
                    feature_matrix.rows(), label_data.NumRows);
    }

    spdlog::info("Finished loading dataset with {} examples in {:.3}s.", label_data.NumCols, timer);

    return MultiLabelData(std::move(feature_matrix), std::move(label_data.NonZeros));
}

dismec::MultiLabelData io::read_slice_dataset(const std::filesystem::path& features, const std::filesystem::path& labels) {
    std::fstream features_file(features, std::fstream::in);
    if (!features_file.is_open()) {
        throw std::runtime_error(fmt::format("Cannot open input file {}", features.c_str()));
    }
    std::fstream labels_file(labels, std::fstream::in);
    if (!labels_file.is_open()) {
        throw std::runtime_error(fmt::format("Cannot open input file {}", labels.c_str()));
    }

    return read_slice_dataset(features_file, labels_file);
}


#include "doctest.h"

using namespace dismec;

TEST_CASE("small dataset") {
    std::stringstream features;
    std::stringstream labels;

    features.str("3 5\n"
                 "1.0  2.5  -1.0  3.5  4.4\n"
                 "-1.0 0.0   0.5  2.5  1.5\n"
                 "0.0   5.4\t 3.4   2.5 1.6\n");

    labels.str("3 3\n"
               "1:1\n"
               "0:1\n"
               "0:1 2:1"
    );

    auto ds = io::read_slice_dataset(features, labels);

    auto df = ds.get_features()->dense();
    REQUIRE(df.rows() == 3);
    REQUIRE(df.cols() == 5);
    float true_features[] = {1.0, 2.5, -1.0, 3.5, 4.4, -1.0, 0.0, 0.5, 2.5, 1.5, 0.0, 5.4, 3.4, 2.5, 1.6};
    for(int i = 0; i < df.size(); ++i) {
        CHECK(df.coeff(i) == true_features[i]);
    }

    // check the labels
    const auto& l0 = ds.get_label_instances(label_id_t{0});
    REQUIRE(l0.size() == 2);
    CHECK(l0[0] == 1);
    CHECK(l0[1] == 2);

    const auto& l1 = ds.get_label_instances(label_id_t{1});
    REQUIRE(l1.size() == 1);
    CHECK(l1[0] == 0);

    const auto& l2 = ds.get_label_instances(label_id_t{2});
    REQUIRE(l2.size() == 1);
    CHECK(l2[0] == 2);
}
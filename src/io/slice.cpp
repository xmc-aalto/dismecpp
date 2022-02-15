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

using namespace io;

namespace {
    DenseFeatures load_numpy_features(std::streambuf& source) {
        auto header = parse_npy_header(source);
        if(header.DataType != data_type_string<real_t>()) {
            THROW_ERROR("Unsupported data type {}", header.DataType);
        }
        if(header.ColumnMajor) {
            THROW_ERROR("Currently, only row-major npy files can be read");
        }

        DenseFeatures target(header.Rows, header.Cols);
        binary_load(source, target.data(), target.data() + header.Rows * header.Cols);
        return target;
    }

    /// \brief Collects the data from the header of a slice feature file in text format \ref slice-data.
    struct SliceHeader {
        long NumRows;
        long NumCols;
    };

    SliceHeader parse_header(const std::string& content) {
        std::stringstream parse_header{content};
        long NumRows;
        long NumCols;

        parse_header >> NumRows >> NumCols;
        if (parse_header.fail()) {
            THROW_ERROR("Error parsing dataset header: '{}'", content);
        }

        // check validity of numbers
        if(NumRows <= 0) {
            THROW_ERROR("Invalid number of examples {} specified in header '{}'", NumRows, content);
        }
        if(NumCols <= 0) {
            THROW_ERROR("Invalid number of features {} specified in header '{}'", NumCols, content);
        }

        std::string rest;
        parse_header >> rest;
        if(!rest.empty()) {
            THROW_ERROR("Found additional text '{}' in header '{}'", rest, content);
        }

        return {NumRows, NumCols};
    }

    DenseFeatures load_features(std::istream& features) {
        if(io::is_npy(features)) {
            return load_numpy_features(*features.rdbuf());
        } else {
            std::string line_buffer;
            std::getline(features, line_buffer);
            SliceHeader header = parse_header(line_buffer);
            DenseFeatures target(header.NumRows, header.NumCols);

            for(int row = 0; row < header.NumRows; ++row) {
                read_vector_from_text(features, target.row(row));
            }
            return target;
        }
    }
}
#include "doctest.h"
MultiLabelData io::read_slice_dataset(std::istream& features, std::istream& labels) {
    spdlog::stopwatch timer;
    DenseFeatures feature_matrix = load_features(features);

    // for now, labels are assumed to come from a text file
    std::string line_buffer;
    std::getline(labels, line_buffer);
    auto header = parse_header(line_buffer);

    std::vector<std::vector<long>> label_data;
    label_data.resize(header.NumCols);
    // TODO reserve space for correct number of labels

    long example = 0;
    long num_rows = header.NumRows;
    long num_cols = header.NumCols;

    if(num_rows != feature_matrix.rows()) {
        THROW_ERROR("Mismatch between number of examples in feature file ({}) and in label file ({})",
                    feature_matrix.rows(), num_rows);
    }

    while (std::getline(labels, line_buffer)) {
        if (line_buffer.empty())
            continue;
        if (line_buffer.front() == '#')
            continue;

        if(example >= num_rows) {
            THROW_ERROR("Encountered row {:5} but buffers only expect {:5} examples.", example, num_rows);
        }

        try {
            io::parse_sparse_vector_from_text(line_buffer.c_str(), [&](long index, double value) {
                long adjusted_index = index;
                if (adjusted_index >= num_cols || adjusted_index < 0) {
                    THROW_ERROR("Encountered label {:5}. Number of labels "
                                "was specified as {}.", index, num_cols);
                }
                // filter out explicit zeros
                if (value != 1) {
                    THROW_ERROR("Encountered label {:5} with value {}.", index, value);
                }
                label_data[adjusted_index].push_back(example);
            });
        } catch (std::runtime_error& e) {
            THROW_ERROR("Error reading example {}: {}.", example + 1, e.what());
        }
        ++example;
    }

    spdlog::info("Finished loading dataset with {} examples in {:.3}s.", example, timer);

    return MultiLabelData(std::move(feature_matrix), std::move(label_data));
}

MultiLabelData io::read_slice_dataset(const std::filesystem::path& features, const std::filesystem::path& labels) {
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

//! \test Checks that valid XMC headers are parsed correctly
TEST_CASE("parse valid header") {
    std::string input;
    SUBCASE("minimal") {
        input = "12 54";
    }
    SUBCASE("trailing space") {
        input = "12 54 ";
    }
    SUBCASE("tab separated") {
        input = "12\t 54";
    }
    SliceHeader valid = parse_header(input);
    CHECK(valid.NumRows == 12);
    CHECK(valid.NumCols == 54);
}

/// \test Check that invalid XMC headers are causing an exception. The headers are invalid if either the number of
/// data does not match, of if any of the supplied counts are non-positive.
TEST_CASE("parse invalid header") {
    // check number of arguments
    CHECK_THROWS(parse_header("6 "));
    CHECK_THROWS(parse_header("6 1 5"));

    // we also know that something is wrong if any of the counts are <= 0
    CHECK_THROWS(parse_header("0 5"));
    CHECK_THROWS(parse_header("5 0"));
    CHECK_THROWS(parse_header("-1 5"));
    CHECK_THROWS(parse_header("5 -1"));
}


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

    auto ds = read_slice_dataset(features, labels);

    auto df = ds.get_features()->dense();
    REQUIRE(df.rows() == 3);
    REQUIRE(df.cols() == 5);
    float true_features[] = {1.0, 2.5, -1.0, 3.5, 4.4, -1.0, 0.0, 0.5, 2.5, 1.5, 0.0, 5.4, 3.4, 2.5, 1.6};
    for(int i = 0; i < df.size(); ++i) {
        CHECK(df.coeff(i) == true_features[i]);
    }

    // check the labels
    auto& l0 = ds.get_label_instances(label_id_t{0});
    REQUIRE(l0.size() == 2);
    CHECK(l0[0] == 1);
    CHECK(l0[1] == 2);

    auto& l1 = ds.get_label_instances(label_id_t{1});
    REQUIRE(l1.size() == 1);
    CHECK(l1[0] == 0);

    auto& l2 = ds.get_label_instances(label_id_t{2});
    REQUIRE(l2.size() == 1);
    CHECK(l2[0] == 2);
}
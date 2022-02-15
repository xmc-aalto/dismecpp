// Copyright (c) 2021, Aalto University, developed by Erik Schultheis
// All rights reserved.
//
// SPDX-License-Identifier: MIT

#include "io/prediction.h"
#include "io/common.h"
#include <fstream>

void io::prediction::save_sparse_predictions(path target_file,
                                             const PredictionMatrix& values,
                                             const IndexMatrix& indices) {
    std::fstream file(target_file, std::fstream::out);
    save_sparse_predictions(file, values, indices);
}

void io::prediction::save_sparse_predictions(std::ostream& target,
                             const PredictionMatrix& values,
                             const IndexMatrix& indices) {
    if(values.rows() != indices.rows()) {
        throw std::invalid_argument(fmt::format("Inconsistent number of rows of values ({}) and indices ({}).",
                                    values.rows(), indices.rows()));
    }
    if(values.cols() != indices.cols()) {
        throw std::invalid_argument(fmt::format("Inconsistent number of columns of values ({}) and indices ({}).",
                                    values.rows(), indices.rows()));
    }

    long last_col = values.cols() - 1;

    // write the header
    target << values.rows() << " " << values.cols() << "\n";
    /// \todo write label range?
    for(int row = 0; row < values.rows(); ++row) {
        for(int col = 0; col < last_col; ++col) {
            target << indices.coeff(row, col) << ":" << values.coeff(row, col) << " ";
        }
        target << indices.coeff(row, last_col) << ":" << values.coeff(row, last_col) << '\n';
    }
}

std::pair<IndexMatrix, PredictionMatrix> io::prediction::read_sparse_prediction(std::istream& source) {
    std::string line_buffer;
    long rows, cols;
    {
        if(!std::getline(source, line_buffer)) {
            throw std::runtime_error("Error while reading header");
        }
        std::stringstream parsing(line_buffer);
        parsing >> rows >> cols;
        if(parsing.bad()) {
            throw std::runtime_error("Error while parsing header");
        }

        if(rows <= 0) {
            throw std::runtime_error(fmt::format("Invalid number of rows {} specified.", rows));
        }

        if(cols <= 0) {
            throw std::runtime_error(fmt::format("Invalid number of columns {} specified.", cols));
        }
    }

    IndexMatrix indices(rows, cols);
    PredictionMatrix values(rows, cols);
    long current_row = 0;

    for(; current_row < rows; ++current_row) {
        if(!std::getline(source, line_buffer)) {
            throw std::runtime_error(fmt::format("Error while reading predictions for instance {}", current_row));
        }
        long k = 0;
        parse_sparse_vector_from_text(line_buffer.c_str(), [&](long index, double value)
        {
            if(k >= cols) {
                THROW_ERROR("Got more predictions than expected ({}) for instance {}", cols, current_row);
            }
            indices.coeffRef(current_row, k) = index;
            values.coeffRef(current_row, k) = value;
            ++k;
        });
        if(k != cols) {
            THROW_ERROR("Expected {} columns, but got only {}", cols, k);
        }
    }

    if(current_row != rows) {
        THROW_ERROR("Expected {} rows, but got only {}", rows, current_row);
    }

    return {std::move(indices), std::move(values)};
}

std::pair<IndexMatrix, PredictionMatrix> io::prediction::read_sparse_prediction(path source) {
    std::fstream stream(source, std::fstream::in);
    return read_sparse_prediction(stream);
}

void io::prediction::save_dense_predictions(path target, const PredictionMatrix & values) {
    std::fstream file(target, std::fstream::out);
    save_dense_predictions(file, values);
}

void io::prediction::save_dense_predictions(std::ostream& target, const PredictionMatrix& values) {
    target << values.rows() << " " << values.cols() << "\n";
    for(int row = 0; row < values.rows(); ++row) {
        io::write_vector_as_text(target, values.row(row)) << '\n';
    }
}

#include "doctest.h"

/*!
 * \test Checks in an example that sparse predictions writes as expected. It also checks that
 * the reading back gives the same result.
 */
TEST_CASE("save_load_sparse_predictions")
{
    IndexMatrix indices(2, 3);
    PredictionMatrix values(2, 3);
    indices.row(0) << 0, 2, 1;
    indices.row(1) << 1, 31, 2;
    values.row(0) << 0.5, 1.5, 0.9;
    values.row(1) << 1.5, 0.9, 0.4;
    std::string as_text =
            "2 3\n"
            "0:0.5 2:1.5 1:0.9\n"
            "1:1.5 31:0.9 2:0.4\n";

    SUBCASE("save") {
        std::stringstream target;
        io::prediction::save_sparse_predictions(target, values, indices);
        CHECK(target.str() == as_text);
    }

    SUBCASE("load") {
        std::stringstream source(as_text);
        auto loaded = io::prediction::read_sparse_prediction(source);
        CHECK(loaded.first == indices);
        CHECK(loaded.second == values);
    }
    SUBCASE("load changed whitespace") {
        std::stringstream source("2  3\t\n"
                                 "0:0.5 2: 1.5  1:0.9\n"
                                 "1:1.5\t31:0.9 2:0.4");
        auto loaded = io::prediction::read_sparse_prediction(source);
        CHECK(loaded.first == indices);
        CHECK(loaded.second == values);
    }
}

/*!
 * \test The test case checks that `save_sparse_predictions` verifies the compatibility of
 * value and index matrices.
 */
TEST_CASE("save_sparse_predictions checking") {
    IndexMatrix indices(2, 3);
    std::stringstream target;
    SUBCASE("mismatched rows") {
        PredictionMatrix values(3, 3);
        CHECK_THROWS(io::prediction::save_sparse_predictions(target, values, indices));
    }
    SUBCASE("mismatched columns") {
        PredictionMatrix values(2, 2);
        CHECK_THROWS(io::prediction::save_sparse_predictions(target, values, indices));
    }
}

/*!
 * \test This test case verifies that (certain) erroneous prediction files are rejected.
 */
TEST_CASE("read_sparse_prediction check") {
    std::stringstream source;
    SUBCASE("missing header") {
        source.str("1:2.0 4:1.0");
        CHECK_THROWS(io::prediction::read_sparse_prediction(source));
    }
    SUBCASE("invalid rows") {
        source.str("-5 4\n");
        CHECK_THROWS(io::prediction::read_sparse_prediction(source));
    }
    SUBCASE("invalid columns") {
        source.str("2 0\n");
        CHECK_THROWS(io::prediction::read_sparse_prediction(source));
    }
    SUBCASE("too many columns") {
        source.str("1 2\n1:5.0 2:0.5 3:5.2");
        CHECK_THROWS(io::prediction::read_sparse_prediction(source));
    }
    SUBCASE("too few columns") {
        source.str("1 2\n1:5.0");
        CHECK_THROWS(io::prediction::read_sparse_prediction(source));
    }
    SUBCASE("too few rows") {
        source.str("2 2\n1:5.0 2:0.5");
        CHECK_THROWS(io::prediction::read_sparse_prediction(source));
    }
    // too many rows is not really an error that we can diagnose at this point.
    // if we are reading a file, we know it is wrong, but when reading from a
    // stream there might just be other data following. Therefore, this isn't checked
    // here
}



/*!
 * \test Checks in an example that dense predictions writes as expected.
 */
TEST_CASE("save_dense_predictions")
{
    PredictionMatrix values(2, 3);
    values.row(0) << 0.5, 1.5, 0.9;
    values.row(1) << 1.5, 0.9, 0.4;
    std::string as_text =
            "2 3\n"
            "0.5 1.5 0.9\n"
            "1.5 0.9 0.4\n";

    std::stringstream target;
    io::prediction::save_dense_predictions(target, values);
    CHECK(target.str() == as_text);
}

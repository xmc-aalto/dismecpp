// Copyright (c) 2021, Aalto University, developed by Erik Schultheis
// All rights reserved.
//
// SPDX-License-Identifier: MIT

#include "test_utils.h"

dismec::SparseFeatures dismec::make_uniform_sparse_matrix(int rows, int cols, int nonzeros_per_row) {
    SparseFeatures matrix(rows, cols);
    std::vector<Eigen::Triplet<real_t>> content;
    for(int i = 0; i < rows; ++i) {
        // and some uniform features
        for(int j = 0; j < nonzeros_per_row; ++j) {
            content.emplace_back(i, rand() % cols, 1.f);
        }
    }
    matrix.setFromTriplets(begin(content), end(content));
    matrix.makeCompressed();
    return matrix;
}

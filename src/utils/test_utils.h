// Copyright (c) 2021, Aalto University, developed by Erik Schultheis
// All rights reserved.
//
// SPDX-License-Identifier: MIT

#ifndef DISMEC_TEST_UTILS_H
#define DISMEC_TEST_UTILS_H

#include "matrix_types.h"

/*!
 * \brief Creates a sparse matrix with the given number of rows and columns.
 * \param rows Number of rows in the matrix.
 * \param cols Number of columns in the martix.
 * \param non_zeros_per_row Number of nonzero entries in each row.
 * The non-zeros will be distributed uniformly among the columns.
 * \return The resulting sparse matrix.
 */
SparseFeatures make_uniform_sparse_matrix(int rows, int cols, int non_zeros_per_row);


#endif //DISMEC_TEST_UTILS_H

// Copyright (c) 2021, Aalto University, developed by Erik Schultheis
// All rights reserved.
//
// SPDX-License-Identifier: MIT

#ifndef DISMEC_MATRIX_TYPES_H
#define DISMEC_MATRIX_TYPES_H

/*! \file
 * In this file we provide common typedefs for Eigen types to be used throughout the rest of the code.
 */

#include <variant>
#include "utils/type_helpers.h"

namespace dismec {
    using real_t = float;

    namespace types {
        template<class T>
        class GenericMatrixRef;

        template<class Dense, class Sparse>
        class GenericMatrix;

        template<class T>
        using GenericVector = GenericMatrix<DenseVector < T>, SparseVector <T>>;

        template<class T>
        class GenericVectorRef;
    }

    using GenericOutMatrix = types::GenericMatrixRef<real_t>;
    using GenericInMatrix = types::GenericMatrixRef<const real_t>;
    using GenericOutVector = types::GenericVectorRef<real_t>;
    using GenericInVector = types::GenericVectorRef<const real_t>;

    /*!
     * \brief Any dense, real values vector
     */
    using DenseRealVector = types::DenseVector<real_t>;
    using SparseRealVector = types::SparseVector<real_t>;
    using GenericRealVector = types::GenericVector<real_t>;

    /*!
     * \brief Sparse Feature Matrix in Row Major format
     * \details This is the format in which we store the features of a sparse dataset. We use a RowMajor format, because
     * we usually want to iterate over all the features of one example, but not over all the examples of a given
     * feature. Each row corresponds to one instance, and each column to a feature.
    */
    using SparseFeatures = types::SparseRowMajor<real_t>;

    /*!
     * \brief Dense Feature Matrix in Row Major format
     * \details This is the format in which we store the features of a dense dataset. We use a RowMajor format, because
     * we usually want to iterate over all the features of one example, but not over all the examples of a given
     * feature. Each row corresponds to one instance, and each column to a feature.
    */
    using DenseFeatures = types::DenseRowMajor<real_t>;

    using GenericFeatureMatrix = types::GenericMatrix<DenseFeatures, SparseFeatures>;

    /*!
     * \brief Dense vector for storing binary labels.
     * \details We use this type to store a dense representation of a binary label where +1 represents
     * presence of the label and -1 represents its absence.
     * \todo decide whether this should be a matrix or an array type
     */
    using BinaryLabelVector = types::DenseVector<std::int8_t>;

    /*!
     * \brief Dense matrix in Row Major format used for predictions
     * \details This is the matrix type used for dense predictions. To facility predictions on an instance-by-instance
     * basis, this is a RowMajor matrix type. Each row corresponds to one instance, and each column to a label.
     */
    using PredictionMatrix = types::DenseRowMajor<real_t>;

    /*!
     * \brief Matrix used for indices in sparse predictions.
     * \details This matrix is used for predictions, thus it is in row-major order.
     */
    using IndexMatrix = types::DenseRowMajor<long>;
}

#endif //DISMEC_MATRIX_TYPES_H

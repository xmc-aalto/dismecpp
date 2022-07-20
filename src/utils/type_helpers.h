// Copyright (c) 2021, Aalto University, developed by Erik Schultheis
// All rights reserved.
//
// SPDX-License-Identifier: MIT

#ifndef DISMEC_TYPE_HELPERS_H
#define DISMEC_TYPE_HELPERS_H

#include <Eigen/Core>
#include <Eigen/SparseCore>

namespace dismec::types {
    namespace type_helpers {
        template<class T, template<class U> typename V>
        using outer_const = std::conditional_t<std::is_const_v<T>, const V<std::remove_const_t < T>>, V<T>>;

        template<class T>
        using dense_vector_h = Eigen::Matrix<T, Eigen::Dynamic, 1>;

        template<class T>
        using sparse_vector_h = Eigen::SparseVector<T>;

        template<class T>
        using dense_row_major_h = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

        template<class T>
        using dense_col_major_h = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>;

        template<class T>
        using sparse_row_major_h = Eigen::SparseMatrix<T, Eigen::RowMajor>;

        template<class T>
        using sparse_col_major_h = Eigen::SparseMatrix<T, Eigen::ColMajor>;

        namespace definitions {
            template<class T>
            using DenseVector = outer_const<T, dense_vector_h>;

            template<class T>
            using SparseVector = outer_const<T, sparse_vector_h>;

            template<class T>
            using DenseRowMajor = outer_const<T, dense_row_major_h>;

            template<class T>
            using DenseColMajor = outer_const<T, dense_col_major_h>;

            template<class T>
            using SparseRowMajor = outer_const<T, sparse_row_major_h>;

            template<class T>
            using SparseColMajor = outer_const<T, sparse_col_major_h>;
        }
    }

    using namespace type_helpers::definitions;
}

#endif //DISMEC_TYPE_HELPERS_H

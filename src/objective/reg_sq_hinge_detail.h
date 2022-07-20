// Copyright (c) 2021, Aalto University, developed by Erik Schultheis
// All rights reserved.
//
// SPDX-License-Identifier: MIT

#ifndef DISMEC_REG_SQ_HINGE_DETAIL_H
#define DISMEC_REG_SQ_HINGE_DETAIL_H

#include <vector>
#include "matrix_types.h"
#include "utils/fast_sparse_row_iter.h"

namespace dismec {
    namespace l2_reg_sq_hinge_detail {
        template<class Derived>
        inline void htd_sum_naive(const std::vector<int>& indices, Eigen::Ref<DenseRealVector> output,
                                  const Eigen::EigenBase<Derived>& features, const DenseRealVector& costs,
                                  const DenseRealVector& direction) {
            for (auto index: indices) {
                float factor = 2.f * features.derived().row(index).dot(direction) * costs.coeff(index);
                output += factor * features.derived().row(index);
            }
        }

        template<int LOOK_AHEAD = 2>
        inline void htd_sum(const std::vector<int>& indices, Eigen::Ref<DenseRealVector> output,
                            const SparseFeatures& features, const DenseRealVector& costs,
                            const DenseRealVector& direction) {
            if (indices.empty())
                return;

            auto val_ptr = features.valuePtr();
            auto inner_ptr = features.innerIndexPtr();
            auto outer_ptr = features.outerIndexPtr();

            int sm1 = static_cast<int>(indices.size()) - 1;
            for (int i = 0; i < indices.size(); ++i) {
                int index = indices[i];
                int next_index = indices[std::min(i + LOOK_AHEAD, sm1)];
                int next_id = outer_ptr[next_index];
                __builtin_prefetch(&val_ptr[next_id], 0, 1);
                __builtin_prefetch(&inner_ptr[next_id], 0, 1);

                FastSparseRowIter row_iter(features, index);
                float factor = 0.f;
                float cost_val = costs.coeff(index);
                for (FastSparseRowIter it = row_iter; it; ++it) {
                    factor += it.value() * direction.coeff(it.col());
                }
                factor *= 2.f * cost_val;
                for (FastSparseRowIter it = row_iter; it; ++it) {
                    output.coeffRef(it.col()) += it.value() * factor;
                }
            }
        }

        inline void __attribute__((hot, optimize("-ffast-math")))
        htd_sum_new(const std::vector<int>& indices, Eigen::Ref<DenseRealVector> output,
                    const SparseFeatures& features, const DenseRealVector& costs, const DenseRealVector& direction) {
            auto val_ptr = features.valuePtr();
            auto inner_ptr = features.innerIndexPtr();
            auto outer_ptr = features.outerIndexPtr();
            int sm1 = static_cast<int>(indices.size()) - 1;

            for (int i = 0; i < indices.size(); ++i) {
                int index = indices[i];
                int next_index = indices[std::min(i + 2, sm1)];
                int next_id = outer_ptr[next_index];
                __builtin_prefetch(&val_ptr[next_id], 0, 1);
                __builtin_prefetch(&inner_ptr[next_id], 0, 1);

                float factor = 2.f * fast_dot(features, index, direction) * costs.coeff(index);
                for (FastSparseRowIter it(features, index); it; ++it) {
                    output.coeffRef(it.col()) += it.value() * factor;
                }
            }
        }

        /// Given a vector expression xTw that contains the product of feature matrix and
        /// weight vector, a label vector and a vector of cost factors, calculates the squared hinge loss
        template<typename Derived>
        real_t value_from_xTw(const DenseRealVector& cost, const BinaryLabelVector& labels,
                              const Eigen::DenseBase<Derived>& xTw) {
            real_t f = 0;
            assert(xTw.size() == labels.size());
            for (std::size_t i = 0; i < labels.size(); ++i) {
                real_t label = labels.coeff(i);
                real_t d = 1.0 - label * xTw.coeff(i);
                if (d > 0) {
                    real_t factor = cost.coeff(i);
                    f += factor * d * d;
                }
            }

            return f;
        }
    }
}

#endif //DISMEC_REG_SQ_HINGE_DETAIL_H

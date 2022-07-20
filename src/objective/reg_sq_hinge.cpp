// Copyright (c) 2021, Aalto University, developed by Erik Schultheis
// All rights reserved.
//
// SPDX-License-Identifier: MIT

#include "reg_sq_hinge.h"

#include <utility>
#include "utils/hash_vector.h"
#include "utils/fast_sparse_row_iter.h"
#include "reg_sq_hinge_detail.h"
#include "spdlog/spdlog.h"
#include "stats/collection.h"

using namespace dismec::objective;
using namespace dismec::l2_reg_sq_hinge_detail;

namespace {
    using dismec::stats::stat_id_t;
    stat_id_t STAT_GRAD_SPARSITY{8};
}

Regularized_SquaredHingeSVC::Regularized_SquaredHingeSVC(std::shared_ptr<const GenericFeatureMatrix> X,
                                                         std::unique_ptr<Objective> regularizer):
        LinearClassifierImpBase(std::move(X), std::move(regularizer))
{

    if(!features().isCompressed()) {
        throw std::logic_error("feature matrix is not compressed.");
    }

    declare_stat(STAT_GRAD_SPARSITY, {"gradient_sparsity", "% non-zeros"});
}

void Regularized_SquaredHingeSVC::gradient_imp(const HashVector& location, Eigen::Ref<DenseRealVector> target) {
    gradient_and_pre_conditioner_tpl(location, target, nullptr);
}

void Regularized_SquaredHingeSVC::hessian_times_direction_imp(
        const HashVector& location, const DenseRealVector& direction, Eigen::Ref<DenseRealVector> output)
{
    margin_error(location);
    htd_sum(m_MVPos, output, features(), costs(), direction);
}

void Regularized_SquaredHingeSVC::diag_preconditioner_imp(const HashVector& location, Eigen::Ref<DenseRealVector> target)
{
    gradient_and_pre_conditioner_tpl(location, nullptr, target);
}

void Regularized_SquaredHingeSVC::gradient_and_pre_conditioner_imp(
        const HashVector& location,
        Eigen::Ref<DenseRealVector> gradient,
        Eigen::Ref<DenseRealVector> pre)
{
    gradient_and_pre_conditioner_tpl(location, gradient, pre);
}

template<class T, class U>
void Regularized_SquaredHingeSVC::gradient_and_pre_conditioner_tpl(const HashVector& location, T&& gradient, U&& pre) {
    // first, we determine whether we want to calculate gradient and/or preconditioning
    constexpr bool calc_grad = !std::is_same_v<T, std::nullptr_t>;
    constexpr bool calc_pre = !std::is_same_v<U, std::nullptr_t>;

    const auto& cost_vec = costs();
    const auto& label_vec = labels();

    margin_error(location);
    record(STAT_GRAD_SPARSITY, static_cast<real_t>(static_cast<double>(100*m_MVPos.size()) / label_vec.size()));

    const auto& ft = features();

    for (int i = 0; i < m_MVPos.size(); ++i)
    {
        int pos = m_MVPos[i];
        real_t cost = 2.0 * cost_vec[pos];
        real_t vi = - cost * label_vec.coeff(pos) * m_MVVal[i];
        for (FastSparseRowIter it(ft, pos); it; ++it)
        {
            if constexpr (calc_grad) {
                gradient.coeffRef(it.col()) += it.value() * vi;
            }
            if constexpr (calc_pre) {
                pre.coeffRef(it.col()) += it.value() * it.value() * cost;
            }
        }
    }
}

#include <iostream>
void Regularized_SquaredHingeSVC::gradient_at_zero_imp(Eigen::Ref<DenseRealVector> target) {
    const auto& cost_vec = costs();
    const auto& label_vec = labels();

    for (int i = 0; i < cost_vec.size(); ++i)
    {
        real_t cost = real_t{2} * cost_vec[i];
        // margin_error = 1
        real_t vi = -cost * label_vec.coeff(i)  ;
        for (FastSparseRowIter it(features(), i); it; ++it)
        {
            target.coeffRef(it.col()) += it.value() * vi;
        }
    }
}

const Regularized_SquaredHingeSVC::features_t& Regularized_SquaredHingeSVC::features() const {
    return sparse_features();
}

void Regularized_SquaredHingeSVC::invalidate_labels() {
    // modifying the true labels invalidates margin caches
    m_Last_MV = {};
}

void Regularized_SquaredHingeSVC::margin_error(const HashVector& w) {
    if(w.hash() == m_Last_MV) {
        return;
    }

    m_MVPos.clear();
    m_MVVal.clear();
    m_Last_MV = w.hash();
    const auto& lbl = labels();
    const auto& xTw = x_times_w(w);
    for(Eigen::Index i = 0; i < lbl.size(); ++i) {
        real_t label = lbl.coeff(i);
        real_t d = real_t{1.0} - label * xTw.coeff(i);
        if (d > 0) {
            m_MVPos.push_back(i);
            m_MVVal.push_back(d);
        }
    }
}

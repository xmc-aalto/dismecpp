// Copyright (c) 2021, Aalto University, developed by Erik Schultheis
// All rights reserved.
//
// SPDX-License-Identifier: MIT

#include "linear.h"
#include "utils/eigen_generic.h"
#include "utils/throw_error.h"
#include "stats/timer.h"

using namespace dismec;
using namespace dismec::objective;

namespace {
    constexpr const dismec::stats::stat_id_t STAT_PERF_MATMUL{7};
}

LinearClassifierBase::LinearClassifierBase(std::shared_ptr<const GenericFeatureMatrix> X) :
    m_FeatureMatrix( std::move(X) ),
    m_X_times_w( m_FeatureMatrix->rows() ),
    m_LsCache_xTd( m_FeatureMatrix->rows() ),
    m_LsCache_xTw( m_FeatureMatrix->rows() ),
    m_Costs( m_FeatureMatrix->rows() ),
    m_Y( m_FeatureMatrix->rows() )
{
    m_Costs.fill(1);
    declare_stat(STAT_PERF_MATMUL, {"perf_matmul", "µs"});
}


long LinearClassifierBase::num_instances() const noexcept {
    return m_FeatureMatrix->rows();
}

long LinearClassifierBase::num_variables() const noexcept {
    return m_FeatureMatrix->cols();
}

const DenseFeatures& LinearClassifierBase::dense_features() const {
    return m_FeatureMatrix->dense();
}

const SparseFeatures& LinearClassifierBase::sparse_features() const {
    return m_FeatureMatrix->sparse();
}

const GenericFeatureMatrix& LinearClassifierBase::generic_features() const {
    return *m_FeatureMatrix;
}

const DenseRealVector& LinearClassifierBase::x_times_w(const HashVector& w) {
    if(w.hash() == m_Last_W) {
        return m_X_times_w;
    }
    auto timer = make_timer(STAT_PERF_MATMUL);
    visit([&](auto&& features) {
            m_X_times_w.noalias() = features * w;
        }, *m_FeatureMatrix);
    m_Last_W = w.hash();
    return m_X_times_w;
}

void LinearClassifierBase::project_linear_to_line(const HashVector& location, const DenseRealVector& direction) {
    visit([&](auto&& features) {
        m_LsCache_xTd.noalias() = features * direction;
    }, *m_FeatureMatrix);
    m_LsCache_xTw = x_times_w(location);
}

BinaryLabelVector& LinearClassifierBase::get_label_ref() {
    invalidate_labels();
    return m_Y;
}

void LinearClassifierBase::update_costs(real_t positive, real_t negative) {
    for(int i = 0; i < m_Costs.size(); ++i) {
        if(m_Y.coeff(i) == 1) {
            m_Costs.coeffRef(i) = positive;
        } else {
            m_Costs.coeffRef(i) = negative;
        }
    }
}

const DenseRealVector& LinearClassifierBase::costs() const {
    return m_Costs;
}

const BinaryLabelVector& LinearClassifierBase::labels() const {
    return m_Y;
}

// Copyright (c) 2021, Aalto University, developed by Erik Schultheis
// All rights reserved.
//
// SPDX-License-Identifier: MIT

#include "dense_and_sparse.h"
#include "utils/throw_error.h"
#include "stats/timer.h"
#include "margin_losses.h"
#include "utils/eigen_generic.h"

using namespace dismec;
using namespace dismec::objective;

namespace {
    using dismec::stats::stat_id_t;
    constexpr const stat_id_t STAT_PERF_MATMUL{7};
}

DenseAndSparseLinearBase::DenseAndSparseLinearBase(std::shared_ptr<const GenericFeatureMatrix> dense_features,
                                                   std::shared_ptr<const GenericFeatureMatrix> sparse_features) :
    m_DenseFeatures( std::move(dense_features) ),
    m_SparseFeatures( std::move(sparse_features) ),
    m_X_times_w( m_DenseFeatures->rows() ),
    m_LsCache_xTd( m_DenseFeatures->rows() ),
    m_LsCache_xTw( m_DenseFeatures->rows() ),
    m_Costs( m_DenseFeatures->rows() ),
    m_Y( m_DenseFeatures->rows() ),
    m_DerivativeBuffer(m_DenseFeatures->rows()), m_SecondDerivativeBuffer(m_DenseFeatures->rows()),
    m_LineStart( get_num_variables() ), m_LineDirection( get_num_variables() ),
    m_LineCache( get_num_variables() ), m_GenericInBuffer(m_DenseFeatures->rows()), m_GenericOutBuffer(m_DenseFeatures->rows())
{
    ALWAYS_ASSERT_EQUAL(m_DenseFeatures->rows(), m_SparseFeatures->rows(), "Mismatching number ({} vs {}) of instances (rows) in dense and sparse part.")
    m_Costs.fill(1);
    declare_stat(STAT_PERF_MATMUL, {"perf_matmul", "µs"});
}


long DenseAndSparseLinearBase::num_instances() const noexcept {
    return m_DenseFeatures->rows();
}

long DenseAndSparseLinearBase::num_variables() const noexcept {
    return get_num_variables();
}

long DenseAndSparseLinearBase::get_num_variables() const noexcept {
    return m_DenseFeatures->cols() + m_SparseFeatures->cols();
}

const DenseFeatures& DenseAndSparseLinearBase::dense_features() const {
    return m_DenseFeatures->dense();
}

const SparseFeatures& DenseAndSparseLinearBase::sparse_features() const {
    return m_SparseFeatures->sparse();
}

#define DENSE_PART(source) source.head(dense_features().cols())
#define SPARSE_PART(source) source.tail(sparse_features().cols())

const DenseRealVector& DenseAndSparseLinearBase::x_times_w(const HashVector& w) {
    if(w.hash() == m_Last_W) {
        return m_X_times_w;
    }
    auto timer = make_timer(STAT_PERF_MATMUL);
    m_X_times_w.noalias() = dense_features() * DENSE_PART(w.get());
    m_X_times_w.noalias() += sparse_features() * SPARSE_PART(w.get());
    m_Last_W = w.hash();
    return m_X_times_w;
}

void DenseAndSparseLinearBase::project_linear_to_line(const HashVector& location, const DenseRealVector& direction) {
    m_LsCache_xTd.noalias() = dense_features() * DENSE_PART(direction);
    m_LsCache_xTd.noalias() += sparse_features() * SPARSE_PART(direction);
    m_LsCache_xTw = x_times_w(location);
    m_LineDirection = direction;
    m_LineStart = location.get();
}

BinaryLabelVector& DenseAndSparseLinearBase::get_label_ref() {
    invalidate_labels();
    return m_Y;
}


void DenseAndSparseLinearBase::update_costs(real_t positive, real_t negative) {
    m_Costs.resize(labels().size());
    for(int i = 0; i < m_Costs.size(); ++i) {
        if(m_Y.coeff(i) == 1) {
            m_Costs.coeffRef(i) = positive;
        } else {
            m_Costs.coeffRef(i) = negative;
        }
    }
}

const DenseRealVector& DenseAndSparseLinearBase::costs() const {
    return m_Costs;
}

const BinaryLabelVector& DenseAndSparseLinearBase::labels() const {
    return m_Y;
}

real_t DenseAndSparseLinearBase::value_unchecked(const HashVector& location) {
    const DenseRealVector& xTw = x_times_w(location);
    return value_from_xTw(xTw) + regularization_value(location.get());
}

real_t DenseAndSparseLinearBase::lookup_on_line(real_t position) {
    m_GenericInBuffer = line_interpolation(position);
    real_t f = value_from_xTw(m_GenericInBuffer);
    m_LineCache = m_LineStart + position * m_LineDirection;
    return f + regularization_value(m_LineCache);
}

real_t DenseAndSparseLinearBase::value_from_xTw(const DenseRealVector& xTw)
{
    m_GenericOutBuffer.resize(labels().size());
    calculate_loss(xTw, labels(), m_GenericOutBuffer);
    return m_GenericOutBuffer.dot(costs());
}

void
DenseAndSparseLinearBase::hessian_times_direction_unchecked(const HashVector& location, const DenseRealVector& direction,
                                                           Eigen::Ref<DenseRealVector> target) {
    regularization_hessian(location.get(), direction, target);

    const auto& hessian = cached_2nd_derivative(location);
    for (int pos = 0; pos < hessian.size(); ++pos) {
        if(real_t h = hessian.coeff(pos); h != 0) {
            real_t factor = dense_features().row(pos).dot(DENSE_PART(direction)) +
                            sparse_features().row(pos).dot(SPARSE_PART(direction));
            DENSE_PART(target) += dense_features().row(pos) * factor * h;
            SPARSE_PART(target) += sparse_features().row(pos) * factor * h;
        }
    }
}

void DenseAndSparseLinearBase::gradient_and_pre_conditioner_unchecked(const HashVector& location,
                                                                     Eigen::Ref<DenseRealVector> gradient,
                                                                     Eigen::Ref<DenseRealVector> pre) {
    regularization_gradient(location.get(), gradient);
    regularization_preconditioner(location.get(), pre);

    const auto& derivative = cached_derivative(location);
    const auto& hessian = cached_2nd_derivative(location);
    for (int pos = 0; pos < derivative.size(); ++pos) {
        if(real_t d = derivative.coeff(pos); d != 0) {
            DENSE_PART(gradient) += dense_features().row(pos) * d;
            SPARSE_PART(gradient) += sparse_features().row(pos) * d;
        }
        if(real_t h = hessian.coeff(pos); h != 0) {
            DENSE_PART(pre) += dense_features().row(pos).cwiseAbs2() * h;
            SPARSE_PART(pre) += sparse_features().row(pos).cwiseAbs2() * h;
        }
    }

}

void DenseAndSparseLinearBase::gradient_unchecked(const HashVector& location, Eigen::Ref<DenseRealVector> target) {
    regularization_gradient(location.get(), target);

    const auto& derivative = cached_derivative(location);
    for (int pos = 0; pos < derivative.size(); ++pos) {
        if(real_t d = derivative.coeff(pos); d != 0) {
            DENSE_PART(target) += dense_features().row(pos) * d;
            SPARSE_PART(target) += sparse_features().row(pos) * d;
        }
    }
}

void DenseAndSparseLinearBase::gradient_at_zero_unchecked(Eigen::Ref<DenseRealVector> target) {
    regularization_gradient_at_zero(target);

    m_GenericInBuffer = DenseRealVector::Zero(labels().size());
    m_GenericOutBuffer.resize(m_GenericInBuffer.size());
    calculate_derivative(m_GenericInBuffer, labels(), m_GenericOutBuffer);
    const auto& cost_vector = costs();
    for (int pos = 0; pos < m_GenericOutBuffer.size(); ++pos) {
        if(real_t d = m_GenericOutBuffer.coeff(pos); d != 0) {
            DENSE_PART(target) += dense_features().row(pos) * (cost_vector.coeff(pos) * d);
            SPARSE_PART(target) += sparse_features().row(pos) * (cost_vector.coeff(pos) * d);
        }
    }
}

void DenseAndSparseLinearBase::diag_preconditioner_unchecked(const HashVector& location, Eigen::Ref<DenseRealVector> target) {
    regularization_preconditioner(location.get(), target);

    const auto& hessian = cached_2nd_derivative(location);
    for (int pos = 0; pos < hessian.size(); ++pos) {
        if(real_t h = hessian.coeff(pos); h != 0) {
            DENSE_PART(target) += dense_features().row(pos).cwiseAbs2() * h;
            SPARSE_PART(target) += sparse_features().row(pos).cwiseAbs2() * h;
        }
    }
}

const DenseRealVector& DenseAndSparseLinearBase::cached_derivative(const HashVector& location) {
    return m_DerivativeBuffer.update(location, [&](const DenseRealVector& input, DenseRealVector& out){
        out.resize(labels().size());
        calculate_derivative(x_times_w(location), labels(), out);
        out.array() *= costs().array();
    });
}

const DenseRealVector& DenseAndSparseLinearBase::cached_2nd_derivative(const HashVector& location) {
    return m_SecondDerivativeBuffer.update(location, [&](const DenseRealVector& input, DenseRealVector& out){
        out.resize(labels().size());
        calculate_2nd_derivative(x_times_w(location), labels(), out);
        out.array() *= costs().array();
    });
}

void DenseAndSparseLinearBase::invalidate_labels() {
    m_DerivativeBuffer.invalidate();
    m_SecondDerivativeBuffer.invalidate();
}

void DenseAndSparseLinearBase::project_to_line_unchecked(const HashVector& location, const DenseRealVector& direction) {
    project_linear_to_line(location, direction);
}

void DenseAndSparseLinearBase::update_features(const DenseFeatures& dense, const SparseFeatures& sparse) {
    m_DenseFeatures = std::make_shared<const GenericFeatureMatrix>(dense);
    m_SparseFeatures = std::make_shared<const GenericFeatureMatrix>(sparse);
}

namespace {
    struct L2Regularizer {
        [[nodiscard]] real_t value(real_t weight) const {
            return weight * weight;
        }

        [[nodiscard]] real_t grad(real_t weight) const {
            return real_t{2} * weight;
        }

        [[nodiscard]] real_t quad(real_t weight) const {
            return real_t{2};
        }
    };
}

std::unique_ptr<DenseAndSparseLinearBase> dismec::objective::make_sp_dense_squared_hinge(
    std::shared_ptr<const GenericFeatureMatrix> dense_features,
    real_t dense_reg_strength,
    std::shared_ptr<const GenericFeatureMatrix> sparse_features,
    real_t sparse_reg_strength) {
    return std::make_unique<objective::DenseAndSparseMargin<SquaredHingePhi, L2Regularizer, L2Regularizer>>(
        std::move(dense_features),
        std::move(sparse_features),
        SquaredHingePhi{}, L2Regularizer{}, dense_reg_strength, L2Regularizer{}, sparse_reg_strength);
}

#include "doctest.h"

using namespace dismec;

namespace {
    struct ZeroPhi {
        [[nodiscard]] real_t value(real_t margin) const {
            return 0;
        }

        [[nodiscard]] real_t grad(real_t margin) const {
            return 0;
        }

        [[nodiscard]] real_t quad(real_t margin) const {
            return 0;
        }
    };
}

TEST_CASE("pure-regularization") {
    DenseFeatures zero_dense = DenseFeatures::Zero(5, 3);
    SparseFeatures zero_sparse = SparseFeatures(5, 4);
    DenseAndSparseMargin<ZeroPhi, L2Regularizer, L2Regularizer> goal(
        std::make_shared<const GenericFeatureMatrix>(zero_dense),
        std::make_shared<const GenericFeatureMatrix>(zero_sparse),
        ZeroPhi{}, L2Regularizer{}, 1.0, L2Regularizer{}, 1.0);

    DenseRealVector weights{{1.0, 2.0, -1.0, 0.0, 1.0, 2.0, 5.0}};
    HashVector hv(weights);

    CHECK(goal.value(hv) == 1.0 + 4.0 + 1.0 + 1.0 + 4.0 + 25.0);

    DenseRealVector out_grad(7);
    DenseRealVector expected_grad = 2.0 * weights;
    goal.gradient(hv, out_grad);
    CHECK(expected_grad == out_grad);
}
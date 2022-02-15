// Copyright (c) 2021, Aalto University, developed by Erik Schultheis
// All rights reserved.
//
// SPDX-License-Identifier: MIT

#include "generic_linear.h"
#include "utils/eigen_generic.h"
#include "utils/throw_error.h"
#include "doctest.h"
#include "stats/collection.h"

using objective::GenericLinearClassifier;

namespace {
    stats::stat_id_t STAT_GRAD_SPARSITY{8};
}

real_t GenericLinearClassifier::value_unchecked(const HashVector& location) {
    const DenseRealVector& xTw = x_times_w(location);
    return value_from_xTw(xTw) + m_Regularizer->value(location);
}

real_t GenericLinearClassifier::lookup_on_line(real_t position) {
    m_GenericInBuffer = line_interpolation(position);
    real_t f = value_from_xTw(m_GenericInBuffer);
    return f + m_Regularizer->lookup_on_line(position);
}

real_t GenericLinearClassifier::value_from_xTw(const DenseRealVector& xTw)
{
    calculate_loss(xTw, labels(), m_GenericOutBuffer);
    return m_GenericOutBuffer.dot(costs());
}

void
GenericLinearClassifier::hessian_times_direction_unchecked(const HashVector& location, const DenseRealVector& direction,
                                                           Eigen::Ref<DenseRealVector> target) {
    m_Regularizer->hessian_times_direction(location, direction, target);

    const auto& hessian = cached_2nd_derivative(location);
    visit([&](const auto& features) {
        for (int pos = 0; pos < hessian.size(); ++pos) {
            if(real_t h = hessian.coeff(pos); h != 0) {
                real_t factor = features.row(pos).dot(direction);
                target += features.row(pos) * factor * h;
            }
        }
    }, generic_features());
}

void GenericLinearClassifier::gradient_and_pre_conditioner_unchecked(const HashVector& location,
                                                                     Eigen::Ref<DenseRealVector> gradient,
                                                                     Eigen::Ref<DenseRealVector> pre) {
    m_Regularizer->gradient(location, gradient);
    m_Regularizer->diag_preconditioner(location, pre);

    const auto& derivative = cached_derivative(location);
    const auto& hessian = cached_2nd_derivative(location);
    visit([&](const auto& features) {
        for (int pos = 0; pos < derivative.size(); ++pos) {
            if(real_t d = derivative.coeff(pos); d != 0) {
                gradient += features.row(pos) * d;
            }
            if(real_t h = hessian.coeff(pos); h != 0) {
                pre += features.row(pos).cwiseAbs2() * h;
            }
        }
    }, generic_features());

}

void GenericLinearClassifier::gradient_unchecked(const HashVector& location, Eigen::Ref<DenseRealVector> target) {
    m_Regularizer->gradient(location, target);

    const auto& derivative = cached_derivative(location);
    visit([&](const auto& features) {
        for (int pos = 0; pos < derivative.size(); ++pos) {
            if(real_t d = derivative.coeff(pos); d != 0) {
                target += features.row(pos) * d;
            }
        }
    }, generic_features());
}

void GenericLinearClassifier::gradient_at_zero_unchecked(Eigen::Ref<DenseRealVector> target) {
    m_Regularizer->gradient_at_zero(target);

    m_GenericInBuffer = DenseRealVector::Zero(labels().size());
    calculate_derivative(m_GenericInBuffer, labels(), m_GenericOutBuffer);
    auto& cost_vector = costs();
    visit([&](const auto& features) {
        for (int pos = 0; pos < m_GenericOutBuffer.size(); ++pos) {
            if(real_t d = m_GenericOutBuffer.coeff(pos); d != 0) {
                target += features.row(pos) * (cost_vector.coeff(pos) * d);
            }
        }
    }, generic_features());
}

void GenericLinearClassifier::diag_preconditioner_unchecked(const HashVector& location, Eigen::Ref<DenseRealVector> target) {
    m_Regularizer->diag_preconditioner(location, target);

    const auto& hessian = cached_2nd_derivative(location);
    visit([&](const auto& features) {
        for (int pos = 0; pos < hessian.size(); ++pos) {
            if(real_t h = hessian.coeff(pos); h != 0) {
                target += features.row(pos).cwiseAbs2() * h;
            }
        }
    }, generic_features());
}

const DenseRealVector& GenericLinearClassifier::cached_derivative(const HashVector& location) {
    return m_DerivativeBuffer.update(location, [&](const DenseRealVector& input, DenseRealVector& out){
        calculate_derivative(x_times_w(location), labels(), out);
        record(STAT_GRAD_SPARSITY, [&](){
            long nnz = 0;
            for(int i = 0; i < out.size(); ++i) {
                if(out.coeff(i) != 0) ++nnz;
            }
            return static_cast<real_t>(static_cast<double>(100*nnz) / out.size()); });
        out.array() *= costs().array();
    });
}

const DenseRealVector& GenericLinearClassifier::cached_2nd_derivative(const HashVector& location) {
    return m_SecondDerivativeBuffer.update(location, [&](const DenseRealVector& input, DenseRealVector& out){
        calculate_2nd_derivative(x_times_w(location), labels(), out);
        out.array() *= costs().array();
    });
}

void objective::GenericLinearClassifier::invalidate_labels() {
    m_DerivativeBuffer.invalidate();
    m_SecondDerivativeBuffer.invalidate();
}

GenericLinearClassifier::GenericLinearClassifier(std::shared_ptr<const GenericFeatureMatrix> X,
                                                            std::unique_ptr<Objective> regularizer)
        : LinearClassifierBase(std::move(X)),
        m_Regularizer(std::move(regularizer)),
        m_DerivativeBuffer(num_instances()), m_SecondDerivativeBuffer(num_instances()),
        m_GenericOutBuffer(num_instances()), m_GenericInBuffer(num_instances())
        {
    declare_stat(STAT_GRAD_SPARSITY, {"gradient_sparsity", "% non-zeros"});
    if(!m_Regularizer) {
        THROW_EXCEPTION(std::invalid_argument, "Regularizer cannot be nullptr");
    }
}

void GenericLinearClassifier::project_to_line_unchecked(const HashVector& location, const DenseRealVector& direction) {
    project_linear_to_line(location, direction);
    m_Regularizer->project_to_line(location, direction);
}


// ---------------------------------------------------------------------------------------------------------------------
//                  Some concrete implementations of common loss functions
// ---------------------------------------------------------------------------------------------------------------------

namespace {
    struct SquaredHingePhi {
        [[nodiscard]] real_t value(real_t margin) const {
            real_t value = std::max(real_t{0}, real_t{1.0} - margin);
            return value * value;
        }

        [[nodiscard]] real_t grad(real_t margin) const {
            real_t value = std::max(real_t{0}, real_t{1.0} - margin);
            return -real_t{2} * value;
        }

        [[nodiscard]] real_t quad(real_t margin) const {
            real_t value = real_t{1.0} - margin;
            return value > 0 ? real_t{2} : real_t{0};
        }
    };

    struct HuberPhi {
        [[nodiscard]] real_t value(real_t margin) const {
            real_t value = std::max(real_t{0}, real_t{1.0} - margin);
            if(value > Epsilon) return value - Epsilon/2;
            return real_t{0.5} * value*value / Epsilon;
        }

        [[nodiscard]] real_t grad(real_t margin) const {
            real_t value = std::max(real_t{0}, real_t{1} - margin);
            if(value > Epsilon) {
                return -real_t{1};
            } else if(value == real_t{0}) {
                return real_t{0};
            } else {
                return -value / Epsilon;
            }
        }

        [[nodiscard]] real_t quad(real_t margin) const {
            real_t value = std::max(real_t{0}, real_t{1.0} - margin);
            if(value > Epsilon) return real_t{1.0} / value;
            if(value == 0) return real_t{0};
            return real_t{1} / Epsilon;
        }

        real_t Epsilon = 1;
    };

    struct LogisticPhi {
        [[nodiscard]] real_t value(real_t margin) const {
            real_t exp_part = std::exp(-margin);
            if(std::isfinite(exp_part)) {
                return std::log1p(exp_part);
            } else {
                return -margin;
            }
        }

        [[nodiscard]] real_t grad(real_t margin) const {
            real_t exp_part = std::exp(margin);
            if(std::isfinite(exp_part)) {
                return -real_t{1} / (real_t{1} + exp_part);
            } else {
                return 0;
            }
        }

        [[nodiscard]] real_t quad(real_t margin) const {
            real_t exp_part = std::exp(margin);
            if(std::isfinite(exp_part)) {
                return exp_part / std::pow(1 + exp_part, real_t{2});
            } else {
                return 0;
            }
        }
    };

    template<class Phi, class... Args>
    std::unique_ptr<GenericLinearClassifier> make_gen_lin_classifier(std::shared_ptr<const GenericFeatureMatrix> X,
                                                                     std::unique_ptr<objective::Objective> regularizer,
                                                                     Args... args) {
        return std::make_unique<objective::GenericMarginClassifier<Phi>>(std::move(X), std::move(regularizer),
                Phi{std::forward<Args>(args)...});
    }
}

std::unique_ptr<GenericLinearClassifier> objective::make_squared_hinge(std::shared_ptr<const GenericFeatureMatrix> X,
                                                                       std::unique_ptr<Objective> regularizer) {
    return make_gen_lin_classifier<SquaredHingePhi>(std::move(X), std::move(regularizer));
}

std::unique_ptr<GenericLinearClassifier> objective::make_logistic_loss(std::shared_ptr<const GenericFeatureMatrix> X,
                                                                       std::unique_ptr<Objective> regularizer) {
    return make_gen_lin_classifier<LogisticPhi>(std::move(X), std::move(regularizer));
}

std::unique_ptr<GenericLinearClassifier> objective::make_huber_hinge(std::shared_ptr<const GenericFeatureMatrix> X,
                                                                     std::unique_ptr<Objective> regularizer,
                                                                     real_t epsilon) {
    return make_gen_lin_classifier<HuberPhi>(std::move(X), std::move(regularizer), epsilon);
}

#include "doctest.h"
#include "regularizers_imp.h"
#include "reg_sq_hinge.h"

namespace {
    void test_equivalence(objective::Objective& a, objective::Objective& b, const HashVector& input) {
        auto test_vector_equal = [](auto&& u, auto&& v, const char* message){
            REQUIRE(u.size() == v.size());
            for(int i = 0; i < u.size(); ++i) {
                REQUIRE_MESSAGE(u.coeff(i) == doctest::Approx(v.coeff(i)), message);
            }
        };
        DenseRealVector buffer_a(input->size());
        DenseRealVector buffer_b(input->size());
        CHECK_MESSAGE(a.value(input) == doctest::Approx(b.value(input)), "values differ");

        a.gradient_at_zero(buffer_a);
        b.gradient_at_zero(buffer_b);
        test_vector_equal(buffer_a, buffer_b, "gradient@0 mismatch");

        a.gradient(input, buffer_a);
        b.gradient(input, buffer_b);
        test_vector_equal(buffer_a, buffer_b, "gradient mismatch");

        a.diag_preconditioner(input, buffer_a);
        b.diag_preconditioner(input, buffer_b);
        test_vector_equal(buffer_a, buffer_b, "pre-conditioner mismatch");

        DenseRealVector direction = DenseRealVector::Random(input->size());
        a.hessian_times_direction(input, direction, buffer_a);
        b.hessian_times_direction(input, direction, buffer_b);
        test_vector_equal(buffer_a, buffer_b, "hessian mismatch");

        DenseRealVector buffer_a2(input->size());
        DenseRealVector buffer_b2(input->size());
        a.gradient_and_pre_conditioner(input, buffer_a, buffer_a2);
        b.gradient_and_pre_conditioner(input, buffer_b, buffer_b2);
        test_vector_equal(buffer_a, buffer_b, "gradient mismatch");
        test_vector_equal(buffer_a2, buffer_b2, "pre-conditioner mismatch");
    }
}

TEST_CASE("sparse/dense equivalence") {
    int rows, cols;
    real_t pos_cost = 1, neg_cost = 1;

    auto run_test = [&](){
        DenseFeatures features_dense = DenseFeatures::Random(rows, cols);
        SparseFeatures features_sparse = features_dense.sparseView();

        Eigen::Matrix<std::int8_t, Eigen::Dynamic, 1> labels = Eigen::Matrix<std::int8_t, Eigen::Dynamic, 1>::Random(rows);
        for(int i = 0; i < labels.size(); ++i) {
            if(labels.coeff(i) > 0) {
                labels.coeffRef(i) = 1;
            } else {
                labels.coeffRef(i) = -1;
            }
        }


        auto reg_dense = make_squared_hinge(std::make_shared<GenericFeatureMatrix>(features_dense),
                                            std::make_unique<objective::SquaredNormRegularizer>());
        auto reg_sparse = make_squared_hinge(std::make_shared<GenericFeatureMatrix>(features_sparse),
                                             std::make_unique<objective::SquaredNormRegularizer>());

        auto reference = objective::Regularized_SquaredHingeSVC(std::make_shared<GenericFeatureMatrix>(features_sparse),
                                                                std::make_unique<objective::SquaredNormRegularizer>());

        auto do_test = [&](auto& first, auto& second) { ;
            first.get_label_ref() = labels;
            second.get_label_ref() = labels;

            first.update_costs(pos_cost, neg_cost);
            second.update_costs(pos_cost, neg_cost);

            DenseRealVector weights = DenseRealVector::Random(cols);
            test_equivalence(first, second, HashVector(weights));
        };

        do_test(*reg_dense, *reg_sparse);
        do_test(reference, *reg_sparse);
    };

    SUBCASE("rows > cols") {
        rows = 20;
        cols = 10;
        run_test();
    }
    SUBCASE("cols > rows") {
        rows = 10;
        cols = 20;
        run_test();
    }

    SUBCASE("pos weighted") {
        rows = 15;
        cols = 32;
        pos_cost = 2.0;
        run_test();
    }

    SUBCASE("neg weighted") {
        rows = 15;
        cols = 32;
        neg_cost = 2.0;
        run_test();
    }
    /*
    SUBCASE("large") {
        rows = 1500;
        cols = 3200;
        // this fails, which indicates different numerical stability
        run_test();
    }
    */
}


TEST_CASE("generic squared hinge") {
    SparseFeatures x(3, 5);
    x.insert(0, 3) = 1.0;
    x.insert(1, 0) = 2.0;
    x.insert(2, 1) = 1.0;
    x.insert(2, 2) = 1.0;

    Eigen::Matrix<std::int8_t, Eigen::Dynamic, 1> y(3);
    y << -1, 1, -1;


    auto loss = make_squared_hinge(std::make_shared<GenericFeatureMatrix>(DenseFeatures (x)),
            std::make_unique<objective::SquaredNormRegularizer>());
    loss->get_label_ref() = y;

    auto reference = objective::Regularized_SquaredHingeSVC(std::make_shared<GenericFeatureMatrix>(x),
                                                            std::make_unique<objective::SquaredNormRegularizer>());
    reference.get_label_ref() = y;

    DenseRealVector weights(5);
    weights << 1.0, 2.0, 0.0, -1.0, 2.0;

    auto do_check = [&](real_t factor){
        // z = (-1, 2, 2)
        // 1 - yz = 0, -1, 3
        CHECK_MESSAGE(loss->value(HashVector{weights}) == doctest::Approx(factor * 9.0 + 5), "wrong value");

        //
        DenseRealVector grad(5);
        loss->gradient(HashVector{weights}, grad);
        // dl/dz = 0, 0, 2*3
        // dl/dx = 6*(0.0, 1.0, 1.0, 0.0, 0.0) + 0.5*weights
        DenseRealVector i(5);
        i << 0.0, 1.0, 1.0, 0.0, 0.0;
        DenseRealVector r = factor * 6 * i + weights;
        CHECK_MESSAGE(grad == r, "wrong gradient");

        // also check numerically
        real_t old_val = loss->value(HashVector{weights});
        DenseRealVector nw = weights + grad * 1e-4;
        real_t new_val = loss->value(HashVector{nw});
        CHECK (new_val - old_val == doctest::Approx(grad.squaredNorm() * 1e-4).epsilon(1e-4));

        // preconditioner == diagonal of Hessian
        DenseRealVector prec_new(5);
        DenseRealVector prec_old(5);
        loss->diag_preconditioner(HashVector{weights}, prec_new);
        reference.diag_preconditioner(HashVector{weights}, prec_old);
        CHECK_MESSAGE(prec_new == prec_old, "wrong preconditioner");

        loss->hessian_times_direction(HashVector{weights}, i, prec_new);
        reference.hessian_times_direction(HashVector{weights}, i, prec_old);
        CHECK_MESSAGE(prec_new == prec_old, "wrong hessian");

        // g@0
        loss->gradient_at_zero(prec_new);
        reference.gradient_at_zero(prec_old);
        CHECK_MESSAGE(prec_new == prec_old, "g@0 wrong");
    };


    // since the positive example is correct with margin,
    // re-weighting positives does not change the outcome,
    // whereas negatives change the result by a constant factor
    SUBCASE("unweighted") {
        do_check(1.0);
    }
    SUBCASE("positive-reweighted") {
        loss->update_costs(2.0, 1.0);
        reference.update_costs(2.0, 1.0);
        do_check(1.0);
    }
    SUBCASE("negative-reweighted") {
        loss->update_costs(1.0, 2.0);
        reference.update_costs(1.0, 2.0);
        do_check(2.0);
    }
}
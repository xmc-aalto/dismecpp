// Copyright (c) 2021, Aalto University, developed by Erik Schultheis
// All rights reserved.
//
// SPDX-License-Identifier: MIT

#ifndef DISMEC_GENERIC_LINEAR_H
#define DISMEC_GENERIC_LINEAR_H

#include "linear.h"

namespace objective {
    /*!
     * \brief This is a non-templated, runtime-polymorphic generic implementation of the linear classifier objective.
     */
    class GenericLinearClassifier : public LinearClassifierBase {
    public:
        GenericLinearClassifier(std::shared_ptr<const GenericFeatureMatrix> X, std::unique_ptr<Objective> regularizer);
    private:
        real_t value_unchecked(const HashVector& location) override;

        real_t lookup_on_line(real_t position) override;

        void gradient_unchecked(const HashVector& location, Eigen::Ref<DenseRealVector> target) override;

        void gradient_at_zero_unchecked(Eigen::Ref<DenseRealVector> target) override;

        void hessian_times_direction_unchecked(const HashVector& location,
                                               const DenseRealVector& direction,
                                               Eigen::Ref<DenseRealVector> target) override;

        void diag_preconditioner_unchecked(const HashVector& location, Eigen::Ref<DenseRealVector> target) override;

        void gradient_and_pre_conditioner_unchecked(const HashVector& location, Eigen::Ref<DenseRealVector> gradient,
                                                    Eigen::Ref<DenseRealVector> pre) override;

        void project_to_line_unchecked(const HashVector& location, const DenseRealVector& direction) override;

        real_t value_from_xTw(const DenseRealVector& xTw);

        virtual void calculate_loss(const DenseRealVector& scores,
                                    const BinaryLabelVector& labels,
                                    DenseRealVector& out) const = 0;

        virtual void calculate_derivative(const DenseRealVector& scores,
                                          const BinaryLabelVector& labels,
                                          DenseRealVector& out) const = 0;

        virtual void calculate_2nd_derivative(const DenseRealVector& scores,
                                              const BinaryLabelVector& labels,
                                              DenseRealVector& out) const = 0;

        const DenseRealVector& cached_derivative(const HashVector& location);
        const DenseRealVector& cached_2nd_derivative(const HashVector& location);

        void invalidate_labels();

        CacheHelper m_DerivativeBuffer;
        CacheHelper m_SecondDerivativeBuffer;

        DenseRealVector m_GenericInBuffer;
        DenseRealVector m_GenericOutBuffer;

        /// Pointer to the regularizer.
        std::unique_ptr<Objective> m_Regularizer;
    };

    /*!
     * \brief A utility class template that, when instatiated with a `MarginFunction`, produces the corresponding
     * linear classifier loss.
     * \tparam MarginFunction A class that needs to provide three scalar functions:
     * `value`, `grad`, and `quad` that give the value, first order, and second order
     * approximation to the function at the given point.
     */
    template<class MarginFunction>
    struct GenericMarginClassifier : public GenericLinearClassifier {
        GenericMarginClassifier(std::shared_ptr<const GenericFeatureMatrix> X,
                                std::unique_ptr<Objective> regularizer,
                                MarginFunction phi) : GenericLinearClassifier( std::move(X), std::move(regularizer) ),
                                                      Phi(std::move(phi)) {

        }

        void calculate_loss(const DenseRealVector& scores,
                            const BinaryLabelVector& labels,
                            DenseRealVector& out) const override {
            assert(scores.size() == labels.size());
            for(int i = 0; i < scores.size(); ++i) {
                real_t margin = scores.coeff(i) * real_t(labels.coeff(i));
                out.coeffRef(i) = Phi.value(margin);
            }
        }

        void calculate_derivative(const DenseRealVector& scores,
                                  const BinaryLabelVector& labels,
                                  DenseRealVector& out) const override {
            assert(scores.size() == labels.size());
            for(int i = 0; i < scores.size(); ++i) {
                real_t label = labels.coeff(i);
                real_t margin = scores.coeff(i) * label;
                out.coeffRef(i) = Phi.grad(margin) * label;
            }
        }

        void calculate_2nd_derivative(const DenseRealVector& scores,
                                      const BinaryLabelVector& labels,
                                      DenseRealVector& out) const override {
            assert(scores.size() == labels.size());
            for(int i = 0; i < scores.size(); ++i) {
                real_t margin = scores.coeff(i) * real_t(labels.coeff(i));
                out.coeffRef(i) = Phi.quad(margin);
            }
        }

        MarginFunction Phi;
    };


    std::unique_ptr<GenericLinearClassifier> make_squared_hinge(std::shared_ptr<const GenericFeatureMatrix> X,
                                                                std::unique_ptr<Objective> regularizer);

    std::unique_ptr<GenericLinearClassifier> make_logistic_loss(std::shared_ptr<const GenericFeatureMatrix> X,
                                                                std::unique_ptr<Objective> regularizer);

    std::unique_ptr<GenericLinearClassifier> make_huber_hinge(std::shared_ptr<const GenericFeatureMatrix> X,
                                                              std::unique_ptr<Objective> regularizer, real_t epsilon);

}

#endif //DISMEC_GENERIC_LINEAR_H

// Copyright (c) 2021, Aalto University, developed by Erik Schultheis
// All rights reserved.
//
// SPDX-License-Identifier: MIT

#ifndef DISMEC_GENERIC_LINEAR_H
#define DISMEC_GENERIC_LINEAR_H

#include "linear.h"

namespace dismec::objective {
    /*!
     * \brief This is a non-templated, runtime-polymorphic generic implementation of the linear classifier objective.
     * \details It is implemented by reducing the operations of an `Objective` to just three functions:
     *  - `calculate_loss()`
     *  - `calculate_derivative()`
     *  - `calculate_2nd_derivative()`
     *
     * These functions operate by writing their result into a pre-allocated array, yielding loss and derivatives for
     * each instance in the feature matrix. By processing these in bulk, we ensure that the indirect-call overhead due
     * to the virtual function becomes negligible. However, the indirect call still means that the optimizer cannot fuse
     * the operations across the call boundary, and the usage of the intermediate buffer increases the memory bandwidth
     * requirements.
     */
    class GenericLinearClassifier : public LinearClassifierBase {
    public:
        GenericLinearClassifier(std::shared_ptr<const GenericFeatureMatrix> X, std::unique_ptr<Objective> regularizer);
    private:
        // declaration of the "unchecked" methods that need to be implemented for an objective.
        //! @{
        real_t value_unchecked(const HashVector& location) override;
        real_t lookup_on_line(real_t position) override;
        void gradient_unchecked(const HashVector& location, Eigen::Ref<DenseRealVector> target) override;
        void gradient_at_zero_unchecked(Eigen::Ref<DenseRealVector> target) override;
        void hessian_times_direction_unchecked(
            const HashVector& location,
            const DenseRealVector& direction,
            Eigen::Ref<DenseRealVector> target) override;
        void diag_preconditioner_unchecked(
            const HashVector& location,
            Eigen::Ref<DenseRealVector> target) override;
        void gradient_and_pre_conditioner_unchecked(
            const HashVector& location,
            Eigen::Ref<DenseRealVector> gradient,
            Eigen::Ref<DenseRealVector> pre) override;
        void project_to_line_unchecked(const HashVector& location, const DenseRealVector& direction) override;
        //! @}

        real_t value_from_xTw(const DenseRealVector& xTw);

        // virtual methods to be implemented in derived classes
        //! @{
        /*!
         * \brief Calculates the loss for each instance.
         * \param[in] scores The scores, i.e. the product weights times features, for each instance.
         * \param[in] labels The binary labels as a vector of plus and minus ones.
         * \param[out] out This vector will be filled with the instance-wise loss value.
         */
        virtual void calculate_loss(const DenseRealVector& scores,
                                    const BinaryLabelVector& labels,
                                    DenseRealVector& out) const = 0;
        /*!
         * \brief Calculates the derivative of the loss with respect to the scores for each instance.
         * \param[in] scores The scores, i.e. the product weights times features, for each instance.
         * \param[in] labels The binary labels as a vector of plus and minus ones.
         * \param[out] out This vector will be filled with the instance-wise loss derivatives.
         */
        virtual void calculate_derivative(const DenseRealVector& scores,
                                          const BinaryLabelVector& labels,
                                          DenseRealVector& out) const = 0;

        /*!
         * \brief Calculates the 2nd derivative of the loss with respect to the scores for each instance.
         * \param[in] scores The scores, i.e. the product weights times features, for each instance.
         * \param[in] labels The binary labels as a vector of plus and minus ones.
         * \param[out] out This vector will be filled with the instance-wise loss 2nd derivatives.
         */
        virtual void calculate_2nd_derivative(const DenseRealVector& scores,
                                              const BinaryLabelVector& labels,
                                              DenseRealVector& out) const = 0;

        /*!
         * \brief Gets the derivative vector for the current location.
         * \details Multiple calls with the same location will reuse a cached result. If no result is available, the
         * derivative is calculated using `calculate_derivative()`.
         * \sa calculate_derivative, m_DerivativeBuffer
         */
        const DenseRealVector& cached_derivative(const HashVector& location);
        /*!
         * \brief Gets the 2nd derivative vector for the current location.
         * \details Multiple calls with the same location will reuse a cached result. If no result is available, the
         * derivative is calculated using `calculate_2nd_derivative()`.
         * \sa calculate_2nd_derivative, m_SecondDerivativeBuffer
         */
        const DenseRealVector& cached_2nd_derivative(const HashVector& location);
        //! @}

        void invalidate_labels() override;

        //! Cached value of the last calculation of the loss derivative. Needs to be invalidated when the labels change.
        CacheHelper m_SecondDerivativeBuffer;
        //! Cached value of the last calculation of the 2nd derivative. Needs to be invalidated when the labels change.
        CacheHelper m_DerivativeBuffer;

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

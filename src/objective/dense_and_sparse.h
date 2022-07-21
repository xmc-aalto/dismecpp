// Copyright (c) 2021, Aalto University, developed by Erik Schultheis
// All rights reserved.
//
// SPDX-License-Identifier: MIT

#ifndef DISMEC_SRC_OBJECTIVE_DENSE_AND_SPARSE_H
#define DISMEC_SRC_OBJECTIVE_DENSE_AND_SPARSE_H

#include "objective.h"
#include "utils/hash_vector.h"

namespace dismec::objective {

    /*!
     * \brief Base class for implementationa of an objective that combines dense features and sparse features.
     * \details This is the shared code for all combined sparse/dense feature linear objectives.
     * \note Unfortunately, I don't think that this can be implemented as a subclass of the `Linear` base,
     * even though it models a linear classifier.
     */
    class DenseAndSparseLinearBase : public Objective {
    public:
        DenseAndSparseLinearBase(std::shared_ptr<const GenericFeatureMatrix> dense_features,
                                 std::shared_ptr<const GenericFeatureMatrix> sparse_features);

        [[nodiscard]] long num_instances() const noexcept;
        [[nodiscard]] long num_variables() const noexcept override;

        [[nodiscard]] BinaryLabelVector& get_label_ref();
        void update_costs(real_t positive, real_t negative);
        void update_features(const DenseFeatures& dense, const SparseFeatures& sparse);

    protected:
        /// actual implementation of `num_variables()`. We need this non-virtual function to be called during the constructor
        [[nodiscard]] long get_num_variables() const noexcept;

        /*!
         * \brief Calculates the vector of feature matrix times weights `w`
         * \param w The weight vector with which to multiply.
         * \details Consecutive calls to this function with the same argument `w`
         * will return a reference to a cached result. However, calling with another
         * value in between will invalidate the cache.
         * \return A reference to the cached result vector.
         */
        const DenseRealVector& x_times_w(const HashVector& w);

        /*!
         * \brief Updates the cached value for x_times_w
         * \param new_weight The new value of w.
         * \param new_result The value of x^T w.
         */
        template<class Derived>
        void update_xtw_cache(const HashVector& new_weight, const Eigen::MatrixBase<Derived>& new_result) {
            // update the cached result to the new value
            m_X_times_w.noalias() = new_result;
            // and set the hash so that we can identify calls using the new weights
            m_Last_W = new_weight.hash();
        }

        /*!
         * \brief Prepares the cache variables for line projection
         * \details This function precomputes \f$ x^T d\f$ and\f$x^T w\f$, so that
         * for a line search parameter `t` we can get \f$ x^T (w + td) \f$ by simple linear
         * combination, i.e. we can skip the matrix multiplication.
         * This can then be used in the implementation of the line search lookup by calling the
         * `line_interpolation()` function.
         * \param location The origin point on the line.
         * \param direction The direction of the line.
         */
        void project_linear_to_line(const HashVector& location, const DenseRealVector& direction);

        [[nodiscard]] auto line_interpolation(real_t t) const {
            return m_LsCache_xTw + t * m_LsCache_xTd;
        }

        void declare_vector_on_last_line(const HashVector& location, real_t t) override {
            update_xtw_cache(location, m_LsCache_xTw + t * m_LsCache_xTd);
        }

        [[nodiscard]] const DenseFeatures& dense_features() const;
        [[nodiscard]] const SparseFeatures& sparse_features() const;

        [[nodiscard]] const DenseRealVector& costs() const;
        [[nodiscard]] const BinaryLabelVector& labels() const;
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

        // needed for regularization
        [[nodiscard]] virtual real_t regularization_value(const DenseRealVector& weights) const = 0;
        virtual void regularization_gradient(const DenseRealVector& weights, Eigen::Ref<DenseRealVector> gradient) const = 0;
        virtual void regularization_gradient_at_zero(Eigen::Ref<DenseRealVector> gradient) const = 0;
        virtual void regularization_preconditioner(const DenseRealVector& weights, Eigen::Ref<DenseRealVector> pre_cond) const = 0;
        virtual void regularization_hessian(const DenseRealVector& weights, const DenseRealVector& direction, Eigen::Ref<DenseRealVector> pre_cond) const = 0;

        /// Pointer to the dense part of the feature matrix
        std::shared_ptr<const GenericFeatureMatrix> m_DenseFeatures;
        /// pointer to the sparse part of the feature matrix
        std::shared_ptr<const GenericFeatureMatrix> m_SparseFeatures;

        /// cache for the last argument to `x_times_w()`.
        VectorHash m_Last_W{};
        /// cache for the last result of `x_times_w()` corresponding to `m_Last_W`.
        DenseRealVector m_X_times_w;

        /// cache for line search implementation: feature times direction
        DenseRealVector m_LsCache_xTd;
        /// cache for line search implementation: feature times weights
        DenseRealVector m_LsCache_xTw;

        /// Label-Dependent costs
        DenseRealVector m_Costs;

        /// Label vector -- use a vector of ints here. We encode label present == 1, absent == -1
        BinaryLabelVector m_Y;
        /// This function will be called whenever m_Y changes so that derived classes can invalidate
        /// their caches.
        void invalidate_labels();

        CacheHelper m_DerivativeBuffer;
        CacheHelper m_SecondDerivativeBuffer;

        DenseRealVector m_LineStart;
        DenseRealVector m_LineDirection;
        DenseRealVector m_LineCache;

        DenseRealVector m_GenericInBuffer;
        DenseRealVector m_GenericOutBuffer;
    };

    template<class MarginFunction, class SparseRegFunction, class DenseRegFunction>
    struct DenseAndSparseMargin : public DenseAndSparseLinearBase {
        DenseAndSparseMargin(std::shared_ptr<const GenericFeatureMatrix> dense_features,
                             std::shared_ptr<const GenericFeatureMatrix> sparse_features,
                             MarginFunction phi, DenseRegFunction dr, real_t drs, SparseRegFunction sr, real_t srs) :
             DenseAndSparseLinearBase(std::move(dense_features), std::move(sparse_features)),
             Phi(std::move(phi)), SparseReg(sr), DenseReg(dr), DenseRegStrength(drs), SparseRegStrength(srs)
        {

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

        [[nodiscard]] real_t regularization_value(const DenseRealVector& weights) const override {
            int sparse_start = dense_features().cols();
            int sparse_end = sparse_start + sparse_features().cols();
            real_t sparse_value = 0;
            real_t dense_value = 0;
            for(int i = 0; i < sparse_start; ++i) {
                dense_value += DenseReg.value(weights.coeff(i));
            }
            for(int i = sparse_start; i < sparse_end; ++i) {
                sparse_value += SparseReg.value(weights.coeff(i));
            }
            return SparseRegStrength * sparse_value + DenseRegStrength * dense_value;
        }

        void regularization_gradient(const DenseRealVector& weights, Eigen::Ref<DenseRealVector> gradient) const override {
            int sparse_start = dense_features().cols();
            int sparse_end = sparse_start + sparse_features().cols();

            // calculate and fill in the pointwise gradient
            for (long i = 0; i < sparse_start; ++i) {
                gradient.coeffRef(i) = DenseRegStrength * DenseReg.grad(weights.coeff(i));
            }
            for (long i = sparse_start; i < sparse_end; ++i) {
                gradient.coeffRef(i) = SparseRegStrength * SparseReg.grad(weights.coeff(i));
            }
        }

        void regularization_gradient_at_zero(Eigen::Ref<DenseRealVector> gradient) const override {
            int sparse_start = dense_features().cols();
            int sparse_end = sparse_start + sparse_features().cols();

            real_t dense_zero = DenseRegStrength * DenseReg.grad(real_t{0});
            real_t sparse_zero = SparseRegStrength * SparseReg.grad(real_t{0});
            // calculate and fill in the pointwise gradient
            for (long i = 0; i < sparse_start; ++i) {
                gradient.coeffRef(i) = dense_zero;
            }
            for (long i = sparse_start; i < sparse_end; ++i) {
                gradient.coeffRef(i) = sparse_zero;
            }
        }

        void regularization_preconditioner(const DenseRealVector& weights, Eigen::Ref<DenseRealVector> pre_cond) const override {
            int sparse_start = dense_features().cols();
            int sparse_end = sparse_start + sparse_features().cols();

            // calculate and fill in the pointwise gradient
            for (long i = 0; i < sparse_start; ++i) {
                pre_cond.coeffRef(i) = DenseRegStrength * DenseReg.quad(weights.coeff(i));
            }
            for (long i = sparse_start; i < sparse_end; ++i) {
                pre_cond.coeffRef(i) = SparseRegStrength * SparseReg.quad(weights.coeff(i));
            }
        }

        void regularization_hessian(const DenseRealVector& weights, const DenseRealVector& direction, Eigen::Ref<DenseRealVector> target) const override {
            int sparse_start = dense_features().cols();
            int sparse_end = sparse_start + sparse_features().cols();

            // calculate and fill in the pointwise gradient
            for (long i = 0; i < sparse_start; ++i) {
                target.coeffRef(i) = DenseRegStrength * DenseReg.quad(weights.coeff(i)) * direction.coeff(i);
            }
            for (long i = sparse_start; i < sparse_end; ++i) {
                target.coeffRef(i) = SparseRegStrength * SparseReg.quad(weights.coeff(i)) * direction.coeff(i);
            }
        }

        MarginFunction Phi;
        real_t DenseRegStrength;
        DenseRegFunction DenseReg;
        real_t SparseRegStrength;
        SparseRegFunction SparseReg;
    };

    std::unique_ptr<DenseAndSparseLinearBase> make_sp_dense_squared_hinge(
        std::shared_ptr<const GenericFeatureMatrix> dense_features,
        real_t dense_reg_strength,
        std::shared_ptr<const GenericFeatureMatrix> sparse_features,
        real_t sparse_reg_strength
        );
}

#endif //DISMEC_SRC_OBJECTIVE_DENSE_AND_SPARSE_H

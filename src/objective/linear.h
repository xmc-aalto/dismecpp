// Copyright (c) 2021, Aalto University, developed by Erik Schultheis
// All rights reserved.
//
// SPDX-License-Identifier: MIT

#ifndef DISMEC_LINEAR_H
#define DISMEC_LINEAR_H

#include "objective.h"
#include "utils/hash_vector.h"

namespace dismec::objective {
    /*!
     * \brief Base class for objectives that use a linear classifier.
     * \details This base provides functions for caching the matrix multiplication \f$ x^T w \f$
     * that can be used by derived classes.
     * This class does not assume a specific type for the feature matrix, but stores it as a
     * `GenericFeatureMatrix`. It also provides caching for the line search computations.
     *
     * This base class also stores the per-instance const vector and the ground-truth label vector.
     * \attention Derived classes may want to cache certain results based on the predicted scores and
     * the labels. For the input, `location` parameter, we can easily check if we can reuse a cached
     * result, but this will break if the labels change. Therefore, such classes should implement the
     * virtual function `invalidate_labels()` so that cached results will be invalidated. This function
     * will be called by the base class whenever the label vector is modified.
     */
    class LinearClassifierBase : public Objective {
    public:
        LinearClassifierBase(std::shared_ptr<const GenericFeatureMatrix> X);

        [[nodiscard]] long num_instances() const noexcept;
        [[nodiscard]] long num_variables() const noexcept override;

        [[nodiscard]] BinaryLabelVector& get_label_ref();
        void update_costs(real_t positive, real_t negative);
    protected:
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

        [[nodiscard]] const GenericFeatureMatrix& generic_features() const;
        [[nodiscard]] const DenseFeatures& dense_features() const;
        [[nodiscard]] const SparseFeatures& sparse_features() const;

        [[nodiscard]] const DenseRealVector& costs() const;
        [[nodiscard]] const BinaryLabelVector& labels() const;
    private:
        /// we keep a refcounted pointer to the training features.
        /// this is to support shared memory parallelization of multilabel training.
        /// Derived classes may form a pointer to the concrete type of m_FeatureMatrix
        std::shared_ptr<const GenericFeatureMatrix> m_FeatureMatrix;

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
        virtual void invalidate_labels() = 0;
    };

    /*!
     * \brief Implementation helper for linear classifier derived classes.
     * \tparam Derived The derived class, which is expected to provide the following
     * functions:
     * \code
     * template<typename Derived>
     * real_t value_from_xTw(const DenseRealVector& cost, const BinaryLabelVector& labels, const Eigen::DenseBase<Derived>& xTw);
     * void gradient_and_diag()
     * \endcode
     */
    template<class Derived>
    class LinearClassifierImpBase : public LinearClassifierBase {
    public:
        LinearClassifierImpBase(std::shared_ptr<const GenericFeatureMatrix> X, std::unique_ptr<Objective> regularizer) :
        LinearClassifierBase( std::move(X) ), m_Regularizer( std::move(regularizer) ) {};
    protected:
        const Derived& derived() const {
            return static_cast<const Derived&>(*this);
        }

        Derived& derived() {
            return static_cast<Derived&>(*this);
        }

        real_t value_unchecked(const HashVector& location) override {
            const DenseRealVector& xTw = x_times_w(location);
            return derived().value_from_xTw(costs(), labels(), xTw) + m_Regularizer->value(location);
        }

        real_t lookup_on_line(real_t position) override {
            real_t f = Derived::value_from_xTw(costs(), labels(), line_interpolation(position));
            return f + m_Regularizer->lookup_on_line(position);
        }

        void project_to_line_unchecked(const HashVector& location, const DenseRealVector& direction) override {
            project_linear_to_line(location, direction);
            m_Regularizer->project_to_line(location, direction);
        }

        void gradient_unchecked(const HashVector& location, Eigen::Ref<DenseRealVector> target) override {
            m_Regularizer->gradient(location, target);
            derived().gradient_imp(location, target);
        }


        void gradient_at_zero_unchecked(Eigen::Ref<DenseRealVector> target) override {
            m_Regularizer->gradient_at_zero(target);
            derived().gradient_at_zero_imp(target);
        }

        void hessian_times_direction_unchecked(const HashVector& location,
                                               const DenseRealVector& direction,
                                               Eigen::Ref<DenseRealVector> target) override {
            m_Regularizer->hessian_times_direction(location, direction, target);
            derived().hessian_times_direction_imp(location, direction, target);
        }

        void diag_preconditioner_unchecked(const HashVector& location, Eigen::Ref<DenseRealVector> target) override {
            m_Regularizer->diag_preconditioner(location, target);
            derived().diag_preconditioner_imp(location, target);
        }

        void gradient_and_pre_conditioner_unchecked(const HashVector& location, Eigen::Ref<DenseRealVector> gradient,
                                                    Eigen::Ref<DenseRealVector> pre) override {
            m_Regularizer->gradient_and_pre_conditioner(location, gradient, pre);
            derived().gradient_and_pre_conditioner_imp(location, gradient, pre);
        }
    private:
        /// Pointer to the regularizer.
        std::unique_ptr<Objective> m_Regularizer;
    };
}

#endif //DISMEC_LINEAR_H

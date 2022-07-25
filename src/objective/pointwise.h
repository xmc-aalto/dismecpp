// Copyright (c) 2021, Aalto University, developed by Erik Schultheis
// All rights reserved.
//
// SPDX-License-Identifier: MIT

#ifndef DISMEC_POINTWISE_H
#define DISMEC_POINTWISE_H

#include "objective.h"
#include "utils/hash_vector.h"
#include "utils/throw_error.h"

namespace dismec::objective {
    /*!
     * \brief Base class for pointwise regularization functions
     * \tparam CRTP This class is implemented as a CRTP (https://en.wikipedia.org/wiki/Curiously_recurring_template_pattern) template
     * \details This class provides default implementations for the objective, given that the CRTP derived class
     * defined three scalar functions:
     *  - `point_wise_value`
     *  - `point_wise_grad`
     *  - `point_wise_quad`
     * The regularizer is defined via \f$ R(w) = \sum_{i=1}^n f(w_i) \f$, and the three functions
     * define the value, derivative and second derivative of `f`. More strictly speaking, `f` is
     * expected to be a scalar function, the function `g_x` defined by `point_wise_value`, `point_wise_grad`
     * and `point_wise_quad` is expected to define a lower-bounded upper bound on `f`. Let the return values
     * of the three function be denoted as `a`, `b`, and `c`, then the function `g_x(t)` fulfills
     * \f$ C \leq g_x(t) \leq f(t) \f$ with \f$ g_x(x) = f(x) \f$.
     *
     * This base class also includes code to automatically remove the last weight from any calculations ("ignore_bias"),
     * and for scaling the regularizer.
     *
     * The line projection algorithm supplied by the default implementation caches the starting point and direction
     * vector. In cases where more efficient algorithms are available, it is possible to override the `project_to_line()`
     * and `lookup_on_line()` functions. In that case, `m_LineStart` and `m_LineDirection` will not be set and will
     * not cause unneeded memory consumption.
     */

    template<class CRTP>
    class PointWiseRegularizer : public Objective {
    public:
        /// Constructor for the regularizer. `scale` defines the prefactor by which the entire regularizer will be
        /// scaled. This value has to be larger than zero. `ignore_bias` declares whether the last entry in the weight
        /// vector should be considered a bias term, and be ignored in the calculations.
        explicit PointWiseRegularizer(real_t scale = 1, bool ignore_bias = false);

        /// The pointwise regularizer can act on arbitrarily sized vectors, so `num_variables() == -1`.
        [[nodiscard]] long num_variables() const noexcept final { return -1; }

        [[nodiscard]] real_t value_unchecked(const HashVector& location) override;

        void hessian_times_direction_unchecked(const HashVector& location,
                                               const DenseRealVector& direction,
                                               Eigen::Ref<DenseRealVector> target) override;

        void gradient_unchecked(const HashVector& location, Eigen::Ref<DenseRealVector> target) override;

        void gradient_at_zero_unchecked(Eigen::Ref<DenseRealVector> target) override;

        void diag_preconditioner_unchecked(const HashVector& location, Eigen::Ref<DenseRealVector> target) override;

        void project_to_line_unchecked(const HashVector& location, const DenseRealVector& direction) override;
        real_t lookup_on_line(real_t a) override;

        /// Returns whether the last entry in the weight vector should be treated as a bias, and be ignored in
        /// the regularization.
        [[nodiscard]] bool dont_regularize_bias() const { return m_LastWeightIsBias; }

        /// Returns the common scale factor for the entire regularizer.
        [[nodiscard]] real_t scale() const { return m_Scale; }

    private:
        bool m_LastWeightIsBias = false;
        real_t m_Scale = 1.0;

        /// This variable will cache the starting position for tracking a line projection.
        /// It is set in `project_to_line()`.
        DenseRealVector m_LineStart;
        /// This variable will cache the direction for tracking a line projection.
        /// It is set in `project_to_line()`.
        DenseRealVector m_LineDirection;

        /// calls `point_wise_value()` of the implementing class
        [[nodiscard]] real_t point_wise_value_(real_t x) const {
            return static_cast<const CRTP*>(this)->point_wise_value(x);
        }

        /// calls `point_wise_grad()` of the implementing class
        [[nodiscard]] real_t point_wise_grad_(real_t x) const {
            return static_cast<const CRTP*>(this)->point_wise_grad(x);
        }

        /// calls `point_wise_quad()` of the implementing class
        [[nodiscard]] real_t point_wise_quad_(real_t x) const {
            return static_cast<const CRTP*>(this)->point_wise_quad(x);
        }

        [[nodiscard]] long get_loop_bound(const HashVector& location) const {
            return dont_regularize_bias() ? location->size() - 1u : location->size();
        }
    };

    template<class T>
    PointWiseRegularizer<T>::PointWiseRegularizer(real_t scale, bool ignore_bias) :
        m_LastWeightIsBias(ignore_bias), m_Scale(scale) {
        if(m_Scale < 0) {
            THROW_EXCEPTION(std::logic_error, "Scale must be non-negative");
        }
    }


    template<class T>
    [[nodiscard]] real_t PointWiseRegularizer<T>::value_unchecked(const HashVector& location) {
        /*
         * Here, we just add up the pointwise values, and perform a single rescaling at the end.
         */
        real_t result = 0.0;
        long loop_bound = get_loop_bound(location);
        for (long i = 0; i < loop_bound; ++i) {
            result += point_wise_value_(location->coeff(i));
        }
        return m_Scale * result;
    }

    template<class T>
    void PointWiseRegularizer<T>::hessian_times_direction_unchecked(
            const HashVector& location,
            const DenseRealVector& direction,
            Eigen::Ref<DenseRealVector> target) {

        // the hessian / quadratic approximation is diagonal, so for the matrix-vector product
        // we have to multiply each diagonal component with the corresponding coefficient of the direction.
        long loop_bound = get_loop_bound(location);
        for (long i = 0; i < loop_bound; ++i) {
            target.coeffRef(i) = m_Scale * point_wise_quad_(location->coeff(i)) * direction.coeff(i);
        }

        // if the last weight is interpreted as bias, we make sure that the corresponding target value is set to zero.
        if (dont_regularize_bias())
            target.coeffRef(loop_bound) = real_t{0};
    }

    template<class T>
    void PointWiseRegularizer<T>::gradient_unchecked(const HashVector& location, Eigen::Ref<DenseRealVector> target) {
        long loop_bound = get_loop_bound(location);

        // calculate and fill in the pointwise gradient
        for (long i = 0; i < loop_bound; ++i) {
            target.coeffRef(i) = m_Scale * point_wise_grad_(location->coeff(i));
        }

        // if the last weight is interpreted as bias, we make sure that the corresponding target value is set to zero.
        if (dont_regularize_bias())
            target.coeffRef(target.size() - 1) = real_t{0};
    }

    template<class T>
    void PointWiseRegularizer<T>::gradient_at_zero_unchecked(Eigen::Ref<DenseRealVector> target) {
        // for gradient at zero, we only need to calculate the point-wise gradient once, and can then
        // copy this over the entire vector.
        real_t grad_at_zero = point_wise_grad_(real_t{0});
        target.setConstant(grad_at_zero * m_Scale);

        // if the last weight is interpreted as bias, we make sure that the corresponding target value is set to zero.
        if (dont_regularize_bias())
            target.coeffRef(target.size() - 1) = real_t{0};
    }

    template<class T>
    void PointWiseRegularizer<T>::diag_preconditioner_unchecked(
            const HashVector& location,
            Eigen::Ref<DenseRealVector> target) {
        // same as hessian_times_direction_unchecked, only now we don't multiply by a direction vector
        long loop_bound = get_loop_bound(location);
        for (long i = 0; i < loop_bound; ++i) {
            target.coeffRef(i) = m_Scale * point_wise_quad_(location->coeff(i));
        }

        if (dont_regularize_bias())
            target.coeffRef(target.size() - 1) = real_t{0};
    }

    template<class T>
    void PointWiseRegularizer<T>::project_to_line_unchecked(const HashVector& location, const DenseRealVector& direction) {
        // projecting to line just saves the location and direction for use with the `lookup_on_line` function.
        m_LineStart = location.get();
        m_LineDirection = direction;
    }

    template<class T>
    real_t PointWiseRegularizer<T>::lookup_on_line(real_t a) {
        // The same as value, only using the interpolated positions
        real_t result = 0.0;
        // make sure we sum over the correct subset.
        long loop_bound = dont_regularize_bias() ? m_LineStart.size() - 1u: m_LineStart.size();
        for(long i = 0; i < loop_bound; ++i) {
            result += point_wise_value_(m_LineStart.coeff(i) + a * m_LineDirection.coeff(i));
        }
        return m_Scale * result;
    }

}

#endif //DISMEC_POINTWISE_H

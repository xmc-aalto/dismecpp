// Copyright (c) 2021, Aalto University, developed by Erik Schultheis
// All rights reserved.
//
// SPDX-License-Identifier: MIT

#ifndef DISMEC_REGULARIZERS_IMP_H
#define DISMEC_REGULARIZERS_IMP_H

#include "objective.h"
#include "hash_vector.h"
#include "pointwise.h"

/*!
 * Regularizers are implementations of `Objective` that depend only on the weight vector, but
 * make no reference to external data. From the `Objective` interface, this is not visible,
 * since the data is not part of the public interface.
 */

namespace objective {

    /*!
     * \brief This class implements a squared norm (L2) regularizer. Thus `f(x) = 0.5 |x|^2`.
     * \details Since this is a quadratic function with diagonal Hessian, the
     * implementation is mostly trivial. The only interesting code is that of
     * `project_to_line()` and `lookup_on_line()`, because these functions can
     * do some smart so that `lookup_on_line()` can be implemented in `O(1)`.
     * The regularizer admits an additional scale parameter by which its value (and thus gradient, hessian etc)
     * will be scaled.
     */
    class SquaredNormRegularizer : public PointWiseRegularizer<SquaredNormRegularizer> {
    public:
        explicit SquaredNormRegularizer(real_t scale = 1, bool ignore_bias=false);

        [[nodiscard]] static real_t point_wise_value(real_t x);
        [[nodiscard]] static real_t point_wise_grad(real_t x);
        [[nodiscard]] static real_t point_wise_quad(real_t x);

        [[nodiscard]] real_t value_unchecked(const HashVector& location) override;

        /*!
         * This function calculates helper values to facilitate a fast implementation of `lookup_on_line()`.
         * It is based on the decomposition
         * \f[
         * \|x + t d\|^2 = \|x\|^2 + 2 t \langle x, d \rangle + t^2 \|d\|^2.
         * \f]
         * Therefore, we calculate \f$ \|x\|^2\f$,  \f$\langle x, d \rangle\f$ and \f$\|d\|^2 \f$
         * here.
         */
        void project_to_line_unchecked(const HashVector& location, const DenseRealVector& direction) override;
        real_t lookup_on_line(real_t a) override;

    private:
        real_t m_LsCache_w02;
        real_t m_LsCache_d2;
        real_t m_LsCache_dTw;
    };

    /*!
     * \brief This class implements a huber regularizer.
     * \details The regularizer acts pointwise on each weight by applying a Huber function. This
     * function is smooth, and consists of a parabola around zero and linear parts for larger values.
     * These linear parts are potentially problematic when doing second order optimization, as the
     * Hessian would be zero there. For that reason, the `hessian_times_direction()` function does not
     * use the actual Hessian of the function, but a quadratic upper bound on the function. The details
     * are discussed in section 2.1 of the pdf.
     *
     * By choosing the switching point between quadratic and linear expression to be very small, this
     * function can be used as a continuously differentiable approximation to L1 regularization.
     */
    class HuberRegularizer : public PointWiseRegularizer<HuberRegularizer> {
    public:
        /*!
         * \brief Constructor for a Huber regularizer objective.
         * \param epsilon The cutoff point between quadratic and linear behaviour.
         * \throws invalid_argument, if `epsilon` < 0.
         */
        explicit HuberRegularizer(real_t epsilon, real_t scale = 1.0, bool ignore_bias=false);

        [[nodiscard]] real_t point_wise_value(real_t x) const;
        [[nodiscard]] real_t point_wise_grad(real_t x) const;
        [[nodiscard]] real_t point_wise_quad(real_t x) const;
    private:
        real_t m_Epsilon{1};
    };

    class ElasticNetRegularizer : public PointWiseRegularizer<ElasticNetRegularizer> {
    public:
        /*!
         * \brief Constructor for a ElasticNet regularizer objective.
         * \param epsilon The cutoff point between quadratic and linear behaviour for the L1 part.
         * \param scale Global scaling factor for the loss
         * \param interp Interpolation factor between L1 and L2 loss. Needs to be between 0 and 1, with 0 resulting in
         * L1 loss and 1 in L2 loss.
         * \throws invalid_argument, if `epsilon` < 0.
         */
        ElasticNetRegularizer(real_t epsilon, real_t scale, real_t interp, bool ignore_bias=false);

        [[nodiscard]] real_t point_wise_value(real_t x) const;
        [[nodiscard]] real_t point_wise_grad(real_t x) const;
        [[nodiscard]] real_t point_wise_quad(real_t x) const;
    private:
        real_t m_Epsilon{1};
        real_t m_L1_Factor{0.5};
        real_t m_L2_Factor{0.5};
    };
}

#endif //DISMEC_REGULARIZERS_IMP_H

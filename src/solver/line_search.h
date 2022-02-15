// Copyright (c) 2021, Aalto University, developed by Erik Schultheis
// All rights reserved.
//
// SPDX-License-Identifier: MIT

#ifndef DISMEC_LINE_SEARCH_H
#define DISMEC_LINE_SEARCH_H

#include <functional>
#include "hyperparams.h"

namespace solvers
{
    /*!
     * \brief Result of a Line Search operation.
     */
    struct sLineSearchResult {
        double Value;               //!< The function value at the accepted position
        double StepSize;            //!< The step size used to reach that position.
        int NumIters;
    };

    /*!
     * \brief Backtracking line search using the armijo rule.
     * \details The algorithm starts with a step size of `initial_step` (default: 1) times
     * the direction vector. If the reduction of function value is not at least
     * `eta` (default 0.01) times the expected reduction according to the gradient (i.e. the linear approximation)
     * a step that is smaller by a factor of `alpha` (default 0.5) is tried. This is repeated until the maximum
     * number of steps has been reached (default: 20). If no adequate point is found then, a step of `0` is returned.
     *
     * This class gets supplied with a "projected objective", i.e. an objective function g: R->R that is related to
     * the original as follows: `g(a) = f(x_0 + a d)` where `d` is the search direction and `x_0` is the starting
     * point.
     */
    class BacktrackingLineSearch : public HyperParameterBase {
    public:
        BacktrackingLineSearch();

        // get and set the parameters
        [[nodiscard]] double get_initial_step() const { return m_StepSize; }
        /// sets the initial step multiplied. Throws `std::invalid_argument` if `s` is not positive.
        void set_initial_step(double s);

        [[nodiscard]] double get_alpha() const { return m_Alpha; }
        /// sets the alpha parameter. Throws `std::invalid_argument` if `a` is not in `(0, 1)`
        void set_alpha(double a);

        [[nodiscard]] double get_eta() const { return m_Eta; }
        /// sets the eta parameter. Throws `std::invalid_argument` if `e` is not in `(0, 1)`
        void set_eta(double e);

        [[nodiscard]] long get_max_steps() const { return m_MaxSteps; }
        /// sets the eta parameter. Throws `std::invalid_argument` if `n` is not positive
        void set_max_steps(long n);

        /*!
         * \param projected_objective: A function that when called with parameter \f$ \alpha \f$ , will return the value of
         * the objective along the search direction \f$d\f$ given by \f$ g(\alpha) = f(x_0 + \alpha d) \f$.
         * \param gTs: Gradient at 0, i.e. \f$ \left. \frac{\partial g}{\partial \alpha}\right|_{\alpha=0}\f$.
         * \param f_init: \f$ g(0) = f(x_0) \f$.
        */
        sLineSearchResult search(const std::function<double(double)>& projected_objective, double gTs, double f_init) const;

    private:
        double m_StepSize = 1.0;

        // scale factor for the step size
        double m_Alpha = 0.5;
        // required reduction
        double m_Eta = 0.01;
        // maximum number of steps to perform
        long m_MaxSteps = 20;
    };
}
#endif //DISMEC_LINE_SEARCH_H

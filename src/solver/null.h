// Copyright (c) 2021, Aalto University, developed by Erik Schultheis
// All rights reserved.
//
// SPDX-License-Identifier: MIT

#ifndef DISMEC_NULL_H
#define DISMEC_NULL_H

#include "solver/minimizer.h"
#include "hash_vector.h"

namespace solvers {
    /*!
     * \brief Optimizer that does not change the initial vector.
     * \details This class is provided for testing purposes, e.g. to save the model consisting only
     * of the weights given by some initialization strategy. Depending on the setting, this optimizer
     * can either calculate the loss and gradient at the initial point (and thus return a bad, yet
     * sensible `MinimizationResult`), or skip the calculation entirely.
     */
    class NullOptimizer : public Minimizer {
    public:
        /*!
         * \brief Constructor, specifies whether to run any calculations at all
         * \param calc If this is true, then the minimizer calculates the value and gradient of the loss
         * function at the initial point and returns a meaningful `MinimizationResult`, otherwise these
         * are just set to zero and the minimization is essentially a no-op.
         */
        NullOptimizer(bool calc);
    private:
        MinimizationResult run(objective::Objective& objective, Eigen::Ref<DenseRealVector> init) override;

        DenseRealVector m_Gradient;
        HashVector      m_Weights;

        bool m_CalcLoss;
    };
}

#endif //DISMEC_NULL_H

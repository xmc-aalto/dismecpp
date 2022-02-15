// Copyright (c) 2021, Aalto University, developed by Erik Schultheis
// All rights reserved.
//
// SPDX-License-Identifier: MIT

#ifndef DISMEC_NEWTON_H
#define DISMEC_NEWTON_H

#include "solver/minimizer.h"
#include "solver/cg.h"
#include "solver/line_search.h"
#include "hash_vector.h"

namespace solvers
{
    class NewtonWithLineSearch : public Minimizer {
    public:
        explicit NewtonWithLineSearch(long num_variables);

        // hyperparameters
        void set_epsilon(double eps);
        double get_epsilon() const { return m_Epsilon; }

        void set_maximum_iterations(long max_iter);
        long get_maximum_iterations() const { return m_MaxIter; }

        void set_alpha_preconditioner(double alpha);
        double get_alpha_preconditioner() const { return m_Alpha_PCG; }

    private:
        MinimizationResult run(objective::Objective& objective, Eigen::Ref<DenseRealVector> init) override;

        // newton parameters
        double m_Epsilon = 0.01;
        double m_Alpha_PCG = 0.01;
        long m_MaxIter = 1000;

        // sub-algorithms
        CGMinimizer m_CG_Solver;
        BacktrackingLineSearch m_LineSearcher;

        // buffers
        DenseRealVector m_Gradient;
        DenseRealVector m_PreConditioner;
        HashVector      m_Weights;

        void record_iteration(int iter, int cg_iter, real_t gnorm, real_t objective, const sLineSearchResult& step, real_t gnorm0);
    };
}

#endif //DISMEC_NEWTON_H

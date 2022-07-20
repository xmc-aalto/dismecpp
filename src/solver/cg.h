// Copyright (c) 2021, Aalto University, developed by Erik Schultheis
// All rights reserved.
//
// SPDX-License-Identifier: MIT

#ifndef DISMEC_CG_H
#define DISMEC_CG_H

#include "matrix_types.h"
#include <functional>
#include "utils/hyperparams.h"

namespace dismec::solvers {

    /*! \class CGMinimizer
     *  \brief Approximately solve a linear equation `Ax + b = 0`.
     *  \details This class uses a CG algorithm to approximately solve `Ax + b = 0`. It does not
     *  require the entire matrix `A` but only a function that allows to calculate `Ad` for an
     *  arbitrary vector `d`. This class has a size of approximately `5n sizeof(double)` for
     *  `n` being the number of variables.
     */
    class CGMinimizer : public HyperParameterBase {
    public:
        using MatrixVectorProductFn = std::function<void(const DenseRealVector &, Eigen::Ref<DenseRealVector>)>;

        explicit CGMinimizer(std::size_t num_vars);

        /// Solves `Ax+b=0`. returns the number of iterations
        int minimize(const MatrixVectorProductFn &A, const DenseRealVector &b, const DenseRealVector &M);

        /// returns the solution vector found by the last minimize call
        const DenseRealVector& get_solution() const { return m_S; }

        [[nodiscard]] double get_epsilon() const { return m_Epsilon; }

        void set_epsilon(double v) { m_Epsilon = v; }

    private:
        int do_minimize(const MatrixVectorProductFn &A, const DenseRealVector &b, const DenseRealVector &M);

        std::size_t m_Size;
        real_t m_Epsilon = 0.5;

        // vector caches to prevent allocations
        DenseRealVector m_A_times_d;
        DenseRealVector m_S;                // s from the CG algorithm
        DenseRealVector m_Residual;         // r_k from the CG algorithm
        DenseRealVector m_Conjugate;        // p_k from the CG algorithm
    };
}

#endif //DISMEC_CG_H

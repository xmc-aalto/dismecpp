// Copyright (c) 2021, Aalto University, developed by Erik Schultheis
// All rights reserved.
//
// SPDX-License-Identifier: MIT

#include "cg.h"
#include "doctest.h"
#include <spdlog/spdlog.h>

using namespace solvers;

CGMinimizer::CGMinimizer(std::size_t num_vars) : m_Size(num_vars) {
    m_A_times_d = DenseRealVector(num_vars);
    m_S = DenseRealVector(num_vars);
    m_Residual = DenseRealVector(num_vars);
    m_Conjugate = DenseRealVector(num_vars);

    declare_hyper_parameter("epsilon", &CGMinimizer::get_epsilon, &CGMinimizer::set_epsilon);
}

int CGMinimizer::minimize(const MatrixVectorProductFn& A, const DenseRealVector& b, const DenseRealVector& M) {
    int result = do_minimize(A, b, M);
    return result;
}

int CGMinimizer::do_minimize(const MatrixVectorProductFn& A, const DenseRealVector& b, const DenseRealVector& M) {
    // in comments, we use z to denote Residual/M
    m_S.setZero();                                  // assume x_0 = 0
    m_Residual = -b;                                // note: We are solving Ax+b = 0, typically CG is used for Ax = b.
    m_Conjugate = m_Residual.array() / M.array();

    real_t Q = 0;

    auto zT_dot_r = m_Conjugate.dot(m_Residual);             // at this point: m_Conjugate == z
    real_t gMinv_norm = std::sqrt(zT_dot_r);                 // = sqrt(-b^T / M b)
    real_t cgtol = std::min(m_Epsilon, std::sqrt(gMinv_norm));

    int max_cg_iter = std::max(m_Size, std::size_t(5));
    for(int cg_iter = 1; cg_iter <= max_cg_iter; ++cg_iter) {
        A(m_Conjugate, m_A_times_d);
        real_t dAd = m_Conjugate.dot(m_A_times_d);
        if(dAd < 1e-16) {
            return cg_iter;
        }
        real_t alpha = zT_dot_r / dAd;
        m_S += alpha * m_Conjugate;
        m_Residual -= alpha * m_A_times_d;

        // Using quadratic approximation as CG stopping criterion
        real_t newQ = -real_t{0.5}*(m_S.dot(m_Residual - b));
        real_t Qdiff = newQ - Q;
        if (newQ <= 0 && Qdiff <= 0)
        {
            if (cg_iter * Qdiff >= cgtol * newQ) {
                return cg_iter;     // success
            }
        }
        else
        {
            spdlog::warn("quadratic approximation > 0 or increasing in {}th CG iteration. Old Q: {}, New Q: {}",
                         cg_iter, Q, newQ);
            return cg_iter;
        }
        Q = newQ;

        // z == r.array() / M.array()
        real_t znewTrnew = (m_Residual.array() / M.array()).matrix().dot(m_Residual);
        real_t beta = znewTrnew / zT_dot_r;
        m_Conjugate = m_Conjugate * beta + (m_Residual.array() / M.array()).matrix();
        zT_dot_r = znewTrnew;
    }

    spdlog::warn("reached maximum number of CG steps ({}). Remaining error is {}", max_cg_iter, Q);

    return max_cg_iter;
}

TEST_CASE("conjugate gradient") {
    auto minimizer = CGMinimizer(5);
    minimizer.set_epsilon(0.001);
    types::DenseColMajor<real_t> A = types::DenseColMajor<real_t>::Random(5, 5);
    A = (A*A.transpose()).eval();  // ensure symmetric, PSD matrix
    DenseRealVector b = DenseRealVector::Random(5);
    DenseRealVector m = DenseRealVector::Ones(5);

    minimizer.minimize([&](const DenseRealVector& d, Eigen::Ref<DenseRealVector> out){
        out = A * d;
    }, b, m);

    DenseRealVector solution = minimizer.get_solution();
    DenseRealVector sol = A * solution + b;
    CHECK(sol.norm() == doctest::Approx(0.0));
}
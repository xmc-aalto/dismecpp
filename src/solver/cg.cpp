// Copyright (c) 2021, Aalto University, developed by Erik Schultheis
// All rights reserved.
//
// SPDX-License-Identifier: MIT

#include "cg.h"
#include <spdlog/spdlog.h>

using namespace dismec;
using dismec::solvers::CGMinimizer;

CGMinimizer::CGMinimizer(long num_vars) : m_Size(num_vars) {
    m_A_times_d = DenseRealVector(num_vars);
    m_S = DenseRealVector(num_vars);
    m_Residual = DenseRealVector(num_vars);
    m_Conjugate = DenseRealVector(num_vars);

    declare_hyper_parameter("epsilon", &CGMinimizer::get_epsilon, &CGMinimizer::set_epsilon);
}

long CGMinimizer::minimize(const MatrixVectorProductFn& A, const DenseRealVector& b, const DenseRealVector& M) {
    long result = do_minimize(A, b, M);
    return result;
}

long CGMinimizer::do_minimize(const MatrixVectorProductFn& A, const DenseRealVector& b, const DenseRealVector& M) {
    // in comments, we use z to denote Residual/M
    m_S.setZero();                                  // assume x_0 = 0
    m_Residual = -b;                                // note: We are solving Ax+b = 0, typically CG is used for Ax = b.
    m_Conjugate = m_Residual.array() / M.array();

    real_t Q = 0;

    auto zT_dot_r = m_Conjugate.dot(m_Residual);             // at this point: m_Conjugate == z
    real_t gMinv_norm = std::sqrt(zT_dot_r);                 // = sqrt(-b^T / M b)
    real_t cg_tol = std::min(m_Epsilon, std::sqrt(gMinv_norm));

    long max_cg_iter = std::max(m_Size, CG_MIN_ITER_BOUND);
    for(long cg_iter = 1; cg_iter <= max_cg_iter; ++cg_iter) {
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
            if (cg_iter * Qdiff >= cg_tol * newQ) {
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

#include "doctest.h"

TEST_CASE("conjugate gradient") {
    const int TEST_SIZE = 5;
    auto minimizer = CGMinimizer(TEST_SIZE);
    minimizer.set_epsilon(0.0001);
    types::DenseColMajor<real_t> A = types::DenseColMajor<real_t>::Random(TEST_SIZE, TEST_SIZE);
    A = (A*A.transpose()).eval();  // ensure symmetric, PSD matrix
    DenseRealVector b = DenseRealVector::Random(TEST_SIZE);
    DenseRealVector m = DenseRealVector::Ones(TEST_SIZE);

    minimizer.minimize([&](const DenseRealVector& d, Eigen::Ref<DenseRealVector> out){
        out = A * d;
    }, b, m);

    DenseRealVector solution = minimizer.get_solution();
    DenseRealVector sol = A * solution + b;
    CHECK(sol.norm() == doctest::Approx(0.0));
}
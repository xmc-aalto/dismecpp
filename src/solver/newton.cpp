// Copyright (c) 2021, Aalto University, developed by Erik Schultheis
// All rights reserved.
//
// SPDX-License-Identifier: MIT

#include "newton.h"
#include "line_search.h"
#include "utils/hash_vector.h"
#include "solver/cg.h"
#include "stats/collection.h"
#include "stats/timer.h"

using namespace dismec::solvers;

namespace {
    using dismec::stats::stat_id_t;

    stat_id_t STAT_GRADIENT_NORM_0{0};
    stat_id_t STAT_OBJECTIVE_VALUE{1};
    stat_id_t STAT_GRADIENT_NORM{2};
    stat_id_t STAT_GRADIENT{3};
    stat_id_t STAT_PRECONDITIONER{4};
    stat_id_t STAT_WEIGHT_VECTOR{5};
    stat_id_t STAT_LINESEARCH_STEPSIZE{6};
    stat_id_t STAT_CG_ITERS{7};
    stat_id_t STAT_ITER_TIME{8};
    stat_id_t STAT_LS_FAIL{9};
    stat_id_t STAT_LS_STEPS{10};
    stat_id_t STAT_PROGRESS{11};
    stat_id_t STAT_ABSOLUTE_STEP{12};

    dismec::stats::tag_id_t TAG_ITERATION{0};
};

NewtonWithLineSearch::NewtonWithLineSearch(long num_variables) : m_CG_Solver(num_variables),
                                                                 m_Gradient(num_variables), m_PreConditioner(num_variables),
                                                                 m_Weights(DenseRealVector(num_variables))
{
    declare_hyper_parameter("epsilon", &NewtonWithLineSearch::get_epsilon, &NewtonWithLineSearch::set_epsilon);
    declare_hyper_parameter("max-steps", &NewtonWithLineSearch::get_maximum_iterations, &NewtonWithLineSearch::set_maximum_iterations);
    declare_hyper_parameter("alpha-pcg", &NewtonWithLineSearch::get_alpha_preconditioner, &NewtonWithLineSearch::set_alpha_preconditioner);
    declare_sub_object("cg", &NewtonWithLineSearch::m_CG_Solver);
    declare_sub_object("search", &NewtonWithLineSearch::m_LineSearcher);

    declare_stat(STAT_GRADIENT_NORM_0, {"grad_norm_0", "|g_0|"});
    declare_stat(STAT_OBJECTIVE_VALUE, {"objective", "loss"});
    declare_stat(STAT_GRADIENT_NORM, {"grad_norm", "|g|"});
    declare_stat(STAT_GRADIENT, {"gradient", "|g_i|"});
    declare_stat(STAT_PRECONDITIONER, {"preconditioner", "|H_ii|"});
    declare_stat(STAT_WEIGHT_VECTOR, {"weight_vector", "|w_i|"});
    declare_stat(STAT_LINESEARCH_STEPSIZE, {"linesearch_step"});
    declare_stat(STAT_CG_ITERS, {"cg_iters", "#iters"});
    declare_stat(STAT_ITER_TIME, {"iter_time", "duration [µs]"});
    declare_stat(STAT_LS_FAIL, {"linesearch_fail", "#instances"});
    declare_stat(STAT_LS_STEPS, {"linesearch_iters", "#steps"});
    declare_stat(STAT_PROGRESS, {"progress", "|g|/|eps g_0|"});
    declare_stat(STAT_ABSOLUTE_STEP, {"newton_step", ""});

    declare_tag(TAG_ITERATION, "iteration");
}


MinimizationResult NewtonWithLineSearch::run(objective::Objective& objective, Eigen::Ref<DenseRealVector> init)
{
    // calculate gradient norm at w=0 for stopping condition.
    // first, check if the objective supports fast grad
    objective.gradient_at_zero(m_Gradient);
    real_t gnorm0 = m_Gradient.norm();
    record(STAT_GRADIENT_NORM_0, gnorm0);


    m_Weights = init;
    real_t f, gnorm;

    /*!
     * \internal We are using \ref Objective::gradient_and_pre_conditioner here, and in the loop below.
     * It can be more efficient to calculate both values at once (if the loss vector is sparse, only one
     * loop is needed), but as a consequence we perform one more calculation of the preconditioner vector
     * than necessary. At least for squared hinge, this seems to result in a net win (the more iterations
     * we have to do, the more speedup we get).
     */
    {
        set_tag(TAG_ITERATION, 0);
        auto scope_timer = make_timer(STAT_ITER_TIME);

        f = objective.value(m_Weights);
        objective.gradient_and_pre_conditioner(m_Weights, m_Gradient, m_PreConditioner);
        gnorm = m_Gradient.norm();

        record_iteration(0, 0, gnorm, f, sLineSearchResult{0, 0, 0}, m_Epsilon * gnorm0);
    }

    real_t f_start = f;
    real_t gnorm_start = gnorm;

    // OK, there is something wrong already!
    if(!std::isfinite(f) || !std::isfinite(gnorm) || !std::isfinite(gnorm0)) {
        spdlog::error("Invalid newton optimization: initial value: {}, gradient norm: {}, gnorm_0: {}", f, gnorm, gnorm0);
        return {MinimizerStatus::FAILED, 0, f, gnorm};
    }

    if(m_Logger) {
        m_Logger->info("initial: f={:<5.3} |g|={:<5.3} |g_0|={:<5.3} eps={:<5.3}", f, gnorm, gnorm0, m_Epsilon);
    }

    if (gnorm <= m_Epsilon * gnorm0)
        return {MinimizerStatus::SUCCESS, 0, f, gnorm, f, gnorm};

    for(int iter = 1; iter <= m_MaxIter; ++iter) {
        set_tag(TAG_ITERATION, iter);
        auto scope_timer = make_timer(STAT_ITER_TIME);

        // regularize the preconditioner: M = (1-a)I + aM
        m_PreConditioner = (1 - m_Alpha_PCG) + (m_PreConditioner * m_Alpha_PCG).array();

        // Here, we solve min \| Hd + g \|
        int cg_iter = m_CG_Solver.minimize([&](const DenseRealVector& d, Eigen::Ref<DenseRealVector> o) {
            objective.hessian_times_direction(m_Weights, d, o);
        }, m_Gradient, m_PreConditioner);

        const auto& cg_solution = m_CG_Solver.get_solution();

        real_t fold = f;
        objective.project_to_line(m_Weights, cg_solution);
        auto ls_result = m_LineSearcher.search([&](real_t a){ return objective.lookup_on_line(a); },
                                               m_Gradient.dot(cg_solution), f);

        if (ls_result.StepSize == 0)
        {
            spdlog::warn("line search failed in iteration {} of newton optimization. Current objective value: {:.3}, "
                         "gradient norm: {:.3} (target: {:.3}), squared search dir: {:.3}",
                         iter, f, gnorm, m_Epsilon * gnorm0, cg_solution.squaredNorm());
            init = m_Weights.get();
            record(STAT_LS_FAIL, 1);
            return {MinimizerStatus::FAILED, iter, f, gnorm, f_start, gnorm_start};
        }

        f = ls_result.Value;
        real_t absolute_improvement = fold - f;
        m_Weights = m_Weights + cg_solution * ls_result.StepSize;
        objective.declare_vector_on_last_line(m_Weights, ls_result.StepSize);
        objective.gradient_and_pre_conditioner(m_Weights, m_Gradient, m_PreConditioner);

        gnorm = m_Gradient.norm();

        record_iteration(iter, cg_iter, gnorm, f, ls_result, m_Epsilon * gnorm0);
        record(STAT_ABSOLUTE_STEP, [&]() -> real_t { return cg_solution.norm(); });

        if (gnorm <= m_Epsilon * gnorm0) {
            init = m_Weights.get();
            return {MinimizerStatus::SUCCESS, iter, f, gnorm, f_start, gnorm_start};
        }
        if (f < -1.0e+32)
        {
            spdlog::warn("Objective appears to be unbounded (got value {:.2})", f);
            return {MinimizerStatus::DIVERGED, iter, f, gnorm, f_start, gnorm_start};
        }
        if (abs(absolute_improvement) <= 1.0e-12 * abs(f))
        {
            spdlog::warn("relative improvement too low");
            return {MinimizerStatus::FAILED, iter, f, gnorm, f_start, gnorm_start};
        }
    }

    init = m_Weights.get();
    return {MinimizerStatus::TIMED_OUT, m_MaxIter, f, gnorm, f_start, gnorm_start};
}

void NewtonWithLineSearch::record_iteration(int iter, int cg_iter, real_t gnorm, real_t objective, const sLineSearchResult& step, real_t gnorm0) {
    record(STAT_GRADIENT_NORM, gnorm);
    record(STAT_GRADIENT, m_Gradient);
    record(STAT_PRECONDITIONER, m_PreConditioner);
    record(STAT_OBJECTIVE_VALUE, objective);
    record(STAT_LINESEARCH_STEPSIZE, real_t(step.StepSize));
    record(STAT_LS_STEPS, step.NumIters);
    record(STAT_CG_ITERS, cg_iter);
    record(STAT_WEIGHT_VECTOR, m_Weights.get());
    record(STAT_PROGRESS, gnorm / gnorm0);

    if(m_Logger) {
        m_Logger->info("iter {:3}: f={:<10.8} |g|={:<8.4} CG={:<3} line-search={:<4.2}",
                       iter, objective, gnorm, cg_iter, step.StepSize);
    }
}

void NewtonWithLineSearch::set_epsilon(double eps) {
    if(eps <= 0) {
        spdlog::error("Non-positive epsilon {} specified for newton minimization", eps);
        throw std::invalid_argument("Epsilon must be larger than zero.");
    }
    m_Epsilon = eps;
}

void NewtonWithLineSearch::set_maximum_iterations(long max_iter) {
    if(max_iter <= 0) {
        spdlog::error("Non-positive iteration limit {} specified for newton minimization", max_iter);
        throw std::invalid_argument("maximum iterations must be larger than zero.");
    }
    m_MaxIter = max_iter;
}

void NewtonWithLineSearch::set_alpha_preconditioner(double alpha) {
    if(alpha <= 0 || alpha >= 1) {
        spdlog::error("The `alpha_pcg` parameter needs to be between 0 and 1, got {} ", alpha);
        throw std::invalid_argument("alpha_pcg not in (0, 1)");
    }
    m_Alpha_PCG = alpha;
}

#include "doctest.h"
#include <Eigen/Dense>

using namespace dismec;

TEST_CASE("newton with line search hyperparameters") {
    NewtonWithLineSearch nwls{2};

    // direct interface
    nwls.set_epsilon(0.1);
    CHECK(nwls.get_epsilon() == 0.1);

    nwls.set_maximum_iterations(500);
    CHECK(nwls.get_maximum_iterations() == 500);

    nwls.set_alpha_preconditioner(0.4);
    CHECK(nwls.get_alpha_preconditioner() == 0.4);

    // error checking
    CHECK_THROWS(nwls.set_epsilon(-0.4));
    CHECK_THROWS(nwls.set_maximum_iterations(0));
    CHECK_THROWS(nwls.set_alpha_preconditioner(-0.1));
    CHECK_THROWS(nwls.set_alpha_preconditioner(1.1));

    // hp interface
    nwls.set_hyper_parameter("epsilon", 0.25);
            CHECK( std::get<double>(nwls.get_hyper_parameter("epsilon")) == 0.25);
    nwls.set_hyper_parameter("max-steps", 50l);
            CHECK( std::get<long>(nwls.get_hyper_parameter("max-steps")) == 50);
    nwls.set_hyper_parameter("alpha-pcg", 0.3);
            CHECK( std::get<double>(nwls.get_hyper_parameter("alpha-pcg")) == 0.3);
}

TEST_CASE("solve square objective") {
    struct QuadraticObjective : public dismec::objective::Objective {
        QuadraticObjective(types::DenseColMajor<real_t> m, DenseRealVector s) : A(std::move(m)), b(std::move(s)),
            m_LocCache(A.row(0)){}

        [[nodiscard]] long num_variables() const noexcept override {
            return b.size();
        }

        real_t value_unchecked(const HashVector& location) override {
            return location->dot(A * location) + location->dot(b);
        }
        void gradient_unchecked(const HashVector& location, Eigen::Ref<DenseRealVector> target) override {
            target= 2 * A * location + b;
        }
        void hessian_times_direction_unchecked(const HashVector& location, const DenseRealVector& direction,
                                     Eigen::Ref<DenseRealVector> target) override
        {
            target =  2 * A * direction;
        }

        void project_to_line_unchecked(const HashVector& location, const DenseRealVector& direction) override {
            m_DirCache = direction;
            m_LocCache = location;
        };
        real_t lookup_on_line(real_t position) override {
            return value(HashVector{m_LocCache + m_DirCache * position});
        };

        types::DenseColMajor<real_t> A;
        DenseRealVector b;

        DenseRealVector m_DirCache;
        HashVector m_LocCache;
    };
    types::DenseColMajor<real_t> mat(4, 4);
    mat << 1.0, 1.0, 0.0, 0.0,
            1.0, 1.0, -1.0, 0.0,
            0.0, -1.0, 2.0, 0.0,
            0.0, 0.0, 0.0, 1.0;
    // ensure PSD symmetric matrix
    mat = (mat.transpose() * mat).eval();

    DenseRealVector vec(4);
    vec << 1.0, 2.0, 0.0, -2.0;
    QuadraticObjective objective{mat, vec};

    DenseRealVector w = DenseRealVector::Random(4);

    NewtonWithLineSearch solver(w.size());
    solver.minimize(objective, w);

    // solve quadratic minimum directly:
    DenseRealVector direct = -mat.inverse() * vec / 2;
    for(int i = 0; i < w.size(); ++i) {
        CHECK(w.coeff(i) == doctest::Approx(direct.coeff(i)));
    }
}

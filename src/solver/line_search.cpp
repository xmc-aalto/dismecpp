// Copyright (c) 2021, Aalto University, developed by Erik Schultheis
// All rights reserved.
//
// SPDX-License-Identifier: MIT

#include "line_search.h"

#include "utils/hash_vector.h"
#include "spdlog/spdlog.h"

using namespace dismec::solvers;

BacktrackingLineSearch::BacktrackingLineSearch() {
    declare_hyper_parameter("step-size", &BacktrackingLineSearch::get_initial_step, &BacktrackingLineSearch::set_initial_step);
    declare_hyper_parameter("alpha", &BacktrackingLineSearch::get_alpha, &BacktrackingLineSearch::set_alpha);
    declare_hyper_parameter("eta", &BacktrackingLineSearch::get_eta, &BacktrackingLineSearch::set_eta);
    declare_hyper_parameter("max-steps", &BacktrackingLineSearch::get_max_steps, &BacktrackingLineSearch::set_max_steps);
}

sLineSearchResult BacktrackingLineSearch::search(
        const std::function<double(double)>& projected_objective, double gTs, double f_init) const
{
    double step = m_StepSize;

    double f_old = f_init;
    double f_new = f_init;

    for(int num_linesearch=0; num_linesearch < m_MaxSteps; ++num_linesearch)
    {
        f_new = projected_objective(step);
        if (f_new - f_old <= m_Eta * step * gTs)
            return {f_new, step, num_linesearch};
        else
            step *= m_Alpha;
    }

    spdlog::warn("Line search failed. Final step size: {:.3}, df = {:.3}",
                 step, f_old - f_new);
    return {f_old, 0.0, (int)m_MaxSteps};
}

void BacktrackingLineSearch::set_initial_step(double s) {
    if(s <= 0) {
        throw std::invalid_argument("step size must be positive");
    }
    m_StepSize = s;
}

void BacktrackingLineSearch::set_max_steps(long n) {
    if(n <= 0) {
        throw std::invalid_argument("max num steps must be positive");
    }
    m_MaxSteps = n;
}

void BacktrackingLineSearch::set_alpha(double a) {
    if(a <= 0) {
        throw std::invalid_argument("alpha must be positive");
    } else if (a >= 1) {
        throw std::invalid_argument("alpha must be less than 1");
    }
    m_Alpha = a;
}

void BacktrackingLineSearch::set_eta(double e) {
    if(e <= 0) {
        throw std::invalid_argument("eta must be positive");
    } else if (e >= 1) {
        throw std::invalid_argument("eta must be less than 1");
    }
    m_Eta = e;
}

#include "doctest.h"

TEST_CASE("test_get_set") {
    BacktrackingLineSearch searcher{};
    searcher.set_alpha(0.4);
    CHECK(searcher.get_alpha() == 0.4);
    searcher.set_initial_step(1.8);
    CHECK(searcher.get_initial_step() == 1.8);
    searcher.set_max_steps(5);
    CHECK(searcher.get_max_steps() == 5);
    searcher.set_eta(0.8);
    CHECK(searcher.get_eta() == 0.8);

    CHECK_THROWS(searcher.set_alpha(-0.1));
    CHECK_THROWS(searcher.set_alpha(0.0));
    CHECK_THROWS(searcher.set_alpha(1.0));
    CHECK_THROWS(searcher.set_eta(0.0));
    CHECK_THROWS(searcher.set_eta(-.1));
    CHECK_THROWS(searcher.set_eta(1.0));
    CHECK_THROWS(searcher.set_initial_step(0.0));
    CHECK_THROWS(searcher.set_initial_step(-.1));
    CHECK_THROWS(searcher.set_max_steps(0));
    CHECK_THROWS(searcher.set_max_steps(-1));
}


TEST_CASE("backtracking line search") {

    auto quad_objective = [](double x_0, double d) {
        auto fun = [=](double a) {
            a *= d;
            return (a + x_0) * (a + x_0);
        };

        return fun;
    };

    BacktrackingLineSearch searcher{};

    // looking in the wrong direction. Search fails
    SUBCASE("x^2 wrong direction") {
        auto objective = quad_objective(1.0, 1.0);
        auto result = searcher.search(objective, 2.0, objective(0));
        CHECK(result.StepSize == 0.0);
        CHECK(result.Value == 1.0);
    }

    // looking in the right direction -- first step fulfills our condition
    // note that inverting the direction also means inverting the gradient g := <df, s>
    SUBCASE("x^2 right direction") {
        auto objective = quad_objective(1.0, -1.0);
        auto result = searcher.search(objective, -2.0, objective(0));
        CHECK(result.StepSize == 1.0);
        CHECK(result.Value == 0.0);
    }


    // looking in the right direction -- we have to backtrack until we reach the minimum
    SUBCASE("x^2 right direction too large") {
        auto objective = quad_objective(1.0, -8.0);
        auto result = searcher.search(objective, -2.0*8, objective(0));
        CHECK(result.StepSize == 1.0/8.0);
        CHECK(result.Value == 0.0);
    }
}
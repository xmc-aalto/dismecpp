// Copyright (c) 2021, Aalto University, developed by Erik Schultheis
// All rights reserved.
//
// SPDX-License-Identifier: MIT

#include "minimizer.h"
#include <vector>
#include <stdexcept>
#include "spdlog/spdlog.h"
#include "stats/collection.h"

using namespace dismec::solvers;

Minimizer::Minimizer(std::shared_ptr<spdlog::logger> logger) :
    m_Logger(std::move(logger)) {
}

Minimizer::~Minimizer() = default;

void Minimizer::set_logger(std::shared_ptr<spdlog::logger> logger) {
    m_Logger = std::move(logger);
}

MinimizationResult Minimizer::minimize(objective::Objective& objective, Eigen::Ref<DenseRealVector> init) {
    long n = objective.num_variables();
    if(init.size() != n) {
        throw std::invalid_argument("Weight vector incompatible with problem size");
    }

    auto start = std::chrono::steady_clock::now();
    auto result = run(objective, init);
    result.Duration = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - start);
    return result;
}

#include "doctest.h"

using namespace dismec;

namespace
{
    using dismec::objective::Objective;

    //! A minimizer to be used in test cases that returns a fixed result.
    class MockMinimizer : public Minimizer {
        MinimizationResult run(Objective& objective, Eigen::Ref<DenseRealVector> init) override {
            MinimizationResult result;
            result.Outcome = MinimizerStatus::DIVERGED;
            result.FinalGrad = 5.0;
            result.FinalValue = 2.0;
            result.NumIters = 55;
            result.Duration = std::chrono::milliseconds(4242);    // this should be set by the parent class - we use this value
            // so we can verify that
            return result;
        }
    };

    //! An objective to be used in test cases. Does not do any computations, but just resturns constants.
    struct MockObjective : public Objective {
        [[nodiscard]] long num_variables() const noexcept override { return 12; }
        real_t value_unchecked(const HashVector& location) override { return 5.0; }
        void gradient_unchecked(const HashVector& location, Eigen::Ref<DenseRealVector> target) override {};
        void hessian_times_direction_unchecked(const HashVector& location, const DenseRealVector& direction,
                                     Eigen::Ref<DenseRealVector> target) override {};
        void project_to_line_unchecked(const HashVector& location, const DenseRealVector& direction) override {};
        real_t lookup_on_line(real_t position) override { return 0.0; };
    };
}

/*!
 * \test These tests verify that the Minimizer base class `minimize` performs the correct parameter checking and
 * result passing. These are:
 *   - verify the correct size of the initial value
 *   - time the duration for the minimization
 *   - pass the result through unchanged
 */
TEST_CASE("minimizer base class minimize") {
    MockMinimizer mnm;
    DenseRealVector vec;
    MockObjective goal;

    // check that having an improperly sized initial vector throws
    SUBCASE("init vector verifier") {
        CHECK_THROWS(mnm.minimize(goal, vec));
    }

    // check that the base class does the timing
    SUBCASE("timing") {
        vec.resize(goal.num_variables());
        auto result = mnm.minimize(goal, vec);
        CHECK(result.Duration != std::chrono::milliseconds(4242));
    }

    // check that the other results are passed through as-is
    SUBCASE("result passing") {
        vec.resize(goal.num_variables());
        auto result = mnm.minimize(goal, vec);
        CHECK(result.Outcome == MinimizerStatus::DIVERGED);
        CHECK(result.FinalGrad == 5.0);
        CHECK(result.FinalValue == 2.0);
        CHECK(result.NumIters == 55);
    }
}

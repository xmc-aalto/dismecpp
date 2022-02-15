// Copyright (c) 2021, Aalto University, developed by Erik Schultheis
// All rights reserved.
//
// SPDX-License-Identifier: MIT

#include "null.h"
#include "stats/collection.h"

using namespace solvers;


namespace {
    /*
    stats::stat_id_t STAT_OBJECTIVE_VALUE{1};
    stats::stat_id_t STAT_GRADIENT_NORM{2};
    stats::stat_id_t STAT_GRADIENT{3};
    stats::stat_id_t STAT_WEIGHT_VECTOR{5};
    stats::stat_id_t STAT_ITER_TIME{8};
    stats::stat_id_t STAT_PROGRESS{11};
*/
    stats::tag_id_t TAG_ITERATION{0};
};



MinimizationResult NullOptimizer::run(objective::Objective& objective, Eigen::Ref<DenseRealVector> init)
{
    set_tag(TAG_ITERATION, 0);
    if(!m_CalcLoss) {
        return {MinimizerStatus::SUCCESS, 0, 0, 0, 0, 0};
    }

    m_Weights = init;
    m_Gradient.resize(init.size());

    real_t f = objective.value(m_Weights);
    objective.gradient(m_Weights, m_Gradient);
    real_t gnorm = m_Gradient.norm();

    // OK, there is something wrong already!
    if(!std::isfinite(f) || !std::isfinite(gnorm)) {
        spdlog::error("Invalid optimization: initial value: {}, gradient norm: {}", f, gnorm);
        return {MinimizerStatus::FAILED, 0, f, gnorm};
    }

    return {MinimizerStatus::SUCCESS, 0, f, gnorm, f, gnorm};

}

NullOptimizer::NullOptimizer(bool calc) : m_Weights(DenseRealVector(1)), m_CalcLoss(calc) {
    declare_tag(TAG_ITERATION, "iteration");
}

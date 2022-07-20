// Copyright (c) 2021, Aalto University, developed by Erik Schultheis
// All rights reserved.
//
// SPDX-License-Identifier: MIT

#ifndef DISMEC_MINIMIZER_H
#define DISMEC_MINIMIZER_H

#include "objective/objective.h"
#include "utils/hyperparams.h"
#include "spdlog/spdlog.h"
#include "stats/tracked.h"
#include <chrono>

namespace dismec::solvers
{
    enum class MinimizerStatus {
        SUCCESS,            //!< The returned result is a minimum according to the stopping criterion of the algorithm
        DIVERGED,           //!< The optimization objective appears to be unbounded.
        TIMED_OUT,          //!< The maximum number of iterations has been reached but no minimum has been found
        FAILED              //!< Some internal operation failed.
    };

    struct MinimizationResult {
        MinimizerStatus Outcome;
        long NumIters;
        double FinalValue;
        double FinalGrad;
        double InitialValue;
        double InitialGrad;
        std::chrono::milliseconds Duration;
    };

    class Minimizer : public HyperParameterBase, public stats::Tracked {
    public:
        explicit Minimizer(std::shared_ptr<spdlog::logger> logger = {});
        ~Minimizer() override;

        MinimizationResult minimize(objective::Objective& objective, Eigen::Ref<DenseRealVector> init);

        /// sets the logger object that is used for progress tracking.
        void set_logger(std::shared_ptr<spdlog::logger> logger);

    protected:
        std::shared_ptr<spdlog::logger> m_Logger;

        virtual MinimizationResult run(objective::Objective& objective, Eigen::Ref<DenseRealVector> init) = 0;
    };
}

#endif //DISMEC_MINIMIZER_H

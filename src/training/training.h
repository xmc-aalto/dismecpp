// Copyright (c) 2021, Aalto University, developed by Erik Schultheis
// All rights reserved.
//
// SPDX-License-Identifier: MIT

#ifndef DISMEC_TRAINING_H
#define DISMEC_TRAINING_H

#include "spec.h"
#include "parallel/task.h"
#include "solver/minimizer.h"
#include <memory>
#include <functional>
#include "fwd.h"
#include "data/types.h"

namespace dismec
{
    /*! \class TrainingTaskGenerator
     *  \brief Generates tasks for training weights for the i'th label
     *  \details This task generator produces tasks which train the weights
     *  for the different labels. The tasks are generated as specified by a `TrainingSpec`:
     *  Each thread generates a minimizer, an objective, and an initializer. The objective and optimizer
     *  are updated for each new label using the corresponding functions of the `TrainingSpec` object.
     */
    class TrainingTaskGenerator : public parallel::TaskGenerator {
    public:
        explicit TrainingTaskGenerator(std::shared_ptr<TrainingSpec> spec, label_id_t begin_label=label_id_t{0},
                              label_id_t end_label=label_id_t{-1});
        ~TrainingTaskGenerator() override;

        void run_tasks(long begin, long end, thread_id_t thread_id) override;
        void prepare(long num_threads, long chunk_size) override;
        void init_thread(thread_id_t thread_id) override;
        void finalize() override;
        [[nodiscard]] long num_tasks() const override;

        [[nodiscard]] const std::shared_ptr<model::Model>& get_model() const { return m_Model; }
        [[nodiscard]] const std::vector<solvers::MinimizationResult>& get_results() const { return m_Results; }

    private:
        void run_task(long task_id, thread_id_t thread_id);

        /*!
         * \brief Runs the training of a single label.
         * \param label_id The id of the label which will be trained.
         * \param thread_id The id of the thread on which training is running.
         * \return The return value of the minimizer.
         */
        solvers::MinimizationResult train_label(label_id_t label_id, thread_id_t thread_id);

        //
        std::shared_ptr<TrainingSpec> m_TaskSpec;

        // training only a partial model?
        label_id_t m_LabelRangeBegin;
        label_id_t m_LabelRangeEnd;

        // result variables
        std::shared_ptr<model::Model> m_Model;
        std::vector<solvers::MinimizationResult> m_Results;

        // thread-local caches
        std::vector<DenseRealVector> m_ThreadLocalWorkingVector;
        std::vector<std::unique_ptr<solvers::Minimizer>> m_ThreadLocalMinimizer;
        std::vector<std::shared_ptr<objective::Objective>> m_ThreadLocalObjective;
        std::vector<std::unique_ptr<init::WeightsInitializer>> m_ThreadLocalWeightInit;
        std::vector<std::unique_ptr<postproc::PostProcessor>> m_ThreadLocalPostProc;
        std::vector<std::unique_ptr<ResultStatsGatherer>> m_ResultGatherers;
    };

    struct TrainingResult {
        bool IsFinished = false;
        std::shared_ptr<model::Model> Model;
        real_t TotalLoss;
        real_t TotalGrad;
    };

    TrainingResult run_training(parallel::ParallelRunner& runner, std::shared_ptr<TrainingSpec> spec,
                                label_id_t begin_label=label_id_t{0}, label_id_t end_label=label_id_t{-1});

}

#endif //DISMEC_TRAINING_H

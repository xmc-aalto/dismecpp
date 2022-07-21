// Copyright (c) 2021, Aalto University, developed by Erik Schultheis
// All rights reserved.
//
// SPDX-License-Identifier: MIT

#ifndef DISMEC_SRC_TRAINING_CASCADE_H
#define DISMEC_SRC_TRAINING_CASCADE_H

#include "training.h"
#include "parallel/numa.h"

namespace dismec {
    class CascadeTraining : public TrainingSpec {
    public:
        CascadeTraining(std::shared_ptr<const DatasetBase> tfidf_data,
                        std::shared_ptr<const GenericFeatureMatrix> dense_data,
                        HyperParameters hyper_params,
                        std::shared_ptr<init::WeightInitializationStrategy> dense_init,
                        real_t dense_reg,
                        std::shared_ptr<init::WeightInitializationStrategy> sparse_init,
                        real_t sparse_reg,
                        std::shared_ptr<postproc::PostProcessFactory> post_proc,
                        std::shared_ptr<TrainingStatsGatherer> gatherer,
                        std::shared_ptr<const std::vector<std::vector<long>>> shortlist = nullptr);

        long num_features() const override { return m_NumFeatures; }

        [[nodiscard]] std::shared_ptr<objective::Objective> make_objective() const override;

        [[nodiscard]] std::unique_ptr<solvers::Minimizer> make_minimizer() const override;

        [[nodiscard]] std::unique_ptr<init::WeightsInitializer> make_initializer() const override;

        [[nodiscard]] std::shared_ptr<model::Model>
        make_model(long num_features, model::PartialModelSpec spec) const override;

        void update_minimizer(solvers::Minimizer& base_minimizer, label_id_t label_id) const override;

        void update_objective(objective::Objective& base_objective, label_id_t label_id) const override;

        [[nodiscard]] std::unique_ptr<postproc::PostProcessor>
        make_post_processor(const std::shared_ptr<objective::Objective>& objective) const override;

        TrainingStatsGatherer& get_statistics_gatherer() override;

    private:
        HyperParameters m_NewtonSettings;

        parallel::NUMAReplicator<const GenericFeatureMatrix> m_SparseReplicator;
        parallel::NUMAReplicator<const GenericFeatureMatrix> m_DenseReplicator;

        std::shared_ptr<const std::vector<std::vector<long>>> m_Shortlist;

        // post processing
        std::shared_ptr<postproc::PostProcessFactory> m_PostProcessor;

        // initial conditions
        std::shared_ptr<init::WeightInitializationStrategy> m_DenseInitStrategy;
        std::shared_ptr<init::WeightInitializationStrategy> m_SparseInitStrategy;

        std::shared_ptr<TrainingStatsGatherer> m_StatsGather;

        long m_NumFeatures;
        double m_BaseEpsilon;

        real_t m_DenseReg;
        real_t m_SparseReg;
    };
}

#endif //DISMEC_SRC_TRAINING_CASCADE_H

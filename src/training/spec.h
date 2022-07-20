// Copyright (c) 2021, Aalto University, developed by Erik Schultheis
// All rights reserved.
//
// SPDX-License-Identifier: MIT

#ifndef DISMEC_TRAINING_SPEC_H
#define DISMEC_TRAINING_SPEC_H

#include <memory>
#include "fwd.h"
#include "matrix_types.h"
#include "spdlog/fwd.h"
#include "objective/regularizers.h"

namespace dismec
{
    /*!
     * \brief This class gathers the setting-specific parts of the training process.
     * \details The \ref TrainingSpec class is responsible for generating and updating
     * the \ref Minimizer and \ref Objective that will be used in the \ref TrainingTaskGenerator.
     * \todo should we give the dataset to each operation, or maybe just set it in the beginning? I think maybe at some
     * point convert all this dataset stuff to use shared_ptr everywhere.
     */
    class TrainingSpec {
    public:
        explicit TrainingSpec(std::shared_ptr<const DatasetBase> data) :
                m_Data(std::move(data)) {
        }
        virtual ~TrainingSpec() = default;

        [[nodiscard]] const DatasetBase& get_data() const { return *m_Data; }

        [[nodiscard]] virtual long num_features() const;

        /*!
         * \brief Makes an \ref Objective object suitable for the dataset.
         * \details This is called before the actual work of the training threads starts, so that we can pre-allocate
         * all the necessary buffers.
         */
        [[nodiscard]] virtual std::shared_ptr<objective::Objective> make_objective() const = 0;

        /*!
         * \brief Makes a \ref Minimizer object suitable for the dataset.
         * \details This is called before the actual work of the training threads starts, so that we can pre-allocate
         * all the necessary buffers. Is is called in the thread will run the minimizer, so that the
         * default NUMA-local strategy should be reasonable.
         */
        [[nodiscard]] virtual std::unique_ptr<solvers::Minimizer> make_minimizer() const = 0;

        /*!
         * \brief Makes a \ref WeightsInitializer object.
         * \details This is called before the actual work of the training threads starts, so that we can pre-allocate
         * all the necessary buffers. Is is called in the thread were the returned initializer will be used. so that the
         * default NUMA-local strategy should be reasonable.
         */
        [[nodiscard]] virtual std::unique_ptr<init::WeightsInitializer> make_initializer() const = 0;

        /*!
         * \brief Makes a \ref PostProcessor object.
         * \details This is called before the actual work of the training threads starts, so that we can pre-allocate
         * all the necessary buffers. Is is called in the thread were the returned post processor will be used, so that the
         * default NUMA-local strategy should be reasonable.
         * The `PostProcessor` can be adapted to the thread_local `objective` that is supplied here.
         */
        [[nodiscard]] virtual std::unique_ptr<postproc::PostProcessor> make_post_processor(const std::shared_ptr<objective::Objective>& objective) const = 0;

        /*!
         * \brief Creates the model that will be used to store the results.
         * \details This extension point gives the `TrainingSpec` a way to decide whether
         * the model storage used shall be a sparse or a dense model, or maybe some wrapper pointing
         * to external memory.
         * TODO why is this a shared_ptr and not a unique_ptr ?
         * \param num_features Number of input features for the model.
         * \param spec Partial model specification for the created model.
         */
        [[nodiscard]] virtual std::shared_ptr<model::Model> make_model(long num_features, model::PartialModelSpec spec) const = 0;

        /*!
         * \brief Updates the setting of the \ref Minimizer for handling label `label_id`.
         * \details This is needed e.g. to set a stopping criterion that depends on the number of
         * positive labels. This function will be called concurrently from different threads, but each thread
         * will make calls with different `minimizer` parameter.
         * \param minimizer A \ref Minimizer. This is assumed to have been created using \ref make_minimizer(),
         * so in particular it should by `dynamic_cast`-able to the actual \ref Minimizer type used by this
         * \ref TrainingSpec.
         * \param label_id The id of the label inside the dataset for which we update the minimizer.
         */
        virtual void update_minimizer(solvers::Minimizer& minimizer, label_id_t label_id) const = 0;

        /*!
         * \brief Updates the setting of the \ref Objective for handling label `label_id`.
         * \details This will e.g. extract the corresponding label vector from the dataset and supply it
         * to the objective. This function will be called concurrently from different threads, but each thread
         * will call with a different `objective` parameter.
         * \param objective An \ref Objective. This is assumed to have been created using \ref make_objective(),
         * so in particular it should by `dynamic_cast`-able to the actual \ref Objective type used by this
         * \ref TrainingSpec.
         * \param label_id The id of the label inside the dataset for which we update the objective.
        */
        virtual void update_objective(objective::Objective& objective, label_id_t label_id) const = 0;

        [[nodiscard]] virtual TrainingStatsGatherer& get_statistics_gatherer() = 0;

        // logger
        [[nodiscard]] const std::shared_ptr<spdlog::logger>& get_logger() const {
            return m_Logger;
        }

        void set_logger(std::shared_ptr<spdlog::logger> l) {
            m_Logger = std::move(l);
        }

    private:
        std::shared_ptr<const DatasetBase> m_Data;

        /// logger to be used for info logging
        std::shared_ptr<spdlog::logger> m_Logger;
    };

    enum class RegularizerType {
        REG_L2,
        REG_L1,
        REG_L1_RELAXED,
        REG_HUBER,
        REG_ELASTIC_50_50,
        REG_ELASTIC_90_10
    };

    enum class LossType {
        SQUARED_HINGE,
        LOGISTIC,
        HUBER_HINGE,
        HINGE
    };

    using real_t = float;

    std::shared_ptr<objective::Objective> make_loss(
            LossType type,
            std::shared_ptr<const GenericFeatureMatrix> X,
            std::unique_ptr<objective::Objective> regularizer);

    using RegularizerSpec = std::variant<objective::SquaredNormConfig, objective::HuberConfig, objective::ElasticConfig>;

    struct DismecTrainingConfig {
        std::shared_ptr<WeightingScheme> Weighting;
        std::shared_ptr<init::WeightInitializationStrategy> Init;
        std::shared_ptr<postproc::PostProcessFactory> PostProcessing;
        std::shared_ptr<TrainingStatsGatherer> StatsGatherer;
        bool Sparse;
        RegularizerSpec Regularizer;
        LossType Loss;
    };

    struct CascadeTrainingConfig {
        std::shared_ptr<init::WeightInitializationStrategy> DenseInit;
        std::shared_ptr<init::WeightInitializationStrategy> SparseInit;
        std::shared_ptr<postproc::PostProcessFactory> PostProcessing;
        std::shared_ptr<TrainingStatsGatherer> StatsGatherer;

        real_t DenseReg  = 1.0;
        real_t SparseReg = 1.0;
    };

    std::shared_ptr<TrainingSpec> create_dismec_training(std::shared_ptr<const DatasetBase> data,
                                                         HyperParameters params,
                                                         DismecTrainingConfig config);

    std::shared_ptr<TrainingSpec> create_cascade_training(std::shared_ptr<const DatasetBase> data,
                                                          std::shared_ptr<const GenericFeatureMatrix> dense,
                                                          std::shared_ptr<const std::vector<std::vector<long>>> shortlist,
                                                          HyperParameters params,
                                                          CascadeTrainingConfig config);
}

#endif //DISMEC_TRAINING_SPEC_H

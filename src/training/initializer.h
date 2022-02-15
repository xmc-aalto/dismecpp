// Copyright (c) 2021, Aalto University, developed by Erik Schultheis
// All rights reserved.
//
// SPDX-License-Identifier: MIT

#ifndef DISMEC_INITIALIZER_H
#define DISMEC_INITIALIZER_H

#include "matrix_types.h"
#include "parallel/numa.h"
#include "stats/tracked.h"
#include "spec.h"
#include <memory>
class label_id_t;
class DatasetBase;

namespace model {
    class Model;
}

namespace objective {
    class Objective;
}

namespace init {
    /*!
     * \brief Base class for all weight initializers.
     * \details Weight initializers are used by `TrainingTaskGenerator` to generate the initial weight vector
     * before the actual training process starts. A well-chosen initial vector can significantly decrease training time.
     * Of course, this requires that the calculation of the initial vector itself be fast. For more discussion as well
     * as the list of available initializers, see \ref initvectors . Instances of `WeightsInitializer` will be created
     * inside the training threads by a `WeightInitializationStrategy`. This ensure that each training thread has its own
     * initializer, so the `get_initial_weight()` method does not need to be mutex-protected.
     *
     * \sa WeightInitializationStrategy
     */
    class WeightsInitializer : public stats::Tracked {
    public:
        virtual ~WeightsInitializer() = default;

        /// Generate an initial vector for the given label. The result should be placed in target.
        virtual void get_initial_weight(label_id_t label_id, Eigen::Ref<DenseRealVector> target,
                                        objective::Objective& objective) = 0;
    };


    /*!
     * \brief Base class for all weight init strategies.
     * \details The `WeightInitializationStrategy` is responsible for generating a `WeightInitializer` for each thread.
     * To that end, the training code calls `WeightInitializationStrategy::make_initializer()` during thread initialization,
     * i.e. the function will be called in the thread that will then also be used to run the initializations.
     *
     * The `WeightInitializationStrategy::make_initializer()` gets passed in a reference to the training features. This is
     * done (as opposed to `WeightInitializationStrategy` saving the features itself) so we can get the numa-local feature
     * copies, and don't have to do the duplication again.
     *
     * \sa WeightsInitializer, create_zero_initializer, create_constant_initializer, create_pretrained_initializer,
     * create_feature_mean_initializer
     */
    class WeightInitializationStrategy {
    public:
        virtual ~WeightInitializationStrategy() = default;

        /*!
         * \brief Creats a new, thread local `WeightsInitializer`.
         * \details This function will be called from the thread in which the returned `WeightsInitializer` will be used.
         * It gets passed in a numa-local copy of the feature matrix.
         * \param features Read-only reference to the numa-local feature matrix.
         * \return A new `WeightsInitializer`.
         */
        [[nodiscard]] virtual std::unique_ptr<WeightsInitializer>
        make_initializer(const std::shared_ptr<const GenericFeatureMatrix>& features) const = 0;
    };

    // constructor functions
    /// Creates an initialization strategy that initializes all weight vectors to zero.
    std::shared_ptr<WeightInitializationStrategy> create_zero_initializer();

    /// Creates an initialization strategy that initializes all weight vectors to the given vector.
    /// TODO allow both dense and sparse vectors.
    std::shared_ptr<WeightInitializationStrategy> create_constant_initializer(DenseRealVector vec);

    /// Creates an initialization strategy that uses an already trained model to set the initial weights.
    std::shared_ptr<WeightInitializationStrategy> create_pretrained_initializer(std::shared_ptr<model::Model> model);

    /// Creates an initialization strategy based on the mean of positive and negative features.
    std::shared_ptr<WeightInitializationStrategy> create_feature_mean_initializer(std::shared_ptr<DatasetBase> data, real_t pos=1, real_t neg=-2);

    /// Creates an initialization strategy based on the mean of positive and negative features.
    std::shared_ptr<WeightInitializationStrategy> create_multi_pos_mean_strategy(std::shared_ptr<DatasetBase> data, int max_pos, real_t pos=1, real_t neg=-2);

    /*!
     * Creates an initialization strategy based on
     * > Huang Fang et al. “Fast training for large-scale one-versus-all linear classi-
     * > fiers using tree-structured initialization”. In: Proceedings of the 2019 SIAM
     * > International Conference on Data Mining.
     *
    */
    std::shared_ptr<WeightInitializationStrategy> create_ova_primal_initializer(
            const std::shared_ptr<DatasetBase>& data, RegularizerSpec regularizer, LossType loss);
}

#endif //DISMEC_INITIALIZER_H

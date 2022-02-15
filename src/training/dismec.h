// Copyright (c) 2021, Aalto University, developed by Erik Schultheis
// All rights reserved.
//
// SPDX-License-Identifier: MIT

#ifndef DISMEC_DISMEC_H
#define DISMEC_DISMEC_H

#include "training.h"
#include "weighting.h"
#include "matrix_types.h"
#include "parallel/numa.h"

namespace init {
    class WeightInitializationStrategy;
}

/*!
 * \brief An implementation of \ref TrainingSpec that models the DiSMEC algorithm.
 * \details The algorithm runs the \ref NewtonWithLineSearch optimizer on a
 * \ref Regularized_SquaredHingeSVC objective. The minimization can be influenced by
 * providing a \ref HyperParameters object that sets e.g. the stopping criterion and
 * number of steps. The squared hinge loss can be influenced by giving a custom
 * \ref WeightingScheme to e.g. have constant weighting or propensity based weighting.
 *
 * The stopping criterion \p epsilon of the \ref NewtonWithLineSearch optimizer is adjusted
 * for the number of positive/negative label instances from the given base value. If `eps`
 * is the value given in `hyper_params` and for a given label id there are `p` and `n` positive
 * and negative instances, then the epsilon used will be
 * \f$\text{epsilon} = \text{eps} \cdot \text{min}(p, n, 1) / (p+n) \f$.
 * \todo Figure out why we do this and put a reference/explanation here.
 */

class DiSMECTraining : public TrainingSpec {
public:
    /*!
     * \brief Creates a DiSMECTraining instance.
     * \param data The dataset on which to train.
     * \param hyper_params Hyper parameters that will be applied to the \ref NewtonWithLineSearch optimizer.
     * \param weighting Positive/Negative label weighting that will be used for the \ref Regularized_SquaredHingeSVC objective.
     */
    DiSMECTraining(std::shared_ptr<const DatasetBase> data, HyperParameters hyper_params,
                   std::shared_ptr<WeightingScheme> weighting,
                   std::shared_ptr<init::WeightInitializationStrategy> init,
                   std::shared_ptr<postproc::PostProcessFactory> post_proc,
                   std::shared_ptr<TrainingStatsGatherer> gatherer,
                   bool use_sparse,
                   RegularizerSpec regularizer, LossType loss);

    [[nodiscard]] std::shared_ptr<objective::Objective> make_objective() const override;
    [[nodiscard]] std::unique_ptr<solvers::Minimizer> make_minimizer() const override;
    [[nodiscard]] std::unique_ptr<init::WeightsInitializer> make_initializer() const override;
    [[nodiscard]] std::shared_ptr<model::Model> make_model(long num_features, model::PartialModelSpec spec) const override;

    void update_minimizer(solvers::Minimizer& base_minimizer, label_id_t label_id) const override;
    void update_objective(objective::Objective& base_objective, label_id_t label_id) const override;

    [[nodiscard]] std::unique_ptr<postproc::PostProcessor> make_post_processor(const std::shared_ptr<objective::Objective>& objective) const override;

    TrainingStatsGatherer& get_statistics_gatherer() override;
private:
    HyperParameters m_NewtonSettings;
    std::shared_ptr<WeightingScheme> m_Weighting;
    bool m_UseSparseModel = false;

    // initial conditions
    std::shared_ptr<init::WeightInitializationStrategy> m_InitStrategy;

    // post processing
    std::shared_ptr<postproc::PostProcessFactory> m_PostProcessor;

    parallel::NUMAReplicator<const GenericFeatureMatrix> m_FeatureReplicator;

    std::shared_ptr<TrainingStatsGatherer> m_StatsGather;

    double m_BaseEpsilon;
    RegularizerSpec m_Regularizer;
    LossType m_Loss;
};


#endif //DISMEC_DISMEC_H

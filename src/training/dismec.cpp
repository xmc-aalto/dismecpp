// Copyright (c) 2021, Aalto University, developed by Erik Schultheis
// All rights reserved.
//
// SPDX-License-Identifier: MIT

#include "dismec.h"
#include "objective/reg_sq_hinge.h"
#include "objective/regularizers_imp.h"
#include "objective/generic_linear.h"
#include "solver/newton.h"
#include "model/model.h"
#include "model/dense.h"
#include "model/sparse.h"
#include "data/data.h"
#include "initializer.h"
#include "weighting.h"
#include "postproc.h"

using namespace dismec;

/*
std::unique_ptr<objective::Objective> make_regularizer(RegularizerSpec spec) {
    switch(reg) {
        case RegularizerType::REG_L2:
            return std::make_unique<objective::SquaredNormRegularizer>(scale, ignore_bias);
        case RegularizerType::REG_L1:
            return std::make_unique<objective::HuberRegularizer>(1e-2, scale, ignore_bias);
        case RegularizerType::REG_L1_RELAXED:
            return std::make_unique<objective::HuberRegularizer>(1e-1, scale, ignore_bias);
        case RegularizerType::REG_HUBER:
            return std::make_unique<objective::HuberRegularizer>(1, scale, ignore_bias);
        case RegularizerType::REG_ELASTIC_50_50:
            return std::make_unique<objective::ElasticNetRegularizer>(1e-1, scale, 0.5, ignore_bias);
        case RegularizerType::REG_ELASTIC_90_10:
            return std::make_unique<objective::ElasticNetRegularizer>(1e-1, scale, 0.9, ignore_bias);
        default:
            throw std::invalid_argument("Unknown regularizer");
    }
}*/

std::shared_ptr<objective::Objective> dismec::make_loss(
        LossType type,
        std::shared_ptr<const GenericFeatureMatrix> X,
        std::unique_ptr<objective::Objective> reg) {
    switch (type) {
        case LossType::SQUARED_HINGE:
            if(X->is_sparse()) {
                return std::make_shared<objective::Regularized_SquaredHingeSVC>(X, std::move(reg));
            } else {
                return make_squared_hinge(X, std::move(reg));
            }
        case LossType::LOGISTIC:
            return make_logistic_loss(X, std::move(reg));
        case LossType::HUBER_HINGE:
            return make_huber_hinge(X, std::move(reg), 1.0);
        case LossType::HINGE:
            return make_huber_hinge(X, std::move(reg), 0.1);
        default:
            THROW_EXCEPTION(std::runtime_error, "Unexpected loss type");
    }
}


std::shared_ptr<objective::Objective> DiSMECTraining::make_objective() const {
    // we make a copy of the features, so they are in the local numa memory
    auto copy = m_FeatureReplicator.get_local();
    auto reg = std::visit([](auto&& config){ return make_regularizer(config); }, m_Regularizer);
    return make_loss(m_Loss, std::move(copy), std::move(reg));
}

std::unique_ptr<solvers::Minimizer> DiSMECTraining::make_minimizer() const {
    auto minimizer = std::make_unique<solvers::NewtonWithLineSearch>(num_features());
    m_NewtonSettings.apply(*minimizer);
    return minimizer;
}

void DiSMECTraining::update_minimizer(solvers::Minimizer& base_minimizer, label_id_t label_id) const
{
    auto* minimizer = dynamic_cast<solvers::NewtonWithLineSearch*>(&base_minimizer);
    if(!minimizer)
        throw std::logic_error("Could not cast minimizer to <NewtonWithLineSearch>");

    // adjust the epsilon parameter according to number of positives/number of negatives
    std::size_t num_pos = get_data().num_positives(label_id);
    double small_count = static_cast<double>(std::min(num_pos, get_data().num_examples() - num_pos));
    double epsilon_scale = std::max(small_count, 1.0) / get_data().num_examples();
    minimizer->set_epsilon(m_BaseEpsilon * epsilon_scale);
}

DiSMECTraining::DiSMECTraining(std::shared_ptr<const DatasetBase> data,
                               HyperParameters hyper_params,
                               std::shared_ptr<WeightingScheme> weighting,
                               std::shared_ptr<init::WeightInitializationStrategy> init,
                               std::shared_ptr<postproc::PostProcessFactory> post_proc,
                               std::shared_ptr<TrainingStatsGatherer> gatherer,
                               bool use_sparse,
                               RegularizerSpec regularizer,
                               LossType loss) :
        TrainingSpec(std::move(data)),
        m_NewtonSettings( std::move(hyper_params) ),
        m_Weighting( std::move(weighting) ),
        m_UseSparseModel( use_sparse ),
        m_InitStrategy( std::move(init) ),
        m_PostProcessor( std::move(post_proc) ),
        m_FeatureReplicator(get_data().get_features() ),
        m_StatsGather( std::move(gatherer) ),
        m_Regularizer( regularizer ),
        m_Loss( loss )
{
    if(!m_InitStrategy) {
        throw std::invalid_argument("Missing weight initialization strategy");
    }

    if(!m_PostProcessor) {
        throw std::invalid_argument("Missing weight post processor");
    }

    // extract the base value of `epsilon` from the `hyper_params` object.
    m_BaseEpsilon = std::get<double>(m_NewtonSettings.get("epsilon"));
}

void DiSMECTraining::update_objective(objective::Objective& base_objective, label_id_t label_id) const {
    auto* objective = dynamic_cast<objective::LinearClassifierBase*>(&base_objective);
    if(!objective)
        throw std::logic_error("Could not cast objective to <LinearClassifierBase>");

    // we need to set the labels before we update the costs, since the label information is needed
    // to determine whether to apply the positive or the negative weighting
    get_data().get_labels(label_id, objective->get_label_ref());
    if(m_Weighting) {
        objective->update_costs(m_Weighting->get_positive_weight(label_id),
                                m_Weighting->get_negative_weight(label_id));
    }
}

std::unique_ptr<init::WeightsInitializer> DiSMECTraining::make_initializer() const {
    return m_InitStrategy->make_initializer(m_FeatureReplicator.get_local());
}

std::shared_ptr<model::Model> DiSMECTraining::make_model(long num_features, model::PartialModelSpec spec) const {
    if(m_UseSparseModel) {
        return std::make_shared<model::SparseModel>(num_features, spec);
    } else {
        return std::make_shared<model::DenseModel>(num_features, spec);
    }
}

std::unique_ptr<postproc::PostProcessor> DiSMECTraining::make_post_processor(const std::shared_ptr<objective::Objective>& objective) const {
    return m_PostProcessor->make_processor(objective);
}

TrainingStatsGatherer& DiSMECTraining::get_statistics_gatherer() {
    return *m_StatsGather;
}


std::shared_ptr<TrainingSpec> dismec::create_dismec_training(std::shared_ptr<const DatasetBase> data,
                                                             HyperParameters params,
                                                             DismecTrainingConfig config) {
    if(!config.Init)
        config.Init = init::create_zero_initializer();
    if(!config.PostProcessing)
        config.PostProcessing = postproc::create_identity();
    return std::make_shared<DiSMECTraining>(std::move(data), std::move(params), std::move(config.Weighting),
                                            std::move(config.Init),
                                            std::move(config.PostProcessing),
                                            std::move(config.StatsGatherer),
                                            config.Sparse,
                                            config.Regularizer,
                                            config.Loss);
}

long TrainingSpec::num_features() const { return get_data().num_features(); }

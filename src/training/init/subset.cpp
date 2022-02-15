// Copyright (c) 2021, Aalto University, developed by Erik Schultheis
// All rights reserved.
//
// SPDX-License-Identifier: MIT

#include "subset.h"
#include "stats/collection.h"
#include "stats/timer.h"
#include "data/types.h"
#include "data/data.h"
#include "data/transform.h"
#include "objective/objective.h"
#include "hash_vector.h"
#include <limits>

using namespace init;

SubsetFeatureMeanInitializer::SubsetFeatureMeanInitializer(
        std::shared_ptr<const DatasetBase> data,
        const DenseRealVector& mean_of_all,
        std::shared_ptr<const GenericFeatureMatrix> local_features,
        real_t pos, real_t neg) :
    m_DataSet(std::move(data)), m_LocalFeatures(std::move(local_features)),
    m_MeanOfAll(DenseRealVector::Zero(1)), m_PosTarget(pos), m_NegTarget(neg)
{
    if(!m_DataSet) {
        throw std::logic_error("dataset is <null>");
    }
    if(!m_LocalFeatures) {
        throw std::logic_error("local features are <null>");
    }

    m_LabelBuffer.resize(m_DataSet->num_examples());
    m_MeanOfAll = mean_of_all;

    m_MeanAllNormSquared = m_MeanOfAll.squaredNorm();

    declare_stat(STAT_DURATION, {"duration", "µs"});
}



std::pair<real_t, real_t> SubsetFeatureMeanInitializer::calculate_factors(
        label_id_t label_id,
        const Eigen::Ref<DenseRealVector>& mean_of_positives)
{

    real_t num_pos = m_DataSet->num_positives(label_id);
    real_t PP = mean_of_positives.squaredNorm();
    real_t PA = mean_of_positives.dot(m_MeanOfAll);
    real_t p = num_pos / m_DataSet->num_examples();

    real_t divide = PA*PA - PP * m_MeanAllNormSquared;
    // TODO spend some more time thinking about numerical stability here
    if(std::abs(PA) < std::numeric_limits<real_t>::epsilon() ) {
        if(std::abs(PA) < std::numeric_limits<real_t>::epsilon() ) {
            return {real_t{0}, real_t{-1.f}};
        }
        return {m_PosTarget / PP, 0};
    }

    // not sure under which situations this may happen, so we're just going with a simple heuristic here
    if(std::abs(divide) < std::numeric_limits<real_t>::epsilon()) {
        spdlog::warn("Cannot use initialization procedure, mean vectors are not linearly independent.");
        return {real_t{0}, real_t{-1.f}};
    }

    // otherwise, do a real calculation
    real_t f = p * (m_PosTarget - m_NegTarget) + m_NegTarget;
    real_t u = (f * PA - m_PosTarget * m_MeanAllNormSquared) / divide;
    real_t v = (m_PosTarget - u * PP) / PA;

    return {u, v};
}

SubsetFeatureMeanStrategy::SubsetFeatureMeanStrategy(std::shared_ptr<const DatasetBase> data, real_t positive_target,
                                               real_t negative_target) :
                                               m_DataSet(std::move(data)),
                                               m_NegativeTarget(negative_target),
                                               m_PositiveTarget(positive_target) {
    if(!m_DataSet) {
        throw std::logic_error("dataset is <null>");
    }
    m_MeanOfAllInstances = get_mean_feature(*m_DataSet->get_features());
}

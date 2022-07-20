// Copyright (c) 2021, Aalto University, developed by Erik Schultheis
// All rights reserved.
//
// SPDX-License-Identifier: MIT

#ifndef DISMEC_SUBSET_H
#define DISMEC_SUBSET_H

#include "training/initializer.h"

namespace dismec::init {
    class SubsetFeatureMeanInitializer : public WeightsInitializer {
    public:
        SubsetFeatureMeanInitializer(std::shared_ptr<const DatasetBase> data,
                                     const DenseRealVector& mean_of_all,
                                     std::shared_ptr<const GenericFeatureMatrix> local_features, real_t pos, real_t neg);

    protected:
        std::shared_ptr<const DatasetBase> m_DataSet;
        std::shared_ptr<const GenericFeatureMatrix> m_LocalFeatures;

        BinaryLabelVector m_LabelBuffer;
        DenseRealVector m_MeanOfAll;
        real_t m_MeanAllNormSquared;

        real_t m_PosTarget;
        real_t m_NegTarget;

        static constexpr stats::stat_id_t STAT_DURATION{0};

        std::pair<real_t, real_t> calculate_factors(
                label_id_t label_id,
                const Eigen::Ref<DenseRealVector>& mean_of_positives);
    };

    class SubsetFeatureMeanStrategy : public WeightInitializationStrategy {
    public:
        SubsetFeatureMeanStrategy(std::shared_ptr<const DatasetBase> data, real_t negative_target,
                                  real_t positive_target);

    protected:
        std::shared_ptr<const DatasetBase> m_DataSet;
        DenseRealVector m_MeanOfAllInstances;
        real_t m_NegativeTarget;
        real_t m_PositiveTarget;
    };
}

#endif //DISMEC_SUBSET_H

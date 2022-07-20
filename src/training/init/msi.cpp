// Copyright (c) 2021, Aalto University, developed by Erik Schultheis
// All rights reserved.
//
// SPDX-License-Identifier: MIT

#include "subset.h"
#include "stats/collection.h"
#include "stats/timer.h"
#include "data/types.h"
#include "data/data.h"
#include "objective/objective.h"
#include "utils/hash_vector.h"
#include <limits>


using namespace dismec::init;

namespace dismec::init {
    class MeanOfFeaturesInitializer : public SubsetFeatureMeanInitializer {
    public:
        MeanOfFeaturesInitializer(std::shared_ptr<const DatasetBase> data,
                                  const DenseRealVector& mean_of_all,
                                  std::shared_ptr<const GenericFeatureMatrix> local_features,
                                  real_t pos, real_t neg);

        void get_initial_weight(label_id_t label_id, Eigen::Ref<DenseRealVector> target, objective::Objective& objective) override;
    private:
        static constexpr stats::stat_id_t STAT_POSITIVE_FACTOR{1};
        static constexpr stats::stat_id_t STAT_ALL_MEAN_FACTOR{2};
        static constexpr stats::stat_id_t STAT_NUM_POS{3};
        static constexpr stats::stat_id_t STAT_LOSS_REDUCTION{4};
    };

    class MeanOfFeaturesStrategy : public SubsetFeatureMeanStrategy {
    public:
        using SubsetFeatureMeanStrategy::SubsetFeatureMeanStrategy;

        [[nodiscard]] std::unique_ptr<WeightsInitializer>
        make_initializer(const std::shared_ptr<const GenericFeatureMatrix>& features) const override;
    };

}

MeanOfFeaturesInitializer::MeanOfFeaturesInitializer(std::shared_ptr<const DatasetBase> data,
                                                     const DenseRealVector& mean_of_all,
                                                     std::shared_ptr<const GenericFeatureMatrix> local_features,
                                                     real_t pos, real_t neg) :
        SubsetFeatureMeanInitializer(std::move(data), mean_of_all, std::move(local_features), pos, neg)
{
    declare_stat(STAT_POSITIVE_FACTOR, {"positive", {}});
    declare_stat(STAT_ALL_MEAN_FACTOR, {"all_mean", {}});
    declare_stat(STAT_NUM_POS, {"num_pos", "#positives"});
    declare_stat(STAT_LOSS_REDUCTION, {"loss_reduction", "(f(0)-f(w))/f(0) [%]"});
}

void MeanOfFeaturesInitializer::get_initial_weight(label_id_t label_id, Eigen::Ref<DenseRealVector> target, objective::Objective& objective)
{
    auto timer = make_timer(STAT_DURATION);
    m_DataSet->get_labels(label_id, m_LabelBuffer);

    target.setZero();
    int num_pos = m_DataSet->num_positives(label_id);
    visit([&](const auto& matrix) {
        // I've put the entire loop into the visit so that the sparse/dense dispatch happens only once
        for(int i = 0; i < m_LabelBuffer.size(); ++i) {
            if(m_LabelBuffer.coeff(i) > 0.0) {
                target += matrix.row(i) / (real_t)num_pos;
            }
        }
    }, *m_LocalFeatures);

    auto [p, a] = calculate_factors(label_id, target);
    target = target * p + m_MeanOfAll * a;

    record(STAT_POSITIVE_FACTOR, p);
    record(STAT_ALL_MEAN_FACTOR, a);
    record(STAT_NUM_POS, num_pos);
    record(STAT_LOSS_REDUCTION, [&]() {
        HashVector temp{target};
        real_t obj_at_new = objective.value(temp);
        temp.modify().setZero();
        real_t obj_at_zero = objective.value(temp);
        return 100.f * (obj_at_zero - obj_at_new) / obj_at_zero;
    });
}

std::unique_ptr<WeightsInitializer> MeanOfFeaturesStrategy::make_initializer(const std::shared_ptr<const GenericFeatureMatrix>& features) const {
    return std::make_unique<MeanOfFeaturesInitializer>(m_DataSet, m_MeanOfAllInstances, features, m_PositiveTarget, m_NegativeTarget);
}

std::shared_ptr<WeightInitializationStrategy> dismec::init::create_feature_mean_initializer(std::shared_ptr<DatasetBase> data, real_t pos, real_t neg) {
    return std::make_shared<MeanOfFeaturesStrategy>(std::move(data), pos, neg);
}

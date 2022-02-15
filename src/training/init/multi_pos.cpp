// Copyright (c) 2021, Aalto University, developed by Erik Schultheis
// All rights reserved.
//
// SPDX-License-Identifier: MIT

#include "subset.h"
#include "hash_vector.h"
#include "stats/collection.h"
#include "stats/timer.h"
#include "data/data.h"
#include "objective/objective.h"

using namespace init;


namespace init {
    template<bool Sparse>
    struct TypeLookup;

    template<>
    struct TypeLookup<false> {
        using MatrixType = DenseFeatures;
        using VectorType = DenseRealVector;
    };

    template<>
    struct TypeLookup<true> {
        using MatrixType = SparseFeatures;
        using VectorType = SparseRealVector;
    };

    template<bool Sparse>
    class MultiPosMeanInitializer : public SubsetFeatureMeanInitializer {
        using MatrixType = typename TypeLookup<Sparse>::MatrixType;
        using VectorType = typename TypeLookup<Sparse>::VectorType;

    public:
        MultiPosMeanInitializer(std::shared_ptr<const DatasetBase> data,
                                const DenseRealVector& mean_of_all,
                                std::shared_ptr<const GenericFeatureMatrix> local_features,
                                int max_pos, real_t pos, real_t neg);

        void get_initial_weight(label_id_t label_id, Eigen::Ref<DenseRealVector> target,
                                objective::Objective& objective) override;

    private:
        std::vector<VectorType> m_PositiveInstances;
        int m_MaxPos;
        types::DenseRowMajor<real_t> m_GramMatrix;
        DenseRealVector m_Target;
        DenseRealVector m_AlphaVector;
        Eigen::LLT<types::DenseRowMajor<real_t>> m_LLT;

        real_t m_Lambda = 0.01;

        void extract_sub_dataset(label_id_t label_id);

        stats::stat_id_t STAT_NUM_POS{1};
        stats::stat_id_t STAT_LOSS_REDUCTION{2};
    };

    class MultiPosMeanStrategy : public SubsetFeatureMeanStrategy {
    public:
        MultiPosMeanStrategy(std::shared_ptr<const DatasetBase> data, real_t negative_target, real_t positive_target,
                             int max_positives) :
                SubsetFeatureMeanStrategy(std::move(data), negative_target, positive_target),
                m_MaxPositives(max_positives)
        {
        }

        [[nodiscard]] std::unique_ptr<WeightsInitializer>
        make_initializer(const std::shared_ptr<const GenericFeatureMatrix>& features) const override;

    private:
        int m_MaxPositives;
    };
}

template<bool b>
MultiPosMeanInitializer<b>::MultiPosMeanInitializer(std::shared_ptr<const DatasetBase> data,
                                                 const DenseRealVector& mean_of_all,
                                                 std::shared_ptr<const GenericFeatureMatrix> local_features, int max_pos,
                                                 real_t pos, real_t neg):
        SubsetFeatureMeanInitializer(std::move(data), mean_of_all, std::move(local_features), pos, neg),
        m_MaxPos(max_pos), m_LLT(max_pos + 1)
{
    m_PositiveInstances.resize(m_MaxPos);

    declare_stat(STAT_NUM_POS, {"num_pos", "#positives"});
    declare_stat(STAT_LOSS_REDUCTION, {"loss_reduction", "(f(0)-f(w))/f(0) [%]"});
}

template<bool Sparse>
void MultiPosMeanInitializer<Sparse>::get_initial_weight(
        label_id_t label_id,
        Eigen::Ref<DenseRealVector> target,
        objective::Objective& objective) {
    auto timer = make_timer(STAT_DURATION);
    m_DataSet->get_labels(label_id, m_LabelBuffer);

    int num_pos = m_DataSet->num_positives(label_id);
    if(num_pos > m_MaxPos) {
        // this code is just copied from avg_of_pos
        target.setZero();
        for(int i = 0; i < m_LabelBuffer.size(); ++i) {
            if(m_LabelBuffer.coeff(i) > 0.0) {
                target += m_LocalFeatures->get<MatrixType>().row(i) / (real_t)num_pos;
            }
        }

        auto [p, a] = calculate_factors(label_id, target);
        target = target * p + m_MeanOfAll * a;
    } else {
        real_t num_samples = m_DataSet->num_examples();
        extract_sub_dataset(label_id);

        // at this point, m_Averages is prepared and we can start calculating the Gram matrix
        m_GramMatrix.resize(num_pos + 1, num_pos + 1);
        m_Target.resize(num_pos + 1);
        m_Target.coeffRef(0) = m_NegTarget;
        for (int i = 1; i < num_pos + 1; ++i) {
            m_Target.coeffRef(i) = m_PosTarget;
        }

        // the negatives are a bit tricky
        // <N, N> = <X, X> - 2 <X, Ai> + <Ai, Ai>
        m_GramMatrix.coeffRef(0, 0) = m_MeanAllNormSquared;

        // fill in the part of the gram matrix that is built by the positives
        for (int i = 0; i < num_pos; ++i) {
            for (int j = i; j < num_pos; ++j) {
                auto& a = m_PositiveInstances[i];
                auto& b = m_PositiveInstances[j];
                real_t dot = a.dot(b);
                m_GramMatrix.coeffRef(i + 1, j + 1) = dot;
                m_GramMatrix.coeffRef(j + 1, i + 1) = dot;
            }

            // adjustments for the negatives
            m_GramMatrix.coeffRef(0, 0) += m_GramMatrix.coeffRef(i + 1, i + 1) / num_samples / num_samples;
            real_t xTa = m_PositiveInstances[i].dot(m_MeanOfAll);
            m_GramMatrix.coeffRef(0, i+1) = xTa;
            m_GramMatrix.coeffRef(0, 0) -= 2*xTa / num_samples;
        }

        // fix up the <N, Aj> elements
        // <N, Aj> = <X, Aj> - sum <Ai, Aj>/n
        for (int i = 0; i < num_pos; ++i) {
            for (int j = 0; j < num_pos; ++j) {
                m_GramMatrix.coeffRef(0, i + 1) -= m_GramMatrix.coeffRef(j, i + 1) / num_samples;
            }
            m_GramMatrix.coeffRef(i + 1, 0) = m_GramMatrix.coeff(0, i + 1);

            // also put in the regularizer
            m_GramMatrix.coeffRef(i + 1, i + 1) += m_Lambda;
        }
        m_GramMatrix.coeffRef(0, 0) += m_Lambda;

        m_LLT.compute(m_GramMatrix);
        m_AlphaVector = m_LLT.solve(m_Target);

        // reconstruct the initial vector
        target = m_AlphaVector[0] * m_MeanOfAll;
        for (int i = 1; i < num_pos + 1; ++i) {
            target += (m_AlphaVector[i] - m_AlphaVector[0] / num_samples) * m_PositiveInstances[i - 1];
        }
    }

    record(STAT_NUM_POS, [&]() -> long { return m_DataSet->num_positives(label_id); });
    record(STAT_LOSS_REDUCTION, [&]() {
        HashVector temp{target};
        real_t obj_at_new = objective.value(temp);
        temp.modify().setZero();
        real_t obj_at_zero = objective.value(temp);
        return 100.f * (obj_at_zero - obj_at_new) / obj_at_zero;
    });
}

template<bool Sparse>
void MultiPosMeanInitializer<Sparse>::extract_sub_dataset(label_id_t label_id) {
    assert( m_DataSet->num_positives(label_id) <= m_MaxPos);

    m_DataSet->get_labels(label_id, m_LabelBuffer);

    int pos_count = 0;
    for(int i = 0; i < m_LabelBuffer.size(); ++i) {
        if(m_LabelBuffer.coeff(i) <= 0.0) {
            continue;
        }

        m_PositiveInstances[pos_count] = m_LocalFeatures->get<MatrixType>().row(i);

        ++pos_count;
    }
}

std::unique_ptr<WeightsInitializer>
MultiPosMeanStrategy::make_initializer(const std::shared_ptr<const GenericFeatureMatrix>& features) const {
    if(features->is_sparse()) {
        return std::make_unique<MultiPosMeanInitializer<true>>(
                m_DataSet, m_MeanOfAllInstances, features, m_MaxPositives, m_PositiveTarget, m_NegativeTarget);
    } else {
        return std::make_unique<MultiPosMeanInitializer<false>>(
                m_DataSet, m_MeanOfAllInstances, features, m_MaxPositives, m_PositiveTarget, m_NegativeTarget);

    }


}


std::shared_ptr<WeightInitializationStrategy> init::create_multi_pos_mean_strategy(std::shared_ptr<DatasetBase> data, int max_pos, real_t pos, real_t neg) {
    return std::make_shared<MultiPosMeanStrategy>(std::move(data), pos, neg, max_pos);
}
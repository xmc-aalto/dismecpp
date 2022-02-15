// Copyright (c) 2021, Aalto University, developed by Erik Schultheis
// All rights reserved.
//
// SPDX-License-Identifier: MIT

#ifndef DISMEC_PREDICTION_H
#define DISMEC_PREDICTION_H

#include "parallel/task.h"
#include "parallel/numa.h"
#include "matrix_types.h"
#include <memory>

class DatasetBase;
namespace model {
    class Model;
}

using model::Model;

class PredictionBase : public parallel::TaskGenerator {
public:
    PredictionBase(const DatasetBase* data, std::shared_ptr<const Model> model);

protected:
    const DatasetBase* m_Data;
    std::shared_ptr<const Model> m_Model;

    parallel::NUMAReplicator<const GenericFeatureMatrix> m_FeatureReplicator;
    std::vector<std::shared_ptr<const GenericFeatureMatrix>> m_ThreadLocalFeatures;

    void make_thread_local_features(int num_threads);

    void init_thread(thread_id_t thread_id) override;

    void do_prediction(long begin, long end, thread_id_t thread_id, Eigen::Ref<PredictionMatrix> target);
};

class PredictionTaskGenerator : public PredictionBase {
public:

    PredictionTaskGenerator(const DatasetBase* data, std::shared_ptr<const Model> model);

    void run_tasks(long begin, long end, thread_id_t thread_id) override;
    [[nodiscard]] long num_tasks() const override;

    [[nodiscard]] const PredictionMatrix& get_predictions() const { return m_Predictions; }
private:
    PredictionMatrix m_Predictions;
};

class TopKPredictionTaskGenerator : public PredictionBase {
public:
    TopKPredictionTaskGenerator(const DatasetBase* data, std::shared_ptr<const Model> model, long K);

    void update_model(std::shared_ptr<const Model> model);

    void run_tasks(long begin, long end, thread_id_t thread_id) override;

    [[nodiscard]] long num_tasks() const override;
    void prepare(long num_threads, long chunk_size) override;
    void finalize() override;

    [[nodiscard]] const PredictionMatrix& get_top_k_values() const { return m_TopKValues; }
    [[nodiscard]] const IndexMatrix& get_top_k_indices() const { return m_TopKIndices; }

    [[nodiscard]] const std::array<std::int64_t, 4>& get_confusion_matrix() const { return m_ConfusionMatrix; }

    static constexpr const int TRUE_POSITIVES  = 0;
    static constexpr const int FALSE_POSITIVES = 1;
    static constexpr const int TRUE_NEGATIVES  = 2;
    static constexpr const int FALSE_NEGATIVES = 3;
private:
    long m_K;

    PredictionMatrix m_TopKValues;
    IndexMatrix m_TopKIndices;

    std::vector<PredictionMatrix> m_ThreadLocalPredictionCache;
    std::vector<PredictionMatrix> m_ThreadLocalTopKValues;
    std::vector<IndexMatrix> m_ThreadLocalTopKIndices;
    std::vector<std::array<std::int64_t, 4>> m_ThreadLocalConfusionMatrix;

    std::vector<std::vector<long>> m_GroundTruth;
    std::array<std::int64_t, 4> m_ConfusionMatrix;
};
#endif //DISMEC_PREDICTION_H

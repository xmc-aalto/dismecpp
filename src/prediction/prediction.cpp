// Copyright (c) 2021, Aalto University, developed by Erik Schultheis
// All rights reserved.
//
// SPDX-License-Identifier: MIT

#include "prediction.h"

#include <utility>
#include "data/data.h"
#include "spdlog/spdlog.h"
#include "spdlog/fmt/fmt.h"
#include "model/model.h"

PredictionBase::PredictionBase(const DatasetBase* data,
                               std::shared_ptr<const Model> model) :
        m_Data(data), m_Model(std::move(model)), m_FeatureReplicator(m_Data->get_features())
{
    if(m_Model->num_labels() != data->num_labels()) {
        throw std::invalid_argument(
                fmt::format("Mismatched number of labels between model ({}) and data ({})",
                            m_Model->num_labels(), data->num_labels()));
    }

    if(m_Model->num_features() != data->num_features()) {
        throw std::invalid_argument(
                fmt::format("Mismatched number of features between model ({}) and data ({})",
                            m_Model->num_features(), data->num_features()));
    }
}

void PredictionBase::make_thread_local_features(int num_threads) {
    m_ThreadLocalFeatures.resize(num_threads);
}

void PredictionBase::init_thread(parallel::thread_id_t thread_id) {
    m_ThreadLocalFeatures.at(thread_id.to_index()) = m_FeatureReplicator.get_local();
}

namespace {
    Model::FeatureMatrixIn make_matrix(const DenseFeatures& features, long begin, long end) {
        return Model::FeatureMatrixIn::DenseRowMajorRef{features.middleRows(begin, end-begin)};
    }
    Model::FeatureMatrixIn make_matrix(const SparseFeatures& features, long begin, long end) {
        return Model::FeatureMatrixIn::SparseRowMajorRef{features.middleRows(begin, end-begin)};
    }
}

void PredictionBase::do_prediction(long begin, long end, thread_id_t thread_id, Eigen::Ref<PredictionMatrix> target) {
    auto& local_features = m_ThreadLocalFeatures.at(thread_id.to_index());
    visit([&](const auto& features){
              m_Model->predict_scores(make_matrix(features, begin, end), target);
        }, *local_features);

}

PredictionTaskGenerator::PredictionTaskGenerator(const DatasetBase* data, std::shared_ptr<const Model> model) :
    PredictionBase(data, std::move(model))
{
    m_Predictions.resize(data->num_examples(), data->num_labels());
}

long PredictionTaskGenerator::num_tasks() const
{
    return m_Data->num_examples();
}

void PredictionTaskGenerator::run_tasks(long begin, long end, thread_id_t thread_id)
{
    do_prediction(begin, end, thread_id, m_Predictions.middleRows(begin, end));
}

TopKPredictionTaskGenerator::TopKPredictionTaskGenerator(const DatasetBase* data, std::shared_ptr<const Model> model, long K) :
        PredictionBase(data, std::move(model)), m_K(K)
{
    m_TopKValues.resize(data->num_examples(), m_K);
    m_TopKIndices.resize(data->num_examples(), m_K);
    m_TopKValues.setConstant(-std::numeric_limits<real_t>::infinity());

    // generate a transpose of the label matrix
    std::vector<std::vector<long>> examples_to_labels(data->num_examples());
    for(label_id_t label{0}; label.to_index() < data->num_labels(); ++label) {
        for(auto example : dynamic_cast<const MultiLabelData*>(data)->get_label_instances(label)) {
            examples_to_labels[example].push_back(label.to_index());
        }
    }

    m_GroundTruth = std::move(examples_to_labels);
    m_ConfusionMatrix.fill(std::int64_t{0});
}

long TopKPredictionTaskGenerator::num_tasks() const {
    return m_Data->num_examples();
};

void TopKPredictionTaskGenerator::prepare(long num_threads, long chunk_size) {
    m_ThreadLocalPredictionCache.resize(num_threads);
    for(auto& cache : m_ThreadLocalPredictionCache) {
        cache.resize(chunk_size, m_Model->num_weights());
    }
    m_ThreadLocalTopKIndices.resize(num_threads);
    for(auto& cache : m_ThreadLocalTopKIndices) {
        cache.resize(chunk_size, m_K);
    }
    m_ThreadLocalTopKValues.resize(num_threads);
    for(auto& cache : m_ThreadLocalTopKValues) {
        cache.resize(chunk_size, m_K);
    }
    make_thread_local_features(num_threads);

    m_ThreadLocalConfusionMatrix.resize(num_threads);
    for(auto& cache : m_ThreadLocalConfusionMatrix) {
        cache.fill(0);
    }
}

void TopKPredictionTaskGenerator::finalize() {
    m_ThreadLocalPredictionCache.clear();
    for(auto& tl_cm: m_ThreadLocalConfusionMatrix) {
        for(int i = 0; i < 4; ++i) {
            m_ConfusionMatrix[i] += tl_cm[i];
        }
    }
}

void TopKPredictionTaskGenerator::run_tasks(long begin, long end, thread_id_t thread_id) {
    auto& prediction_matrix = m_ThreadLocalPredictionCache.at(thread_id.to_index());
    auto& topk_vals = m_ThreadLocalTopKValues.at(thread_id.to_index());
    auto& topk_idx = m_ThreadLocalTopKIndices.at(thread_id.to_index());
    auto& cm = m_ThreadLocalConfusionMatrix.at(thread_id.to_index());

    // quick access to the label indices that are currently active
    long index_offset = m_Model->labels_begin().to_index();
    long last_index = m_Model->labels_end().to_index();

    // load from global buffer, in case we do a reduction
    topk_idx = m_TopKIndices.middleRows(begin, end-begin);
    topk_vals = m_TopKValues.middleRows(begin, end-begin);

    // generate raw predictions in prediction_matrix
    do_prediction(begin, end, thread_id, prediction_matrix.middleRows(0, end-begin));

    // confusion matrix
    std::int64_t true_positives = 0;
    std::int64_t num_gt_positives = 0;
    for(long sample = begin; sample < end; ++sample) {
        // iterate over all true values
        for(auto& gt : m_GroundTruth[sample])
        {
            // we have to take into account that we are potentially only looking at a subset of the labels.
            if(gt < index_offset) continue;
            if(gt >= last_index) break;

            // correctly predicted true label
            if(prediction_matrix.coeff(sample - begin, gt - index_offset) > 0) {
                ++true_positives;
            }
            ++num_gt_positives;
        }
    }

    std::int64_t positive_prediction = 0;
    for(long t = 0; t < end - begin; ++t) {
        double threshold = topk_vals.coeff(t, m_TopKValues.cols() - 1);

        // reduce to top k
        for(long j = 0; j < prediction_matrix.cols(); ++j)
        {
            real_t value = prediction_matrix.coeff(t, j);
            if(value > 0)   ++positive_prediction;
            if(value < threshold) {
                continue;
            }

            long index = index_offset + j;
            for(long k = 0; k < m_K; ++k) {
                // search for the first entry where we are larger. Once we've inserted this value,
                // move the other values to the right.
                if(value > topk_vals.coeff(t, k)) {
                    value = std::exchange(topk_vals.coeffRef(t, k), value);
                    index = std::exchange(topk_idx.coeffRef(t, k), index);
                }
            }

            // update the threshold: this is the value in the last column
            threshold = topk_vals.coeff(t, topk_vals.cols() - 1);
        }
    }

    std::int64_t total = (end - begin) * prediction_matrix.cols();
    std::int64_t true_neg = total - positive_prediction - num_gt_positives + true_positives;

    cm[TRUE_POSITIVES]  += true_positives;
    cm[FALSE_NEGATIVES] += num_gt_positives - true_positives;
    cm[FALSE_POSITIVES] += positive_prediction - true_positives;
    cm[TRUE_NEGATIVES]  += true_neg;

    // copy to global buffer
    m_TopKIndices.middleRows(begin, end-begin) = topk_idx;
    m_TopKValues.middleRows(begin, end-begin) = topk_vals;
}

void TopKPredictionTaskGenerator::update_model(std::shared_ptr<const Model> model) {
    m_Model = std::move(model);
}

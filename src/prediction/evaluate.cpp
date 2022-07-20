// Copyright (c) 2021, Aalto University, developed by Erik Schultheis
// All rights reserved.
//
// SPDX-License-Identifier: MIT

#include "evaluate.h"
#include "metrics.h"
#include "spdlog/fmt/fmt.h"

using namespace dismec::prediction;

EvaluateMetrics::EvaluateMetrics(const LabelList* sparse_labels, const IndexMatrix* sparse_predictions, long num_labels) :
    m_Labels(sparse_labels), m_Predictions(sparse_predictions), m_NumLabels(num_labels) {
    if(m_Predictions->rows() != m_Labels->size()) {
        throw std::invalid_argument("number of predictions does not match number of labels");
    }

    m_Collectors.resize(1);
}

EvaluateMetrics::~EvaluateMetrics() = default;

void EvaluateMetrics::process_prediction(const std::vector<label_id_t>& raw_labels, const prediction_t& raw_prediction,
                        std::vector<sTrueLabelInfo>& proc_labels, std::vector<sPredLabelInfo>& proc_pred) {
    proc_pred.clear();
    proc_labels.reserve(raw_labels.size());
    std::transform(begin(raw_labels), end(raw_labels), std::back_inserter(proc_labels),
                   [](label_id_t label) {
                       return sTrueLabelInfo{label, -1};
                   });

    // figure out which predictions are correct and which are wrong
    for(long j = 0; j < raw_prediction.size(); ++j) {
        // figure out if this prediction is in the true labels
        auto lookup = std::lower_bound(begin(raw_labels), end(raw_labels), label_id_t{raw_prediction.coeff(j)});
        bool is_correct = false;
        if(lookup != end(raw_labels)) {
            // if so, mark it as correct and register its rank
            is_correct = (*lookup) == label_id_t{raw_prediction.coeff(j)};
            proc_labels[std::distance(begin(raw_labels), lookup)].Rank = j;
        }
        proc_pred.push_back(sPredLabelInfo{label_id_t{raw_prediction.coeff(j)}, is_correct});
    }
}

void EvaluateMetrics::run_task(long task_id, thread_id_t thread_id) {
    auto  prediction = m_Predictions->row(task_id);
    auto& labels     = (*m_Labels)[task_id];

    auto& predicted_cache = m_ThreadLocalPredictedLabels[thread_id.to_index()];
    auto& true_cache = m_ThreadLocalTrueLabels[thread_id.to_index()];

    process_prediction(labels, prediction, true_cache, predicted_cache);

    for(auto& c : m_Collectors[thread_id.to_index()]) {
        c->update(predicted_cache, true_cache);
    }
}

void EvaluateMetrics::run_tasks(long begin, long end, thread_id_t thread_id) {
    for(long t = begin; t < end; ++t) {
        run_task(t, thread_id);
    }
}

long EvaluateMetrics::num_tasks() const {
    return m_Predictions->rows();
}

void EvaluateMetrics::add_precision_at_k(long k, std::string name) {
    if(k > m_Predictions->cols()) {
        throw std::invalid_argument("Cannot calculate top-k precision for k > #predictions");
    }

    if(name.empty()) {
        name = fmt::format("InstanceP@{}", k);
    }

    auto collector = std::make_unique<InstanceRankedPositives>(m_NumLabels, k);
    m_Metrics.push_back( std::make_unique<InstanceWiseMetricReporter>(name, collector.get()) );
    m_Collectors[0].push_back(std::move(collector));
}

void EvaluateMetrics::add_dcg_at_k(long k, bool normalize, std::string name) {
    if(k > m_Predictions->cols()) {
        throw std::invalid_argument("Cannot calculate top-k DCG for k > #predictions");
    }

    if(name.empty()) {
        name = fmt::format("Instance{}DCG@{}", normalize ? "n" : "", k);
    }

    std::vector<double> weights(k);
    for(int i = 0; i < k; ++i) {
        // definition of dcg assumes 1-based indexing; 2 + i ensures finite values
        weights[i] = 1.0 / std::log(2 + i);
    }

    auto collector = std::make_unique<InstanceRankedPositives>(m_NumLabels, k, normalize, std::move(weights));
    m_Metrics.push_back( std::make_unique<InstanceWiseMetricReporter>(name, collector.get()) );
    m_Collectors[0].push_back(std::move(collector));
}


void EvaluateMetrics::add_abandonment_at_k(long k, std::string name) {
    if(k > m_Predictions->cols()) {
        throw std::invalid_argument("Cannot calculate top-k abandonment for k > #predictions");
    }

    if(name.empty()) {
        name = fmt::format("Abd@{}", k);
    }

    auto collector = std::make_unique<AbandonmentAtK>(m_NumLabels, k);
    m_Metrics.push_back( std::make_unique<InstanceWiseMetricReporter>(name, collector.get()) );
    m_Collectors[0].push_back(std::move(collector));
}

MacroMetricReporter* EvaluateMetrics::add_macro_at_k(long k) {
    if(k > m_Predictions->cols()) {
        throw std::invalid_argument("Cannot calculate top-k abandonment for k > #predictions");
    }

    auto collector = std::make_unique<ConfusionMatrixRecorder>(m_NumLabels, k);
    auto metrics = std::make_unique<MacroMetricReporter>(collector.get());
    auto result = metrics.get();
    m_Metrics.push_back( std::move(metrics) );
    m_Collectors[0].push_back(std::move(collector));
    return result;
}


std::vector<std::pair<std::string, double>> EvaluateMetrics::get_metrics() const {
    std::vector<std::pair<std::string, double>> results;
    for(auto& m : m_Metrics)  {
        auto result = m->get_values();
        std::copy(begin(result), end(result), std::back_inserter(results));
    }
    return results;
}

void EvaluateMetrics::prepare(long num_threads, long chunk_size) {
    m_ThreadLocalPredictedLabels.resize(num_threads);
    m_ThreadLocalTrueLabels.resize(num_threads);
    m_Collectors.resize(num_threads);
}

void EvaluateMetrics::finalize() {
    for(int i = 1; i < m_Collectors.size(); ++i) {
        for(int j = 0; j < m_Collectors[0].size(); ++j) {
            m_Collectors[0][j]->reduce(*m_Collectors[i][j]);
        }
    }
}

void EvaluateMetrics::init_thread(parallel::thread_id_t thread_id) {
    /*for(auto& collector : m_Collectors.front()) {
        m_Collectors[thread_id.to_index()].push_back(collector->clone());
    }*/
    if(thread_id.to_index() == 0) return;

    std::transform(begin(m_Collectors.front()), end(m_Collectors.front()),
                   std::back_inserter(m_Collectors[thread_id.to_index()]),
                   [](const auto& other){
        return other->clone();
    });

}

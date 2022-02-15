// Copyright (c) 2021, Aalto University, developed by Erik Schultheis
// All rights reserved.
//
// SPDX-License-Identifier: MIT

#include "metrics.h"
#include "spdlog/fmt/fmt.h"
#include <atomic>

CalculateMetrics::CalculateMetrics(const LabelList* sparse_labels, const IndexMatrix* sparse_predictions) :
        m_Labels(sparse_labels), m_Predictions(sparse_predictions) {
    if(m_Predictions->rows() != m_Labels->size()) {
        throw std::invalid_argument("number of predictions does not match number of labels");
    }
}

CalculateMetrics::~CalculateMetrics() = default;

class MetricBase {
public:
    using prediction_t =  Eigen::Ref<const Eigen::Matrix<long, 1, Eigen::Dynamic>>;
    using labels_t = std::vector<long>;

    virtual ~MetricBase() = default;
    virtual void update(const prediction_t& prediction, const labels_t& labels) = 0;
    [[nodiscard]] virtual double value() const = 0;
};

void CalculateMetrics::run_task(long task_id, thread_id_t thread_id) {
    auto  prediction = m_Predictions->row(task_id);
    auto& labels     = (*m_Labels)[task_id];
    for(auto& m : m_Metrics) {
        m.second->update(prediction, labels);
    }
}

void CalculateMetrics::run_tasks(long begin, long end, thread_id_t thread_id) {
    for(long t = begin; t < end; ++t) {
        run_task(t, thread_id);
    }
}

long CalculateMetrics::num_tasks() const {
    return m_Predictions->rows();
}


class P_At_K : public MetricBase {
public:
    explicit P_At_K(long k) : m_K(k) {}

    void update(const prediction_t& prediction, const labels_t& labels) override {
        for(long j = 0; j < m_K; ++j) {
            // we assume the labels are ordered, but predictions may be in arbitrary order
            if(std::binary_search(begin(labels), end(labels), prediction.coeff(j))) {
                m_Correct += 1;
            }
        }
        m_Total += m_K;
    }

    [[nodiscard]] double value() const override {
        if(m_Correct == 0) return 0.0;
        return static_cast<double>(m_Correct) / static_cast<double>(m_Total);
    }
private:
    long m_K;
    std::atomic<long> m_Correct = 0;
    std::atomic<long> m_Total   = 0;
};

void CalculateMetrics::add_p_at_k(long k, std::string name) {
    if(k > m_Predictions->cols()) {
        throw std::invalid_argument("Cannot calculate top_k precision for k > #predictions");
    }

    if(name.empty()) {
        name = fmt::format("P@{}", k);
    }

    m_Metrics.insert( std::make_pair(std::move(name), std::make_unique<P_At_K>(k) ));
}

std::unordered_map<std::string, double> CalculateMetrics::get_metrics() const {
    std::unordered_map<std::string, double> result;
    for(auto& metric : m_Metrics) {
        result[metric.first] = metric.second->value();
    }
    return result;
}

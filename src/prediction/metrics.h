// Copyright (c) 2021, Aalto University, developed by Erik Schultheis
// All rights reserved.
//
// SPDX-License-Identifier: MIT

#ifndef DISMEC_POSTPROCESSING_H
#define DISMEC_POSTPROCESSING_H

#include "parallel/task.h"
#include "matrix_types.h"
#include <memory>

class MetricBase;

class CalculateMetrics : public parallel::TaskGenerator {
public:
    using LabelList = std::vector<std::vector<long>>;

    CalculateMetrics(const LabelList* sparse_labels, const IndexMatrix* sparse_predictions);
    ~CalculateMetrics() override;

    void add_p_at_k(long k, std::string name={});

    std::unordered_map<std::string, double> get_metrics() const;

    void run_task(long task_id, thread_id_t thread_id);
    void run_tasks(long begin, long end, thread_id_t thread_id) override;
    [[nodiscard]] long num_tasks() const override;


private:
    const LabelList* m_Labels;
    const IndexMatrix* m_Predictions;

    std::unordered_map<std::string, std::unique_ptr<MetricBase>> m_Metrics;
};

#endif //DISMEC_POSTPROCESSING_H

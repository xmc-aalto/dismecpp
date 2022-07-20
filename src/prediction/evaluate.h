// Copyright (c) 2021, Aalto University, developed by Erik Schultheis
// All rights reserved.
//
// SPDX-License-Identifier: MIT

#ifndef DISMEC_SRC_PREDICTION_EVALUATE_H
#define DISMEC_SRC_PREDICTION_EVALUATE_H

#include "parallel/task.h"
#include "matrix_types.h"
#include "data/types.h"
#include <memory>

namespace dismec::prediction {

    struct sTrueLabelInfo {
        label_id_t Label;
        long Rank;
    };
    struct sPredLabelInfo {
        label_id_t Label;
        bool Correct;
    };

    class MetricCollectionInterface;
    class MetricReportInterface;
    class MacroMetricReporter;

    /*!
     * \brief This `TaskGenerator` enables the calculation of evaluation metrics on top-k style sparse predictions.
     * \details In order to enable efficient calculation of multiple metrics form a given instance of sparse predictions,
     * this class generates per-thread caches that determine which labels are true/false positives and false negatives.
     */
    class EvaluateMetrics : public parallel::TaskGenerator {
    public:
        using LabelList = std::vector<std::vector<label_id_t>>;

        EvaluateMetrics(const LabelList* sparse_labels, const IndexMatrix* sparse_predictions, long num_labels);
        ~EvaluateMetrics() override;

        void prepare(long num_threads, long chunk_size) override;
        void init_thread(thread_id_t thread_id) override;

        void add_precision_at_k(long k, std::string name = {});
        void add_dcg_at_k(long k, bool normalize, std::string name = {});
        void add_abandonment_at_k(long k, std::string name = {});
        MacroMetricReporter* add_macro_at_k(long k);

        [[nodiscard]] std::vector<std::pair<std::string, double>> get_metrics() const;

        void run_task(long task_id, thread_id_t thread_id);
        void run_tasks(long begin, long end, thread_id_t thread_id) override;

        void finalize() override;

        [[nodiscard]] long num_tasks() const override;

        using prediction_t = Eigen::Ref<const Eigen::Matrix<long, 1, Eigen::Dynamic>>;
        static void process_prediction(const std::vector<label_id_t>& raw_labels, const prediction_t& raw_prediction,
                                       std::vector<sTrueLabelInfo>& proc_labels, std::vector<sPredLabelInfo>& proc_pred);

    private:
        const LabelList* m_Labels;
        const IndexMatrix* m_Predictions;
        long m_NumLabels;

        std::vector<std::vector<std::unique_ptr<MetricCollectionInterface>>> m_Collectors;
        std::vector<std::unique_ptr<MetricReportInterface>> m_Metrics;

        // thread local work variables
        std::vector<std::vector<sTrueLabelInfo>> m_ThreadLocalTrueLabels;
        std::vector<std::vector<sPredLabelInfo>> m_ThreadLocalPredictedLabels;
    };

}


#endif //DISMEC_SRC_PREDICTION_EVALUATE_H

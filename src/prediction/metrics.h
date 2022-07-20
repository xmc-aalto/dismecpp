// Copyright (c) 2021, Aalto University, developed by Erik Schultheis
// All rights reserved.
//
// SPDX-License-Identifier: MIT

#ifndef DISMEC_POSTPROCESSING_H
#define DISMEC_POSTPROCESSING_H

#include <vector>
#include <atomic>
#include "matrix_types.h"
#include "data/types.h"
#include "evaluate.h"
#include "utils/sum.h"
#include "utils/confusion_matrix.h"

namespace dismec::prediction {
    using ConfusionMatrix = ConfusionMatrixBase<long>;

    /*!
     * \brief Base class for all metrics that can be calculated during the evaluation phase.
     * \details This is the interface that is used by the evaluation task to handle the individual
     * metrics. This interface handles the data collection part. It provides the following operations:
     *  1) Accumulating a new prediction into the metric
     *  2) reducing multiple instances of the metric interface.
     * The second operation is necessary to allow for efficient multithreaded calculation of the metric.
     */
    class MetricCollectionInterface {
    public:
        using gt_info_vec = std::vector<sTrueLabelInfo>;
        using pd_info_vec = std::vector<sPredLabelInfo>;

        /// Constructor, gets the total number of labels, since these cannot be inferred from the sparse prediction and
        /// ground-truth arrays in the update step.
        explicit MetricCollectionInterface(long num_labels);
        virtual ~MetricCollectionInterface() = default;
        /// Gets the number of labels.
        [[nodiscard]] long num_labels() const { return m_NumLabels; }

        virtual void update(const pd_info_vec& prediction, const gt_info_vec& labels) = 0;
        virtual void reduce(const MetricCollectionInterface& other) = 0;
        [[nodiscard]] virtual std::unique_ptr<MetricCollectionInterface> clone() const = 0;
    private:
        long m_NumLabels;
    };

    class ConfusionMatrixRecorder : public MetricCollectionInterface {
    public:
        ConfusionMatrixRecorder(long num_labels, long k);
        void update(const pd_info_vec& prediction, const gt_info_vec& labels) override;
        void reduce(const MetricCollectionInterface& other) override;
        [[nodiscard]] std::unique_ptr<MetricCollectionInterface> clone() const override;

        [[nodiscard]] long get_k() const { return m_K; }
        [[nodiscard]] ConfusionMatrix get_confusion_matrix(label_id_t label) const;
    private:
        long m_K;
        long m_InstanceCount = 0;
        std::vector<ConfusionMatrix> m_Confusion;
    };

    class InstanceAveragedMetric : public MetricCollectionInterface {
    public:
        explicit InstanceAveragedMetric(long num_labels);
        void reduce(const MetricCollectionInterface& other) override;

        [[nodiscard]] double value() const {
            if(m_NumSamples == 0) return 0.0;
            return m_Accumulator.value() / static_cast<double>(m_NumSamples);
        }
    protected:
        void accumulate(double value);
    private:
        KahanAccumulator<double> m_Accumulator = {};
        long m_NumSamples = 0;
    };


    class InstanceRankedPositives : public InstanceAveragedMetric {
    public:
        InstanceRankedPositives(long num_labels, long k, bool normalize=false);
        InstanceRankedPositives(long num_labels, long k, bool normalize, std::vector<double> weights);
        void update(const pd_info_vec& prediction, const gt_info_vec& labels) override;
        [[nodiscard]] std::unique_ptr<MetricCollectionInterface> clone() const override;
    private:
        long m_K;
        bool m_Normalize;
        std::vector<double> m_Weights;
        std::vector<double> m_Cumulative;
    };

    class AbandonmentAtK : public InstanceAveragedMetric {
    public:
        explicit AbandonmentAtK(long num_labels, long k);
        void update(const pd_info_vec& prediction, const gt_info_vec& labels) override;
        [[nodiscard]] std::unique_ptr<MetricCollectionInterface> clone() const override;
    private:
        long m_K;
    };

    class MetricReportInterface {
    public:
        virtual ~MetricReportInterface() = default;

        using metric_t = std::pair<std::string, double>;
        [[nodiscard]] virtual std::vector<metric_t> get_values() const = 0;
    };

    class InstanceWiseMetricReporter : public MetricReportInterface {
    public:
        InstanceWiseMetricReporter(std::string name, const InstanceAveragedMetric* metric);
        [[nodiscard]] std::vector<metric_t> get_values() const override;
    private:
        std::string m_Name;
        const InstanceAveragedMetric* m_Metric;
    };

    class MacroMetricReporter : public MetricReportInterface {
    public:
        explicit MacroMetricReporter(const ConfusionMatrixRecorder* confusion);
        [[nodiscard]] std::vector<metric_t> get_values() const override;

        enum ReductionType {
            MICRO, MACRO
        };

        void add_coverage(double threshold, std::string name={});
        void add_precision(ReductionType reduction = MACRO, std::string name={});
        void add_accuracy(ReductionType reduction = MACRO, std::string name={});
        void add_specificity(ReductionType reduction = MACRO, std::string name={});
        void add_balanced_accuracy(ReductionType reduction = MACRO, std::string name={});
        void add_informedness(ReductionType reduction = MACRO, std::string name={});
        void add_markedness(ReductionType reduction = MACRO, std::string name={});
        void add_recall(ReductionType reduction = MACRO, std::string name={});
        void add_fowlkes_mallows(ReductionType reduction = MACRO, std::string name={});
        void add_negative_predictive_value(ReductionType reduction = MACRO, std::string name={});
        void add_matthews(ReductionType reduction = MACRO, std::string name={});
        void add_positive_likelihood_ratio(ReductionType reduction = MACRO, std::string name={});
        void add_negative_likelihood_ratio(ReductionType reduction = MACRO, std::string name={});
        void add_diagnostic_odds_ratio(ReductionType reduction = MACRO, std::string name={});
        void add_f_measure(ReductionType reduction = MACRO, double beta = 1.0, std::string name={});
        void add_confusion_matrix();

        void add_reduction(std::string name, ReductionType type, std::function<double(const ConfusionMatrix&)>);
    private:
        void add_reduction_helper(std::string name, const char* pattern, ReductionType type,
                                  std::function<double(const ConfusionMatrix&)> fn);
        using reduction_fn = std::function<double(const ConfusionMatrix&)>;
        std::vector<std::pair<std::string, reduction_fn>> m_MacroReductions;
        std::vector<std::pair<std::string, reduction_fn>> m_MicroReductions;
        const ConfusionMatrixRecorder* m_ConfusionMatrix;
    };
}

#endif //DISMEC_POSTPROCESSING_H

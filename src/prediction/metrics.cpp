// Copyright (c) 2021, Aalto University, developed by Erik Schultheis
// All rights reserved.
//
// SPDX-License-Identifier: MIT

#include "metrics.h"
#include "spdlog/fmt/fmt.h"
#include "utils/throw_error.h"
#include <numeric>

using namespace dismec::prediction;

MetricCollectionInterface::MetricCollectionInterface(long num_labels) : m_NumLabels(num_labels) {
    if(num_labels <= 0) {
        THROW_EXCEPTION(std::invalid_argument, "Number of labels must be positive. Got {}", num_labels);
    }
}

// ---------------------------------------------------------------------------------------------------------------------
//                          Micro Confusion Matrix
// ---------------------------------------------------------------------------------------------------------------------

ConfusionMatrixRecorder::ConfusionMatrixRecorder(long num_labels, long k) : MetricCollectionInterface(num_labels), m_K(k) {
    m_Confusion.resize(num_labels);
}

void ConfusionMatrixRecorder::update(const pd_info_vec& prediction, const gt_info_vec& labels) {
    for(long j = 0; j < m_K; ++j) {
        if(prediction[j].Correct) {
            ++m_Confusion[prediction[j].Label.to_index()].TruePositives;
        } else {
            ++m_Confusion[prediction[j].Label.to_index()].FalsePositives;
        }
    }

    for(const auto& lbl : labels) {
        if(lbl.Rank >= m_K) {
            ++m_Confusion[lbl.Label.to_index()].FalseNegatives;
        }
    }
    ++m_InstanceCount;
}

void ConfusionMatrixRecorder::reduce(const MetricCollectionInterface& other) {
    const auto& other_direct = dynamic_cast<const ConfusionMatrixRecorder&>(other);

    ALWAYS_ASSERT_EQUAL(m_K, other_direct.m_K, "Mismatch in confusion matrix K: {} and {}");
    ALWAYS_ASSERT_EQUAL(num_labels(), other.num_labels(), "Mismatch in number of labels: {} and {}");

    m_InstanceCount += other_direct.m_InstanceCount;
    for(int i = 0; i < m_Confusion.size(); ++i) {
        m_Confusion[i] += other_direct.m_Confusion[i];
    }
}

ConfusionMatrix ConfusionMatrixRecorder::get_confusion_matrix(label_id_t label) const {
    assert(label.to_index() < m_Confusion.size());
    auto base = m_Confusion[label.to_index()];
    base.TrueNegatives = m_InstanceCount - base.TruePositives - base.FalsePositives - base.FalseNegatives;
    return base;
}

std::unique_ptr<MetricCollectionInterface> ConfusionMatrixRecorder::clone() const {
    return std::make_unique<ConfusionMatrixRecorder>(num_labels(), m_K);
}

// ---------------------------------------------------------------------------------------------------------------------
InstanceAveragedMetric::InstanceAveragedMetric(long num_labels) : MetricCollectionInterface(num_labels) {

}

void InstanceAveragedMetric::accumulate(double value) {
    m_Accumulator += value;
    ++m_NumSamples;
}

void InstanceAveragedMetric::reduce(const MetricCollectionInterface& other) {
    const auto& cast = dynamic_cast<const InstanceAveragedMetric&>(other);
    // add up weights and accumulated values
    m_Accumulator += cast.m_Accumulator.value();
    m_NumSamples += cast.m_NumSamples;
}


// ---------------------------------------------------------------------------------------------------------------------
//                              Generalization of Precision and DCG at K
// ---------------------------------------------------------------------------------------------------------------------

namespace {
    std::vector<double> uniform_weights(long k) {
        std::vector<double> weights;
        weights.reserve(k);
        std::fill_n(std::back_inserter(weights), k, 1.0 / static_cast<double>(k));
        return weights;
    }
}

InstanceRankedPositives::InstanceRankedPositives(long num_labels, long k, bool normalize) :
    InstanceRankedPositives(num_labels, k, normalize, uniform_weights(k))
{
}
#include <iostream>
InstanceRankedPositives::InstanceRankedPositives(long num_labels, long k, bool normalize, std::vector<double> weights) :
    InstanceAveragedMetric(num_labels), m_K(k), m_Normalize(normalize), m_Weights( std::move(weights) ) {
    ALWAYS_ASSERT_EQUAL(m_K, m_Weights.size(), "Mismatch between k={} and #weights = {}");
    // Exclusive scan -- prepend a zero
    m_Cumulative.push_back(0.0);
    std::partial_sum(begin(m_Weights), end(m_Weights), std::back_inserter(m_Cumulative));
}


// With -O3 GCC 9 produces an ICE here, so we manually fix the optimization options here
#pragma GCC push_options
#pragma GCC optimize("-O1")
void InstanceRankedPositives::update(const pd_info_vec& prediction, const gt_info_vec& labels) {
    assert(prediction.size() >= m_K);
    double correct = 0;
    for(long j = 0; j < m_K; ++j) {
        if(prediction[j].Correct) {
            correct += m_Weights[j];
        }
    }

    if(m_Normalize) {
        long step = std::min(m_K, (long)labels.size());
        correct /= m_Cumulative[step];
    }

    accumulate(correct);
}
#pragma GCC pop_options

std::unique_ptr<MetricCollectionInterface> InstanceRankedPositives::clone() const {
    return std::make_unique<InstanceRankedPositives>(num_labels(), m_K, m_Normalize, m_Weights);
}


// ---------------------------------------------------------------------------------------------------------------------
//                              Abandonment At K
// ---------------------------------------------------------------------------------------------------------------------
AbandonmentAtK::AbandonmentAtK(long num_labels, long k) : InstanceAveragedMetric(num_labels), m_K(k) {
}

void AbandonmentAtK::update(const pd_info_vec& prediction, const gt_info_vec& labels) {
    assert(prediction.size() >= m_K);
    double correct = 0.0;
    for(long j = 0; j < m_K; ++j) {
        if(prediction[j].Correct) {
            correct = 1.0;
            break;
        }
    }
    accumulate(correct);
}

std::unique_ptr<MetricCollectionInterface> AbandonmentAtK::clone() const {
    return std::make_unique<AbandonmentAtK>(num_labels(), m_K);
}

// ---------------------------------------------------------------------------------------------------------------------
//                              Metric Reporters
// ---------------------------------------------------------------------------------------------------------------------

InstanceWiseMetricReporter::InstanceWiseMetricReporter(std::string name, const InstanceAveragedMetric* metric) :
    m_Name(std::move(name)), m_Metric(metric) {

}

auto InstanceWiseMetricReporter::get_values() const -> std::vector<metric_t> {
    return {{m_Name, m_Metric->value()}};
}

void MacroMetricReporter::add_coverage(double threshold, std::string name) {
    if(name.empty()) {
        name = fmt::format("Cov@{}", m_ConfusionMatrix->get_k());
    }

    auto fn = [threshold](const ConfusionMatrix& cm){
        if(recall(cm) > threshold) {
            return 1.0;
        } else {
            return 0.0;
        }
    };
    add_reduction(std::move(name), MACRO, fn);
}

namespace {
    constexpr const char* reduction_name(MacroMetricReporter::ReductionType type) {
        switch (type) {
            case dismec::prediction::MacroMetricReporter::MACRO: return "Macro";
            case dismec::prediction::MacroMetricReporter::MICRO: return "Micro";
        }
        __builtin_unreachable();
    }
}

// NOLINTNEXTLINE(cppcoreguidelines-macro-usage)
#define IMPLEMENT_ADD_METRIC(METRIC, SHORTHAND)                                                 \
void MacroMetricReporter::add_##METRIC(ReductionType reduction, std::string name) {             \
auto fn = [](const ConfusionMatrix& cm){ return METRIC(cm); };                                  \
add_reduction_helper(std::move(name), "{}" SHORTHAND "@{}", reduction, fn);                     \
}

IMPLEMENT_ADD_METRIC(precision, "P");
IMPLEMENT_ADD_METRIC(accuracy, "ACC");
IMPLEMENT_ADD_METRIC(specificity, "SPC");
IMPLEMENT_ADD_METRIC(balanced_accuracy, "BA");
IMPLEMENT_ADD_METRIC(informedness, "BM");
IMPLEMENT_ADD_METRIC(markedness, "MK");
IMPLEMENT_ADD_METRIC(recall, "R");
IMPLEMENT_ADD_METRIC(fowlkes_mallows, "FM");
IMPLEMENT_ADD_METRIC(negative_predictive_value, "NPV");
IMPLEMENT_ADD_METRIC(matthews, "MCC");
IMPLEMENT_ADD_METRIC(positive_likelihood_ratio, "LR+");
IMPLEMENT_ADD_METRIC(negative_likelihood_ratio, "LR-");
IMPLEMENT_ADD_METRIC(diagnostic_odds_ratio, "DOR");


void MacroMetricReporter::add_f_measure(ReductionType reduction, double beta, std::string name) {
    if(name.empty()) {
        name = fmt::format("{}F{}@{}", reduction_name(reduction), beta, m_ConfusionMatrix->get_k());
    }

    auto fn = [beta](const ConfusionMatrix& cm){
        return f_beta(cm, beta);
    };

    add_reduction(std::move(name), reduction, fn);
}

void MacroMetricReporter::add_confusion_matrix() {
    add_reduction(fmt::format("MicroTP@{}", m_ConfusionMatrix->get_k()), MICRO,
                  [](const ConfusionMatrix& cm){ return true_positive_fraction(cm); });
    add_reduction(fmt::format("MicroFP@{}", m_ConfusionMatrix->get_k()), MICRO,
                  [](const ConfusionMatrix& cm){ return false_positive_fraction(cm); });
    add_reduction(fmt::format("MicroTN@{}", m_ConfusionMatrix->get_k()), MICRO,
                  [](const ConfusionMatrix& cm){ return true_negative_fraction(cm); });
    add_reduction(fmt::format("MicroFN@{}", m_ConfusionMatrix->get_k()), MICRO,
                  [](const ConfusionMatrix& cm){ return false_negative_fraction(cm); });
}

void MacroMetricReporter::add_reduction_helper(std::string name, const char* pattern, ReductionType reduction,
                                               std::function<double(const ConfusionMatrix&)> fn) {
    if(name.empty()) {
        name = fmt::format(pattern, reduction_name(reduction), m_ConfusionMatrix->get_k());
    }
    add_reduction(std::move(name), reduction, std::move(fn));
}

void MacroMetricReporter::add_reduction(std::string name, ReductionType type, std::function<double(const ConfusionMatrix&)> fn) {
    if(type == MACRO) {
        m_MacroReductions.emplace_back(std::move(name), std::move(fn));
    } else {
        m_MicroReductions.emplace_back(std::move(name), std::move(fn));
    }
}

MacroMetricReporter::MacroMetricReporter(const ConfusionMatrixRecorder* confusion) : m_ConfusionMatrix(confusion) {
    if(confusion == nullptr) {
        THROW_EXCEPTION(std::invalid_argument, "ConfusionMatrixRecorder cannot be null");
    }
}

auto MacroMetricReporter::get_values() const -> std::vector<metric_t> {
    std::vector<metric_t> metric;
    metric.reserve(m_MacroReductions.size());
    for(const auto& red : m_MacroReductions) {
        metric.emplace_back(red.first, 0.0);
    }

    ConfusionMatrix micro;

    for(int l = 0; l < m_ConfusionMatrix->num_labels(); ++l) {
        ConfusionMatrix cm = m_ConfusionMatrix->get_confusion_matrix(label_id_t{l});
        micro += cm;
        for(int i = 0; i < m_MacroReductions.size(); ++i) {
            metric[i].second += m_MacroReductions[i].second(cm);
        }
    }

    auto normalize = static_cast<double>(m_ConfusionMatrix->num_labels());
    if(normalize != 0) {
        for(int i = 0; i < m_MacroReductions.size(); ++i) {
            metric[i].second /= normalize;
        }
    } else {
        for(int i = 0; i < m_MacroReductions.size(); ++i) {
            if(metric[i].second != 0) {
                metric[i].second = std::numeric_limits<double>::quiet_NaN();
            }
        }
    }

    for(const auto& [name, fn] : m_MicroReductions) {
        metric.emplace_back(name, fn(micro));
    }

    return metric;
}

#include "doctest.h"

// NOLINTBEGIN(cppcoreguidelines-avoid-magic-numbers)

namespace {
    using pred_mat_t = Eigen::Matrix<long, 1, Eigen::Dynamic>;
    auto make_labels(std::initializer_list<long> init_list) {
        auto vec = std::vector<dismec::label_id_t>{};
        vec.reserve(init_list.size());
        for(const auto& i : init_list) {
            vec.emplace_back(i);
        }
        return vec;
    }

    template<class T>
    void update_metric(T& target, std::initializer_list<long> prediction, std::initializer_list<long> labels) {
        auto labels_vec = make_labels(labels);
        auto pred_mat = pred_mat_t{prediction};
        std::vector<sTrueLabelInfo> true_info;
        std::vector<sPredLabelInfo> pred_info;
        EvaluateMetrics::process_prediction(labels_vec, pred_mat, true_info, pred_info);
        target.update(pred_info, true_info);
    }

}

/*! \test Validate precision at k by using a few examples. Note that this test validates that the calculations are
 * correct, but not that the code will also work when called from multiple threads.
 */
TEST_CASE("precision_at_k") {
    auto pat3 = InstanceRankedPositives(13, 3);
    CHECK(pat3.value() == 0.0);
    update_metric(pat3, {2, 4, 6, 12}, {1, 4, 8, 12});
    CHECK(pat3.value() == 1.0 / 3.0);
    update_metric(pat3, {3, 1, 2, 5}, {2, 3});
    CHECK(pat3.value() == 3.0 / 6.0);
    update_metric(pat3, {1, 2, 3, 4, 5}, {4, 5, 6});
    CHECK(pat3.value() == 3.0 / 9.0);
    update_metric(pat3, {1, 2, 3}, {});
    CHECK(pat3.value() == 3.0 / 12.0);
    update_metric(pat3, {3, 2, 1}, {1, 2, 3});
    CHECK(pat3.value() == 6.0 / 15.0);
}

/*! \test Validate abandonment at k by using a few examples. Note that this test validates that the calculations are
 * correct, but not that the code will also work when called from multiple threads.
 */
TEST_CASE("abandonment_at_k") {
    auto aat3 = AbandonmentAtK(13, 3);
    CHECK(aat3.value() == 0.0);
    update_metric(aat3, {2, 4, 6, 12}, {1, 4, 8, 12});
    CHECK(aat3.value() == 1.0 / 1.0);
    update_metric(aat3, {3, 1, 2, 5}, {2, 3});
    CHECK(aat3.value() == 2.0 / 2.0);
    update_metric(aat3, {1, 2, 3, 4, 5}, {4, 5, 6});
    CHECK(aat3.value() == 2.0 / 3.0);
    update_metric(aat3, {1, 2, 3}, {});
    CHECK(aat3.value() == 2.0 / 4.0);
    update_metric(aat3, {3, 2, 1}, {1, 2, 3});
    CHECK(aat3.value() == 3.0 / 5.0);
}


/*! \test Validate coverage at k by using a few examples. Note that this test validates that the calculations are
 * correct, but not that the code will also work when called from multiple threads.
 */
 /*
TEST_CASE("coverage_at_k") {
    auto cat3 = CoverageAtK(3, 20);
    CHECK(cat3.value() == 0.0);
    cat3.update(pred_mat_t{{2, 4, 6, 12}}, make_labels({1, 4, 8, 12}));
    CHECK(cat3.value() == 1.0 / 20.0);
    cat3.update(pred_mat_t{{3, 1, 2, 5}}, make_labels({2, 3}));
    CHECK(cat3.value() == 3.0 / 20.0);
    cat3.update(pred_mat_t{{1, 2, 3, 4, 5}}, make_labels({4, 5, 6}));
    CHECK(cat3.value() == 3.0 / 20.0);
    cat3.update(pred_mat_t{{1, 2, 3}}, {});
    CHECK(cat3.value() == 3.0 / 20.0);
    cat3.update(pred_mat_t{{3, 2, 1}}, make_labels({1, 2, 3}));
    CHECK(cat3.value() == 4.0 / 20.0);
}
*/
// NOLINTEND(cppcoreguidelines-avoid-magic-numbers)
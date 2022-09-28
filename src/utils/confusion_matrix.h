// Copyright (c) 2021, Aalto University, developed by Erik Schultheis
// All rights reserved.
//
// SPDX-License-Identifier: MIT

#ifndef DISMEC_SRC_UTILS_CONFUSION_MATRIX_H
#define DISMEC_SRC_UTILS_CONFUSION_MATRIX_H

#include <cmath>

namespace dismec::confusion_matrix_detail {
    template<class T>
    struct ConfusionMatrixBase {
        T TruePositives = 0;
        T FalsePositives = 0;
        T TrueNegatives = 0;
        T FalseNegatives = 0;
    };

    template<class T>
    constexpr ConfusionMatrixBase<T>& operator+=(ConfusionMatrixBase<T>& lhs, const ConfusionMatrixBase<T>& rhs) {
        lhs.TruePositives += rhs.TruePositives;
        lhs.FalsePositives += rhs.FalsePositives;
        lhs.TrueNegatives += rhs.TrueNegatives;
        lhs.FalseNegatives += rhs.FalseNegatives;
        return lhs;
    }

    // https://en.wikipedia.org/wiki/Confusion_matrix

    template<class T>
    constexpr T predicted_positives(const ConfusionMatrixBase<T>& matrix) {
        return matrix.TruePositives + matrix.FalsePositives;
    }

    template<class T>
    constexpr T predicted_negatives(const ConfusionMatrixBase<T>& matrix) {
        return matrix.TrueNegatives + matrix.FalseNegatives;
    }

    template<class T>
    constexpr T positives(const ConfusionMatrixBase<T>& matrix) {
        return matrix.TruePositives + matrix.FalseNegatives;
    }

    template<class T>
    constexpr T negatives(const ConfusionMatrixBase<T>& matrix) {
        return matrix.TrueNegatives + matrix.FalsePositives;
    }

    template<class T>
    constexpr T total_samples(const ConfusionMatrixBase<T>& matrix) {
        return matrix.TrueNegatives + matrix.TruePositives + matrix.FalseNegatives + matrix.FalsePositives;
    }

    /// Division that returns 0.0 whenever the numerator is 0, even if the denominator is 0.
    /// This means that 0/0 := 0 for the purpose of this operation. Any other division by zero will result in +-inf.
    template<class Scalar>
    constexpr double save_div(Scalar num, Scalar den) {
        if(num == 0) return 0.0;
        return static_cast<double>(num) / static_cast<double>(den);
    }

    // total-normalized values
    template<class T>
    constexpr double true_positive_fraction(const ConfusionMatrixBase<T>& matrix) {
        return save_div(matrix.TruePositives, total_samples(matrix));
    }

    template<class T>
    constexpr double false_positive_fraction(const ConfusionMatrixBase<T>& matrix) {
        return save_div(matrix.FalsePositives, total_samples(matrix));
    }

    template<class T>
    constexpr double true_negative_fraction(const ConfusionMatrixBase<T>& matrix) {
        return save_div(matrix.TrueNegatives, total_samples(matrix));
    }

    template<class T>
    constexpr double false_negative_fraction(const ConfusionMatrixBase<T>& matrix) {
        return save_div(matrix.FalseNegatives, total_samples(matrix));
    }

    template<class T>
    constexpr double accuracy(const ConfusionMatrixBase<T>& matrix) {
        return save_div(matrix.TruePositives + matrix.TrueNegatives, total_samples(matrix));
    }

    template<class T>
    constexpr double prevalence(const ConfusionMatrixBase<T>& matrix) {
        return save_div(positives(matrix), total_samples(matrix));
    }

    // prediction-normalized values

    template<class T>
    constexpr double positive_predictive_value(const ConfusionMatrixBase<T>& matrix) {
        return save_div(matrix.TruePositives, predicted_positives(matrix));
    }

    template<class T>
    constexpr double false_discovery_rate(const ConfusionMatrixBase<T>& matrix) {
        return 1.0 - positive_predictive_value(matrix);
    }

    template<class T>
    constexpr double negative_predictive_value(const ConfusionMatrixBase<T>& matrix) {
        return save_div(matrix.TrueNegatives, predicted_negatives(matrix));
    }

    template<class T>
    constexpr double false_omission_rate(const ConfusionMatrixBase<T>& matrix) {
        return 1.0 - negative_predictive_value(matrix);
    }

    // detection rates
    template<class T>
    constexpr double true_positive_rate(const ConfusionMatrixBase<T>& matrix) {
        return save_div(matrix.TruePositives, positives(matrix));
    }

    template<class T>
    constexpr double false_negative_rate(const ConfusionMatrixBase<T>& matrix) {
        return save_div(matrix.FalseNegatives, positives(matrix));
    }

    template<class T>
    constexpr double false_positive_rate(const ConfusionMatrixBase<T>& matrix) {
        return save_div(matrix.FalsePositives, negatives(matrix));
    }

    template<class T>
    constexpr double true_negative_rate(const ConfusionMatrixBase<T>& matrix) {
        return save_div(matrix.TrueNegatives, negatives(matrix));
    }

    // Common names

    template<class T>
    constexpr double precision(const ConfusionMatrixBase<T>& matrix) {
        return positive_predictive_value(matrix);
    }

    template<class T>
    constexpr double recall(const ConfusionMatrixBase<T>& matrix) {
        return true_positive_rate(matrix);
    }

    template<class T>
    constexpr double sensitivity(const ConfusionMatrixBase<T>& matrix) {
        return true_positive_rate(matrix);
    }

    template<class T>
    constexpr double specificity(const ConfusionMatrixBase<T>& matrix) {
        return true_negative_rate(matrix);
    }

    // complex metrics

    template<class T>
    constexpr double informedness(const ConfusionMatrixBase<T>& matrix) {
        return true_positive_rate(matrix) + true_negative_rate(matrix) - 1.0;
    }

    template<class T>
    constexpr double markedness(const ConfusionMatrixBase<T>& matrix) {
        return positive_predictive_value(matrix) + negative_predictive_value(matrix) - 1.0;
    }

    template<class T>
    constexpr double fowlkes_mallows(const ConfusionMatrixBase<T>& matrix) {
        return std::sqrt(positive_predictive_value(matrix) * true_positive_rate(matrix));
    }

    template<class T>
    constexpr double positive_likelihood_ratio(const ConfusionMatrixBase<T>& matrix) {
        return save_div(true_positive_rate(matrix), false_positive_rate(matrix));
    }

    template<class T>
    constexpr double negative_likelihood_ratio(const ConfusionMatrixBase<T>& matrix) {
        return save_div(false_negative_rate(matrix), true_negative_rate(matrix));
    }

    template<class T>
    constexpr double diagnostic_odds_ratio(const ConfusionMatrixBase<T>& matrix) {
        return save_div(positive_likelihood_ratio(matrix), negative_likelihood_ratio(matrix));
    }

    template<class T>
    constexpr double matthews(const ConfusionMatrixBase<T>& matrix) {
        return std::sqrt(true_positive_rate(matrix) * true_negative_rate(matrix) * positive_predictive_value(matrix) *
                             negative_predictive_value(matrix)) - std::sqrt(false_negative_rate(matrix) *
                             false_positive_rate(matrix) * false_omission_rate(matrix) * false_discovery_rate(matrix));
    }

    template<class T>
    constexpr double balanced_accuracy(const ConfusionMatrixBase<T>& matrix) {
        return (true_positive_rate(matrix) + true_negative_rate(matrix)) / 2.0;
    }

    template<class T>
    constexpr double f_beta(const ConfusionMatrixBase<T>& matrix, double beta) {
        double bs = beta * beta;
        double num = (1.0 + bs) * static_cast<double>(matrix.TruePositives);
        double den = num + bs * static_cast<double>(matrix.FalseNegatives) + static_cast<double>(matrix.FalsePositives);
        if(den == 0.0) return 0.0;
        return num / den;
    }
}

namespace dismec {
    using confusion_matrix_detail::ConfusionMatrixBase;
}

#endif //DISMEC_SRC_UTILS_CONFUSION_MATRIX_H

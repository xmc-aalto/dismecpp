// Copyright (c) 2021, Aalto University, developed by Erik Schultheis
// All rights reserved.
//
// SPDX-License-Identifier: MIT

// We are deliberately doing things wrong here, and apparently that triggers a gcc warning.
#include "model/dense.h"
#include "spdlog/fmt/fmt.h"
#include "utils/eigen_generic.h"

using namespace dismec;
using namespace dismec::model;

namespace {
    /*!
     * \brief Checks that `v` is positive and returns `v`.
     * \details This function can be used to verify arguments in the initializer list of the constructor.
     * \param v Number to check.
     * \param error_msg This message will be used in the construction of the `std::invalid_argument` exception in case
     * of error.
     * \throws std::invalid_argument if `v <= 0`.
     */
    long check_positive(long v, const char* error_msg) {
        if(v > 0) {
            return v;
        }
        throw std::invalid_argument(error_msg);
    }
}

DenseModel::DenseModel(const weight_matrix_ptr& weights) :
        DenseModel(weights, {label_id_t{0}, weights->cols(), weights->cols()}) {
    /// \internal Deliberately using const& here, instead of moving the shared_ptr, because that is still cheap
    /// and with move it becomes difficult to avoid reading from moved-from shared_ptr
}

DenseModel::DenseModel(weight_matrix_ptr weights, PartialModelSpec partial) :
        Model(partial), m_Weights(std::move(weights))
{
    if(m_Weights->cols() != partial.label_count) {
        throw std::invalid_argument(fmt::format("Declared {} weights, but got matrix with {} columns",
                                                partial.label_count, m_Weights->cols()));
    }
}

DenseModel::DenseModel(long num_features, long num_labels):
        DenseModel(num_features, PartialModelSpec{label_id_t{0}, num_labels, num_labels})
{
}

DenseModel::DenseModel(long num_features, PartialModelSpec partial) :
        DenseModel(std::make_shared<WeightMatrix>(
            check_positive(num_features, "Number of features must be positive!"),
            check_positive(partial.label_count, "Number of weight must be positive!")),
        partial)
{
}

long DenseModel::num_features() const {
    return m_Weights->rows();
}


void DenseModel::get_weights_for_label_unchecked(label_id_t label, Eigen::Ref<DenseRealVector> target) const
{
    target = m_Weights->col(label.to_index());
}

void DenseModel::set_weights_for_label_unchecked(label_id_t label, const WeightVectorIn& weights)
{
    visit([this, label](auto&& v){
        m_Weights->col(label.to_index()) = v;
    }, weights);
}

void DenseModel::predict_scores_unchecked(const FeatureMatrixIn& instances, PredictionMatrixOut target) const {
    visit([&, this](const auto& features) {
        target.noalias() = features * (*m_Weights);
        }, instances);
}

// Copyright (c) 2021, Aalto University, developed by Erik Schultheis
// All rights reserved.
//
// SPDX-License-Identifier: MIT

#include "model/model.h"
#include "spdlog/fmt/fmt.h"
#include "utils/eigen_generic.h"

using namespace dismec;
using namespace dismec::model;

Model::Model(PartialModelSpec spec) :
    m_LabelsBegin(spec.first_label), m_LabelsEnd(spec.first_label + spec.label_count), m_NumLabels(spec.total_labels)
{
    if(m_NumLabels <= 0) {
        throw std::invalid_argument( fmt::format("Total number of labels must be positive! Got {}.", m_NumLabels) );
    }

    if(m_LabelsEnd <= m_LabelsBegin || m_LabelsBegin < label_id_t{0} || m_LabelsEnd.to_index() > m_NumLabels) {
        throw std::invalid_argument( fmt::format("Invalid label range [{}, {}) specified. Total number of labels"
                                            "was declared as {}.",
                                            m_LabelsBegin.to_index(), m_LabelsEnd.to_index(), m_NumLabels) );
    }
}


label_id_t Model::adjust_label(label_id_t label) const {
    if(label < labels_begin() || label >= labels_end()) {
        throw std::out_of_range(
                fmt::format("label index {} is invalid. Labels must be in [{}, {})",
                            label.to_index(), labels_begin().to_index(), labels_end().to_index()));
    }
    return label_id_t{label - labels_begin()};
}

bool Model::is_partial_model() const {
    return m_LabelsBegin != label_id_t{0} || m_LabelsEnd.to_index() != num_labels();
}

void Model::get_weights_for_label(label_id_t label, Eigen::Ref<DenseRealVector> target) const {
    if(target.size() != num_features()) {
        throw std::invalid_argument(
                fmt::format("target size {} does not match number of features {}.",
                            target.size(), num_features()));
    }

    get_weights_for_label_unchecked(adjust_label(label), target);
}

void Model::set_weights_for_label(label_id_t label, const WeightVectorIn& weights) {
    if(weights.size() != num_features()) {
        throw std::invalid_argument(
                fmt::format("weight size {} does not match number of features {}.",
                            weights.size(), num_features()));
    }
    set_weights_for_label_unchecked(adjust_label(label), weights);
}

void Model::predict_scores(const FeatureMatrixIn& instances, PredictionMatrixOut target) const {
    // check number of instances
    if(instances.rows() != target.rows()) {
        throw std::logic_error(fmt::format("Mismatch in number of rows between instances ({}) and target ({})",
                                           instances.rows(), target.rows()));
    }

    // check number of labels
    if(target.cols() != num_weights()) {
        throw std::logic_error(
                fmt::format("Wrong number of columns in target ({}). Expect one column for each of the {} labels.",
                            target.cols(), num_weights()));
    }

    if(instances.cols() != num_features()) {
        throw std::logic_error(
                fmt::format("Wrong number of columns in instances ({}). Expect one column for each of the {} features.",
                            instances.cols(), num_features()));
    }
    predict_scores_unchecked(instances, target);
}


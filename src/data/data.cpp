// Copyright (c) 2021, Aalto University, developed by Erik Schultheis
// All rights reserved.
//
// SPDX-License-Identifier: MIT

#include <fstream>
#include "data.h"
#include "utils/conversion.h"
#include "spdlog/spdlog.h"

using namespace dismec;

long DatasetBase::num_positives(label_id_t id) const {
    return (get_labels(id)->array() == 1.0).count();
}

long DatasetBase::num_negatives(label_id_t id) const {
    return num_examples() - num_positives(id);
}

std::shared_ptr<const BinaryLabelVector> DatasetBase::get_labels(label_id_t id) const {
    // convert sparse to dense
    auto label_vector = std::make_shared<BinaryLabelVector>(num_examples());
    get_labels(id, *label_vector);
    return std::move(label_vector);
}

void BinaryData::get_labels(label_id_t i, Eigen::Ref<BinaryLabelVector> target) const {
    if(i != label_id_t{0}) {
        throw std::out_of_range("Binary problems only have a single class with id `0`");
    }
    target = *m_Labels;
}

long BinaryData::num_labels() const noexcept {
    return 1;
}

std::shared_ptr<const GenericFeatureMatrix> DatasetBase::get_features() const {
    return std::const_pointer_cast<const GenericFeatureMatrix>(m_Features);
}

std::shared_ptr<GenericFeatureMatrix> DatasetBase::edit_features() {
    return m_Features;
}


long DatasetBase::num_features() const noexcept {
    return m_Features->cols();
}

long DatasetBase::num_examples() const noexcept {
    return m_Features->rows();
}

DatasetBase::DatasetBase(SparseFeatures x) : m_Features(std::make_shared<GenericFeatureMatrix>(x.markAsRValue())) {}
DatasetBase::DatasetBase(DenseFeatures x) : m_Features(std::make_shared<GenericFeatureMatrix>(std::move(x))) {}

long MultiLabelData::num_labels() const noexcept {
    return ssize(m_Labels);
}

void MultiLabelData::get_labels(label_id_t label, Eigen::Ref<BinaryLabelVector> target) const {
    // convert sparse to dense
    const auto& examples = m_Labels.at(label.to_index());
    target.setConstant(-1);
    for(const auto& ex : examples) {
        target.coeffRef(ex) = 1;
    }
}

const std::vector<long>& MultiLabelData::get_label_instances(label_id_t label) const {
    return m_Labels.at(label.to_index());
}

long MultiLabelData::num_positives(label_id_t id) const {
    return ssize(m_Labels.at(id.to_index()));
}

long MultiLabelData::num_negatives(label_id_t id) const {
    return num_examples() - ssize(m_Labels.at(id.to_index()));
}

void MultiLabelData::select_labels(label_id_t start, label_id_t end) {
    if(end.to_index() < 0 || end.to_index() > num_labels()) {
        end = label_id_t{static_cast<int_fast32_t>(m_Labels.size())};
    }

    std::vector<std::vector<long>> sub_labels;
    sub_labels.reserve(end-start);
    std::move(begin(m_Labels) + start.to_index(), std::begin(m_Labels) + end.to_index(), std::back_inserter(sub_labels));
    m_Labels = std::move(sub_labels);
}

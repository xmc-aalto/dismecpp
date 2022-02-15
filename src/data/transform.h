// Copyright (c) 2021, Aalto University, developed by Erik Schultheis
// All rights reserved.
//
// SPDX-License-Identifier: MIT

#ifndef DISMEC_SRC_DATA_TRANSFORM_H
#define DISMEC_SRC_DATA_TRANSFORM_H

#include "data/types.h"
#include "matrix_types.h"

void augment_features_with_bias(DatasetBase& data, real_t bias=1);
SparseFeatures augment_features_with_bias(const SparseFeatures& features, real_t bias=1);
DenseFeatures augment_features_with_bias(const DenseFeatures & features, real_t bias=1);

DenseRealVector get_mean_feature(const GenericFeatureMatrix& features);
DenseRealVector get_mean_feature(const SparseFeatures& features);
DenseRealVector get_mean_feature(const DenseFeatures& features);

std::vector<long> count_features(const SparseFeatures& features);

void normalize_instances(DatasetBase& data);
void normalize_instances(SparseFeatures& features);
void normalize_instances(DenseFeatures & features);

Eigen::PermutationMatrix<Eigen::Dynamic, Eigen::Dynamic, int> sort_features_by_frequency(DatasetBase& data);
Eigen::PermutationMatrix<Eigen::Dynamic, Eigen::Dynamic, int> sort_features_by_frequency(SparseFeatures& features);
Eigen::PermutationMatrix<Eigen::Dynamic, Eigen::Dynamic, int> sort_features_by_frequency(DenseFeatures& features);

enum class DatasetTransform {
    IDENTITY,           // x
    ONE_PLUS_LOG,       // 1 + log(x)
    LOG_ONE_PLUS,       // log(1+x)
};

void transform_features(DatasetBase& data, DatasetTransform transform);
void transform_features(SparseFeatures& features, DatasetTransform transform);
void transform_features(DenseFeatures& features, DatasetTransform transform);

#endif //DISMEC_SRC_DATA_TRANSFORM_H

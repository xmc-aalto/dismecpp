// Copyright (c) 2021, Aalto University, developed by Erik Schultheis
// All rights reserved.
//
// SPDX-License-Identifier: MIT

#include "transform.h"
#include "data/data.h"
#include <random>

using namespace dismec;

namespace {
    struct VisitorBias {
        void operator()(SparseFeatures& features) const {
            features = augment_features_with_bias(features, Bias);
        }
        void operator()(DenseFeatures& features) const {
            features = augment_features_with_bias(features, Bias);
        }

        real_t Bias;
    };
}

void dismec::augment_features_with_bias(DatasetBase& data, real_t bias) {
    visit(VisitorBias{bias}, *data.edit_features());
}

SparseFeatures dismec::augment_features_with_bias(const SparseFeatures& features, real_t bias) {
    SparseFeatures new_sparse{features.rows(), features.cols() + 1};
    new_sparse.reserve(features.nonZeros() + features.rows());
    for (int k=0; k < features.outerSize(); ++k) {
        new_sparse.startVec(k);
        for (SparseFeatures::InnerIterator it(features, k); it; ++it)
        {
            new_sparse.insertBack(it.row(), it.col()) = it.value();
        }
        new_sparse.insertBack(k, features.cols()) = bias;
    }
    new_sparse.finalize();
    return new_sparse;
}

DenseFeatures dismec::augment_features_with_bias(const DenseFeatures & features, real_t bias) {
    DenseFeatures new_features{features.rows(), features.cols() + 1};
    new_features.leftCols(features.cols()) = features;
    new_features.col(features.cols()).setOnes();
    return new_features;
}

DenseRealVector dismec::get_mean_feature(const GenericFeatureMatrix& features) {
    return visit([](auto&& matrix){ return get_mean_feature(matrix); }, features);
}


DenseRealVector dismec::get_mean_feature(const SparseFeatures& features) {
    DenseRealVector result(features.cols());
    result.setZero();

    auto start = features.outerIndexPtr()[0];
    auto end = features.outerIndexPtr()[features.rows()];

    auto* indices = features.innerIndexPtr();
    auto* values = features.valuePtr();
    for(auto index = start; index < end; ++index) {
        auto col = indices[index];
        result[col] += values[index];
    }

    result /= features.rows();
    return result;
}

DenseRealVector dismec::get_mean_feature(const DenseFeatures& features) {
    DenseRealVector result(features.cols());
    result.setZero();

    for(int i = 0; i < features.rows(); ++i) {
        result += features.row(i);
    }

    result /= features.rows();
    return result;
}


void dismec::normalize_instances(DatasetBase& data) {
    visit([](auto&& f){ normalize_instances(f); }, *data.edit_features());
}

void dismec::normalize_instances(SparseFeatures& features) {
    for(int i = 0; i < features.rows(); ++i) {
        real_t norm = features.row(i).norm();
        if(norm > 0) {
            features.row(i) /= norm;
        }
    }
}

void dismec::normalize_instances(DenseFeatures & features) {
    for(int i = 0; i < features.rows(); ++i) {
        real_t norm = features.row(i).norm();
        if(norm > 0) {
            features.row(i) /= norm;
        }
    }
}

Eigen::PermutationMatrix<Eigen::Dynamic, Eigen::Dynamic, int> dismec::sort_features_by_frequency(DatasetBase& data) {
    return visit([](auto&& f){ return sort_features_by_frequency(f); }, *data.edit_features());
}

std::vector<long> dismec::count_features(const SparseFeatures& features) {
    std::vector<long> counts(features.cols(), 0);
    assert(features.isCompressed());

    // count the nonzero features
    // the outer index is the row (instance index), the inner index is the feature id
    auto last = features.innerIndexPtr() + features.nonZeros();
    for(auto start = features.innerIndexPtr(); start != last; ++start) {
        counts[*start] += 1;
    }
    return counts;
}

Eigen::PermutationMatrix<Eigen::Dynamic, Eigen::Dynamic, int> dismec::sort_features_by_frequency(SparseFeatures& features) {
    if(!features.isCompressed()) {
        features.makeCompressed();
    }
    std::vector<long> counts = count_features(features);

    // do an argsort
    types::DenseVector<int> reorder = types::DenseVector<int>::LinSpaced(features.cols(), 0, features.cols());
    std::sort(reorder.begin(), reorder.end(), [&](int a, int b){
        return counts[a] < counts[b];
    });

    //create permutation Matrix with the size of the columns
    Eigen::PermutationMatrix<Eigen::Dynamic, Eigen::Dynamic, int> permutation(reorder);

    features = features * permutation;
    return permutation;
}

Eigen::PermutationMatrix<Eigen::Dynamic, Eigen::Dynamic, int> dismec::sort_features_by_frequency(DenseFeatures& features) {
    Eigen::PermutationMatrix<Eigen::Dynamic, Eigen::Dynamic, int> permutation(features.cols());
    permutation.setIdentity();
    return permutation;
}

void dismec::transform_features(DatasetBase& data, DatasetTransform transform) {
    visit([&](auto&& f){ return transform_features(f, transform); }, *data.edit_features());
}

namespace {
    template<class T>
    void transform_features_imp(T& features, DatasetTransform transform) {
        switch(transform) {
            case DatasetTransform::IDENTITY:
                break;
            case DatasetTransform::LOG_ONE_PLUS:
                features = features.unaryExpr([](const real_t& value) { return std::log1p(value); });
                break;
            case DatasetTransform::ONE_PLUS_LOG:
                features = features.unaryExpr([](const real_t& value) { return real_t{1} + std::log(value); });
                break;
            case DatasetTransform::SQRT:
                features = features.unaryExpr([](const real_t& value) { return std::sqrt(value); });
                break;
        }
    }
}

void dismec::transform_features(SparseFeatures& features, DatasetTransform transform) {
    transform_features_imp(features, transform);
}

void dismec::transform_features(DenseFeatures& features, DatasetTransform transform) {
    transform_features_imp(features, transform);
}

void dismec::hash_sparse_features(SparseFeatures& features, unsigned seed, int buckets, int repeats) {
    // First, count in how many instances each feature is used
    if(!features.isCompressed()) {
        features.makeCompressed();
    }

    std::ranlux24 rng(seed);
    std::uniform_int_distribution<int> mapping(0, buckets - 1);
    Eigen::MatrixXi hash = Eigen::MatrixXi::NullaryExpr(features.cols(), repeats, [&](){
        return mapping(rng);
    });
    DenseRealVector new_row(buckets * repeats);
    SparseFeatures result(features.rows(), buckets * repeats);
    for (int k=0; k < features.rows(); ++k) {
        result.startVec(k);
        new_row.setZero();
        for (SparseFeatures::InnerIterator it(features, k); it; ++it)
        {
            for(int j = 0; j < repeats; ++j) {
                new_row.coeffRef(hash.coeff(it.col(), j) + j * buckets) += it.value();
            }
        }

        // copy the dense vector into the row of the sparse matrix
        for(int i = 0; i < new_row.size(); ++i) {
            if(new_row.coeff(i) > 0) {
                result.insertBack(k, i) = new_row.coeff(i);
            }
        }
    }
    result.finalize();

    // overwrite features
    features = std::move(result);
}

SparseFeatures dismec::shortlist_features(const SparseFeatures& source, const std::vector<long>& shortlist) {
    SparseFeatures new_features(shortlist.size(), source.cols());
    new_features.reserve(2 * source.nonZeros() * double(shortlist.size()) / double(source.rows()));
    long new_row = 0;
    for (auto row : shortlist) {
        new_features.startVec(new_row);
        for (SparseFeatures::InnerIterator it(source, row); it; ++it)
        {
            new_features.insertBack(new_row, it.col()) = it.value();
        }
        ++new_row;
    }
    new_features.finalize();
    return new_features;
}

DenseFeatures dismec::shortlist_features(const DenseFeatures& source, const std::vector<long>& shortlist) {
    DenseFeatures new_features(shortlist.size(), source.cols());
    long new_row = 0;
    for (auto row : shortlist) {
        new_features.row(new_row) = source.row(row);
        ++new_row;
    }
    return new_features;
}


#include "doctest.h"

TEST_CASE("augment sparse") {
    SparseFeatures test(5, 5);
    test.insert(3, 2) = 2.0;
    test.insert(1, 3) = -1.0;
    test.insert(0, 4) = 5.0;
    test.insert(2, 2) = 2.0;
    test.insert(2, 3) = 4.0;

    SparseFeatures extended = augment_features_with_bias(test, 1.0);

    // these checks are easier done using a dense matrix
    DenseFeatures dense_test = test;
    DenseFeatures dense_ext = extended;

    CHECK(dense_test.leftCols(Eigen::fix<5>) == dense_ext.leftCols(Eigen::fix<5>));
    CHECK(dense_ext.col(Eigen::fix<5>) == DenseFeatures::Ones(5, 1));
}

TEST_CASE("sort features") {
    SparseFeatures test(5, 4);
    test.insert(3, 2) = 2.0;
    test.insert(1, 3) = -1.0;
    test.insert(2, 2) = 2.0;
    test.insert(1, 2) = 2.0;
    test.insert(2, 3) = 4.0;
    test.insert(2, 0) = -4.0;

    // column freqs: 0 - 1, 1: 0, 2: 3, 3: 2
    // => reorder: 1, 0, 3, 2

    SparseFeatures expected(5, 4);
    expected.insert(3, 3) = 2.0;
    expected.insert(1, 2) = -1.0;
    expected.insert(2, 3) = 2.0;
    expected.insert(1, 3) = 2.0;
    expected.insert(2, 2) = 4.0;
    expected.insert(2, 1) = -4.0;

    sort_features_by_frequency(test);

    CHECK(test.toDense() == expected.toDense());
}
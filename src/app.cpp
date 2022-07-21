// Copyright (c) 2021, Aalto University, developed by Erik Schultheis
// All rights reserved.
//
// SPDX-License-Identifier: MIT

#include "app.h"
#include "io/xmc.h"
#include "io/slice.h"
#include "data/data.h"
#include <spdlog/spdlog.h>

using namespace dismec;

void DataProcessing::setup_data_args(CLI::App& app) {
    app.add_option("data-file", DataSetFile,
                   "The file from which the data will be loaded.")->required()->check(CLI::ExistingFile);

    app.add_flag("--xmc-one-based-index", OneBasedIndex,
                 "If this flag is given, then we assume that the input dataset in xmc format"
                 " has one-based indexing, i.e. the first label and feature are at index 1  (as opposed to the usual 0)");
    AugmentForBias = app.add_flag("--augment-for-bias", Bias,
                                  "If this flag is given, then all training examples will be augmented with an additional"
                                  "feature of value 1 or the specified value.")->default_val(1.0);
    app.add_flag("--normalize-instances", NormalizeInstances,
                 "If this flag is given, then the feature vectors of all instances are normalized to one.");
    app.add_option("--transform", TransformData, "Apply a transformation to the features of the dataset.")->default_str("identity")
        ->transform(CLI::Transformer(std::map<std::string, DatasetTransform>{
            {"identity",     DatasetTransform::IDENTITY},
            {"log-one-plus", DatasetTransform::LOG_ONE_PLUS},
            {"one-plus-log", DatasetTransform::ONE_PLUS_LOG},
            {"sqrt",         DatasetTransform::SQRT}
        },CLI::ignore_case));

    app.add_option("--label-file", LabelFile, "For SLICE-type datasets, this specifies where the labels can be found")->check(CLI::ExistingFile);


    auto* hash_option = app.add_flag("--hash-features", "If this Flag is given, then feature hashing is performed.");
    auto* bucket_option = app.add_option("--hash-buckets", HashBuckets, "Number of buckets for each hash function when feature hashing is enabled.")
        ->needs(hash_option)->check(CLI::PositiveNumber);
    app.add_option("--hash-repeat", HashRepeats, "Number of hash functions to use for feature hashing.")
        ->needs(hash_option)->default_val(32)->check(CLI::PositiveNumber);
    app.add_option("--hash-seed", HashSeed, "Seed to use when feature hashing.")
        ->needs(hash_option)->default_val(42);
    hash_option->needs(bucket_option);
}

std::shared_ptr<MultiLabelData> DataProcessing::load(int verbose) {
    if(verbose >= 0) {
        spdlog::info("Loading training data from file '{}'", DataSetFile);
    }
    auto data = std::make_shared<MultiLabelData>([&]() {
        if(LabelFile.empty()) {
            return read_xmc_dataset(DataSetFile, OneBasedIndex ? io::IndexMode::ONE_BASED : io::IndexMode::ZERO_BASED);
        } else {
            return io::read_slice_dataset(DataSetFile, LabelFile);
        }
    } ());

    if(HashBuckets > 0) {
        if(!data->get_features()->is_sparse()) {
            spdlog::error("Feature hashing is currently only implemented for sparse features.");
        }
        if(verbose >= 0) {
            spdlog::info("Hashing features");
        }
        hash_sparse_features(data->edit_features()->sparse(), HashSeed, HashBuckets, HashRepeats);
    }

    if(TransformData != DatasetTransform::IDENTITY) {
        if(verbose >= 0)
            spdlog::info("Applying data transformation");
        transform_features(*data, TransformData);
    }

    if(NormalizeInstances) {
        if(verbose >= 0)
            spdlog::info("Normalizing instances.");
        normalize_instances(*data);
    }

    if(!AugmentForBias->empty()) {
        if(verbose >= 0)
            spdlog::info("Appending bias features with value {}", Bias);
        augment_features_with_bias(*data, Bias);
    }

    if(verbose >= 0) {
        if(data->get_features()->is_sparse()) {
            double total = data->num_features() * data->num_examples();
            auto nnz = data->get_features()->sparse().nonZeros();
            spdlog::info("Processed feature matrix has {} rows and {} columns. Contains {} non-zeros ({:.3} %)", data->num_examples(),
                         data->num_features(), nnz, 100.0 * (nnz / total));
        } else {
            spdlog::info("Processed feature matrix has {} rows and {} columns", data->num_examples(),
                         data->num_features());
        }
    }

    return data;
}

bool DataProcessing::augment_for_bias() const {
    return AugmentForBias->count() > 0;
}


// Copyright (c) 2021, Aalto University, developed by Erik Schultheis
// All rights reserved.
//
// SPDX-License-Identifier: MIT

#ifndef DISMEC_SRC_APP_H
#define DISMEC_SRC_APP_H

#include <string>
#include <CLI/CLI.hpp>
#include "fwd.h"
#include "data/transform.h"

// common code for train and prediction executables
namespace dismec {
    class DataProcessing {
    public:
        void setup_data_args(CLI::App& app);
        std::shared_ptr<MultiLabelData> load(int verbose);
        [[nodiscard]] bool augment_for_bias() const;
    private:
        /// The file from which the dataset should be read.
        std::string DataSetFile;
        std::string LabelFile;
        bool OneBasedIndex = false;
        bool NormalizeInstances = false;
        DatasetTransform TransformData = DatasetTransform::IDENTITY;
        CLI::Option* AugmentForBias = nullptr;
        real_t Bias = 0;

        // Feature Hashing
        int HashBuckets = -1;
        int HashRepeats = -1;
        unsigned HashSeed = 42;
    };
}
#endif //DISMEC_SRC_APP_H

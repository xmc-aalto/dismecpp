// Copyright (c) 2021, Aalto University, developed by Erik Schultheis
// All rights reserved.
//
// SPDX-License-Identifier: MIT

#include "training/initializer.h"
#include "data/types.h"

using namespace init;

namespace init {
    class ZeroInitializer : public WeightsInitializer {
    public:
        void get_initial_weight(label_id_t label_id, Eigen::Ref<DenseRealVector> target, objective::Objective& objective) override {
            target.setZero();
        }
    };

    class ZeroInitializationStrategy : public WeightInitializationStrategy {
    public:
        [[nodiscard]] std::unique_ptr<WeightsInitializer> make_initializer(
                const std::shared_ptr<const GenericFeatureMatrix>& features) const override;
    };
}

std::unique_ptr<WeightsInitializer> ZeroInitializationStrategy::make_initializer(
        const std::shared_ptr<const GenericFeatureMatrix>& features) const {
    return std::make_unique<ZeroInitializer>();
}

std::shared_ptr<WeightInitializationStrategy> init::create_zero_initializer() {
    return std::make_shared<ZeroInitializationStrategy>();
}

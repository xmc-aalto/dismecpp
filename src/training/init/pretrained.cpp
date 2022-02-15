// Copyright (c) 2021, Aalto University, developed by Erik Schultheis
// All rights reserved.
//
// SPDX-License-Identifier: MIT

#include "training/initializer.h"
#include "data/types.h"
#include "model/model.h"

using namespace init;

namespace init {
    class PreTrainedInitializer : public WeightsInitializer {
    public:
        explicit PreTrainedInitializer(std::shared_ptr<const model::Model> pre_trained) :
            m_PreTrainedWeights(std::move(pre_trained))
        {
            if(!m_PreTrainedWeights) {
                throw std::logic_error("pre trained model is <null>");
            }
        }
        void get_initial_weight(label_id_t label_id, Eigen::Ref<DenseRealVector> target, objective::Objective& objective) override {
            m_PreTrainedWeights->get_weights_for_label(label_id, target);
        }
    private:
        std::shared_ptr<const model::Model> m_PreTrainedWeights;
    };

    class PreTrainedInitializationStrategy : public WeightInitializationStrategy {
    public:
        PreTrainedInitializationStrategy(std::shared_ptr<const model::Model> pre_trained);
        [[nodiscard]] std::unique_ptr<WeightsInitializer> make_initializer(const std::shared_ptr<const GenericFeatureMatrix>& features) const override;
    private:
        std::shared_ptr<const model::Model> m_PreTrained;
    };
}

PreTrainedInitializationStrategy::PreTrainedInitializationStrategy(std::shared_ptr<const model::Model> pre_trained) :
    m_PreTrained(std::move(pre_trained)) {

}
std::unique_ptr<WeightsInitializer> PreTrainedInitializationStrategy::make_initializer(const std::shared_ptr<const GenericFeatureMatrix>& features) const {
    return std::make_unique<PreTrainedInitializer>(m_PreTrained);
}

std::shared_ptr<WeightInitializationStrategy> init::create_pretrained_initializer(std::shared_ptr<model::Model> model) {
    return std::make_shared<PreTrainedInitializationStrategy>(std::move(model));
}
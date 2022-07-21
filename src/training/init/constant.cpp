// Copyright (c) 2021, Aalto University, developed by Erik Schultheis
// All rights reserved.
//
// SPDX-License-Identifier: MIT

#include "training/initializer.h"
#include "data/types.h"

using namespace dismec::init;

namespace dismec::init {
    class ConstantInitializer : public WeightsInitializer {
    public:
        explicit ConstantInitializer(std::shared_ptr<const DenseRealVector> vec) : m_InitVector(std::move(vec)) {
            if (!m_InitVector) {
                throw std::logic_error("Initial vector is <null>");
            }
        }

        void get_initial_weight(label_id_t label_id, Eigen::Ref <DenseRealVector> target,
                                objective::Objective &objective) override {
            target = *m_InitVector;
        }

    private:
        std::shared_ptr<const DenseRealVector> m_InitVector;
    };

    /*!
     * \brief An initialization strategy that sets the weight vector to a given constant.
     * \details We create NUMA-local copies of the given vector, and provide pointers to these to the resulting
     * `ConstantInitializer` objects.
     */
    class ConstantInitializationStrategy : public WeightInitializationStrategy {
    public:
        explicit ConstantInitializationStrategy(DenseRealVector vec);

        [[nodiscard]] std::unique_ptr <WeightsInitializer>
        make_initializer(const std::shared_ptr<const GenericFeatureMatrix>& features) const override;

    private:
        parallel::NUMAReplicator <DenseRealVector> m_InitVector;
    };

}

ConstantInitializationStrategy::ConstantInitializationStrategy(DenseRealVector vec) :
        m_InitVector(std::make_shared<DenseRealVector>(std::move(vec))) {
}

std::unique_ptr<WeightsInitializer> ConstantInitializationStrategy::make_initializer(
        const std::shared_ptr<const GenericFeatureMatrix>& features) const {
    return std::make_unique<ConstantInitializer>(m_InitVector.get_local() );
}

std::shared_ptr<WeightInitializationStrategy> dismec::init::create_constant_initializer(DenseRealVector vec) {
    return std::make_shared<ConstantInitializationStrategy>(std::move(vec));
}
// Copyright (c) 2021, Aalto University, developed by Erik Schultheis
// All rights reserved.
//
// SPDX-License-Identifier: MIT

#include "training/initializer.h"
#include "data/types.h"
#include "io/numpy.h"


using namespace dismec::init;
namespace {
    using WeightMatrix = Eigen::Matrix<real_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
}

namespace dismec::init {
    class NumpyInitializer : public WeightsInitializer {
    public:
        explicit NumpyInitializer(std::shared_ptr<const WeightMatrix> weights,  std::shared_ptr<const DenseRealVector> biases) :
            m_PreTrainedWeights(std::move(weights)),
            m_Biases(std::move(biases))
        {
        }
        void get_initial_weight(label_id_t label_id, Eigen::Ref<DenseRealVector> target, objective::Objective& objective) override {
            if(m_Biases) {
                target.head(target.size() - 1) = m_PreTrainedWeights->row(label_id.to_index());
                //spdlog::info("BIAS: {}", m_Biases->coeff(label_id.to_index()));
                target.tail(1).coeffRef(0) = m_Biases->coeff(label_id.to_index());
            } else {
                target = m_PreTrainedWeights->row(label_id.to_index());
            }
        }
    private:
        std::shared_ptr<const WeightMatrix> m_PreTrainedWeights;
        std::shared_ptr<const DenseRealVector> m_Biases;
    };

    class NumpyInitializationStrategy : public WeightInitializationStrategy {
    public:
        explicit NumpyInitializationStrategy(std::shared_ptr<const WeightMatrix>, std::shared_ptr<const DenseRealVector> biases);
        [[nodiscard]] std::unique_ptr<WeightsInitializer> make_initializer(const std::shared_ptr<const GenericFeatureMatrix>& features) const override;
    private:
        std::shared_ptr<const WeightMatrix> m_WeightMatrix;
        std::shared_ptr<const DenseRealVector> m_BiasVector;
    };
}

NumpyInitializationStrategy::NumpyInitializationStrategy(
    std::shared_ptr<const WeightMatrix> weights,
    std::shared_ptr<const DenseRealVector> biases) :
    m_WeightMatrix(std::move(weights)), m_BiasVector(std::move(biases)) {

}
std::unique_ptr<WeightsInitializer> NumpyInitializationStrategy::make_initializer(const std::shared_ptr<const GenericFeatureMatrix>& features) const {
    return std::make_unique<NumpyInitializer>(m_WeightMatrix, m_BiasVector);
}

std::shared_ptr<WeightInitializationStrategy> dismec::init::create_numpy_initializer(const std::filesystem::path& weight_file,
                                                                             std::optional<std::filesystem::path> bias_file) {
    auto weights = std::make_shared<WeightMatrix>(io::load_matrix_from_npy(weight_file));
    spdlog::info("Loaded weight matrix from {}: {} x {}", weight_file.string(), weights->rows(), weights->cols());
    std::shared_ptr<const DenseRealVector> biases = nullptr;
    if(bias_file) {
        biases = std::make_shared<const DenseRealVector>(io::load_matrix_from_npy(bias_file->string()));
    }
    return std::make_shared<NumpyInitializationStrategy>(weights, biases);
}
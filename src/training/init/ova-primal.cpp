// Copyright (c) 2021, Aalto University, developed by Erik Schultheis
// All rights reserved.
//
// SPDX-License-Identifier: MIT

#include "training/initializer.h"
#include "solver/newton.h"
#include "objective/linear.h"
#include "data/data.h"

#include <spdlog/spdlog.h>

using namespace init;

std::shared_ptr<WeightInitializationStrategy> init::create_ova_primal_initializer(
        const std::shared_ptr<DatasetBase>& data, RegularizerSpec regularizer, LossType loss) {
    auto minimizer = std::make_unique<solvers::NewtonWithLineSearch>(data->num_features());
    auto reg = std::visit([](auto&& config){ return make_regularizer(config); }, regularizer);
    auto loss_fn = make_loss(loss, data->get_features(), std::move(reg));
    dynamic_cast<objective::LinearClassifierBase&>(*loss_fn).get_label_ref().fill(-1);
    //minimizer->set_epsilon(0.01 / data->num_examples());

    DenseRealVector target(data->num_features());
    target.setZero();
    spdlog::info("Starting to calculate OVA-Primal init vector");
    auto result = minimizer->minimize(*loss_fn, target);

    spdlog::info("OVA-Primal init vector has been calculated in {} ms. Loss {} -> {}",
                 result.Duration.count(), result.InitialValue, result.FinalValue);


    return create_constant_initializer(std::move(target));
}

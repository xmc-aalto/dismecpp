// Copyright (c) 2021, Aalto University, developed by Erik Schultheis
// All rights reserved.
//
// SPDX-License-Identifier: MIT

#ifndef DISMEC_SRC_OBJECTIVE_REGULARIZERS_H
#define DISMEC_SRC_OBJECTIVE_REGULARIZERS_H

#include <memory>

using real_t = float;

namespace dismec::objective {
    class Objective;
    struct SquaredNormConfig { real_t Strength; bool IgnoreBias; };
    struct HuberConfig { real_t Strength; real_t Epsilon; bool IgnoreBias; };
    struct ElasticConfig { real_t Strength; real_t Epsilon; real_t Interpolation; bool IgnoreBias; };

    std::unique_ptr<Objective> make_regularizer(const SquaredNormConfig& config);
    std::unique_ptr<Objective> make_regularizer(const HuberConfig& config);
    std::unique_ptr<Objective> make_regularizer(const ElasticConfig& config);
}

#endif //DISMEC_SRC_OBJECTIVE_REGULARIZERS_H

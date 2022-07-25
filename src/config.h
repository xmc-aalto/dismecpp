// Copyright (c) 2022, Aalto University, developed by Erik Schultheis
// All rights reserved.
//
// SPDX-License-Identifier: MIT

/*! \file config.h
 * \brief Defines configuration variables.
 * \details If you want to change the defaults of the program, this is the header in which these numbers are defined.
 *
*/

#ifndef DISMEC_SRC_CONFIG_H
#define DISMEC_SRC_CONFIG_H

namespace dismec {
    /// The default type for floating point values.
    using real_t = float;

    /// The minimum upper bound for the number of CG iterations. If the problem has less than this many dimensions,
    /// the upper-bound for the number of CG iterations is still given by this number.
    /// TODO is it even sensible to define something like this?
    constexpr const long CG_MIN_ITER_BOUND = 5;

    /// The default value for the tolerance parameter of the conjugate gradient optimization
    constexpr const real_t CG_DEFAULT_EPSILON = 0.5;

    /// If the time needed per chunk of work is less than this, we display a warning
    constexpr const int MIN_TIME_PER_CHUNK_MS = 5;

    namespace parallel {
        /// Load balancing cost for placing a thread on a core
        constexpr const int COST_PLACE_THREAD = 10;

        /// Load balancing cost for placing a thread on a SMT shared core
        constexpr const int COST_PLACE_HYPER_THREAD = 5;
    }

    /// Default chunk size for predicting scores
    constexpr const int PREDICTION_RUN_CHUNK_SIZE = 1024;

    /// Default chunk size for calculating metrics
    constexpr const int PREDICTION_METRICS_CHUNK_SIZE = 4096;
}

#endif //DISMEC_SRC_CONFIG_H

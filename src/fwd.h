// Copyright (c) 2022, Aalto University, developed by Erik Schultheis
// All rights reserved.
//
// SPDX-License-Identifier: MIT

/*! \file fwd.h
 * \brief Forward-declares types.
*/
#ifndef DISMEC_SRC_FWD_H
#define DISMEC_SRC_FWD_H

/*!
 * \brief Main namespace in which all types, classes, and functions are defined.
 */
namespace dismec
{
    class DatasetBase;
    class MultiLabelData;
    class label_id_t;

    class WeightingScheme;
    class TrainingSpec;
    class TrainingStatsGatherer;
    class ResultStatsGatherer;

    class HashVector;
    class HyperParameters;

    namespace model {
        class Model;
        struct PartialModelSpec;
    }

    namespace objective {
        class Objective;
    }

    namespace solvers {
        class Minimizer;
        struct MinimizationResult;
    }

    namespace init {
        class WeightsInitializer;
        class WeightInitializationStrategy;
    }

    namespace postproc {
        class PostProcessor;
        class PostProcessFactory;
    }

    namespace parallel {
        class ParallelRunner;
        class thread_id_t;
    }
}

#endif //DISMEC_SRC_FWD_H

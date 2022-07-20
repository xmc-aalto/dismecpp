// Copyright (c) 2021, Aalto University, developed by Erik Schultheis
// All rights reserved.
//
// SPDX-License-Identifier: MIT

#ifndef DISMEC_POSTPROC_H
#define DISMEC_POSTPROC_H

#include "matrix_types.h"
#include "fwd.h"
#include <memory>
#include "stats/tracked.h"

namespace dismec::postproc {
    class PostProcessFactory;
    using FactoryPtr = std::shared_ptr<PostProcessFactory>;
    class PostProcessor : public stats::Tracked {
    public:
        virtual ~PostProcessor() = default;

        /// Apply post-processing for the `weight_vector` corresponding to the label `label_id`.
        virtual void process(label_id_t label_id, Eigen::Ref<DenseRealVector> weight_vector, solvers::MinimizationResult& result) = 0;
    };

    class PostProcessFactory {
    public:
        virtual ~PostProcessFactory() = default;

        [[nodiscard]] virtual std::unique_ptr<PostProcessor> make_processor(const std::shared_ptr<objective::Objective>& objective) const = 0;
    };

    FactoryPtr create_identity();
    FactoryPtr create_culling(real_t eps);
    FactoryPtr create_sparsify(real_t tolerance);
    FactoryPtr create_combined(std::vector<FactoryPtr> processor);
    FactoryPtr create_reordering(Eigen::PermutationMatrix<Eigen::Dynamic, Eigen::Dynamic, int> ordering);
}


#endif //DISMEC_POSTPROC_H

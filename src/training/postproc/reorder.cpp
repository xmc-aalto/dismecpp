// Copyright (c) 2021, Aalto University, developed by Erik Schultheis
// All rights reserved.
//
// SPDX-License-Identifier: MIT

#include <utility>
#include "training/postproc/generic.h"
#include "data/types.h"

namespace dismec::postproc {
    class ReorderPostProc : public PostProcessor {
    public:
        ReorderPostProc(const std::shared_ptr<objective::Objective>& objective,
                        Eigen::PermutationMatrix<Eigen::Dynamic, Eigen::Dynamic, int> order) : m_Ordering(std::move(order)) {
        }

        void process(label_id_t label_id, Eigen::Ref<DenseRealVector> weight_vector, solvers::MinimizationResult& result) override {
            weight_vector.applyOnTheLeft(m_Ordering);
        }
    private:
        Eigen::PermutationMatrix<Eigen::Dynamic, Eigen::Dynamic, int> m_Ordering;
    };
}

std::shared_ptr<dismec::postproc::PostProcessFactory> dismec::postproc::create_reordering(Eigen::PermutationMatrix<Eigen::Dynamic, Eigen::Dynamic, int> ordering) {
    return std::make_shared<GenericPostProcFactory<ReorderPostProc, Eigen::PermutationMatrix<Eigen::Dynamic, Eigen::Dynamic, int>>>(ordering);
}
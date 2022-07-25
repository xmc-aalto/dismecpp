// Copyright (c) 2021, Aalto University, developed by Erik Schultheis
// All rights reserved.
//
// SPDX-License-Identifier: MIT

#include "postproc.h"
#include "data/types.h"
#include "objective/objective.h"
#include "utils/hash_vector.h"
#include "spdlog/spdlog.h"
#include "postproc/generic.h"

namespace dismec::postproc {
    struct IdentityPostProc : public PostProcessor {
        explicit IdentityPostProc(const std::shared_ptr<objective::Objective>&) {}
        void process([[maybe_unused]] label_id_t label_id,
                     [[maybe_unused]] Eigen::Ref<DenseRealVector> weight_vector,
                     [[maybe_unused]] solvers::MinimizationResult& result) override {};
    };

    class CullingPostProcessor : public PostProcessor {
    public:
        CullingPostProcessor(const std::shared_ptr<objective::Objective>& objective, real_t eps);
        void process(label_id_t label_id, Eigen::Ref<DenseRealVector> weight_vector, solvers::MinimizationResult& result) override;
    private:
        real_t m_Epsilon;
    };

    void CullingPostProcessor::process([[maybe_unused]] label_id_t label_id,
                                       Eigen::Ref<DenseRealVector> weight_vector,
                                       [[maybe_unused]] solvers::MinimizationResult& result) {
        for(long i = 0; i < weight_vector.size(); ++i) {
            real_t& w = weight_vector.coeffRef(i);
            if(abs(w) <= m_Epsilon) {
                w = real_t{0};
            }
        }
    }

    CullingPostProcessor::CullingPostProcessor([[maybe_unused]] const std::shared_ptr<objective::Objective>& objective,
                                               real_t eps) : m_Epsilon(eps) {
        if(eps < 0) {
            throw std::invalid_argument("Epsilon has to be positive");
        }
    }
}

using dismec::postproc::PostProcessFactory;

std::shared_ptr<PostProcessFactory> dismec::postproc::create_identity() {
    return std::make_shared<GenericPostProcFactory<IdentityPostProc>>();
}

std::shared_ptr<PostProcessFactory> dismec::postproc::create_culling(real_t eps) {
    return std::make_shared<GenericPostProcFactory<CullingPostProcessor, real_t>>( eps );
}


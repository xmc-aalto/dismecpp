// Copyright (c) 2021, Aalto University, developed by Erik Schultheis
// All rights reserved.
//
// SPDX-License-Identifier: MIT

#include "training/postproc.h"

namespace postproc {
    class CombinePostProcessor : public PostProcessor {
    public:
        explicit CombinePostProcessor(std::vector<std::unique_ptr<PostProcessor>> children);
        void process(label_id_t label_id, DenseRealVector& weight_vector, solvers::MinimizationResult& result) override;
    private:
        std::vector<std::unique_ptr<PostProcessor>> m_Children;
    };

    CombinePostProcessor::CombinePostProcessor(std::vector<std::unique_ptr<PostProcessor>> children) :
            m_Children(std::move(children)) {

    }

    void CombinePostProcessor::process(label_id_t label_id,
                                       DenseRealVector& weight_vector,
                                       solvers::MinimizationResult& result) {
        for(auto& child : m_Children) {
            child->process(label_id, weight_vector, result);
        }
    }


    class CombinedFactory : public PostProcessFactory {
    public:
        explicit CombinedFactory(std::vector<std::shared_ptr<PostProcessFactory>> children) :
                m_Children(std::move(children)) {

        }

        [[nodiscard]] std::unique_ptr<PostProcessor>
        make_processor(const std::shared_ptr<objective::Objective>& objective) const override {
            std::vector<std::unique_ptr<PostProcessor>> children;
            children.reserve(m_Children.size());
            std::transform(begin(m_Children), end(m_Children), std::back_inserter(children),
                           [&](auto&& factory) {
                               return factory->make_processor(objective);
                           });
            return std::make_unique<CombinePostProcessor>(std::move(children));
        }

        std::vector<std::shared_ptr<PostProcessFactory>> m_Children;
    };
}

std::shared_ptr<postproc::PostProcessFactory> postproc::create_combined(std::vector<std::shared_ptr<PostProcessFactory>> children) {
    return std::make_shared<CombinedFactory>( std::move(children) );
}

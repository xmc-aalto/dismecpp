// Copyright (c) 2021, Aalto University, developed by Erik Schultheis
// All rights reserved.
//
// SPDX-License-Identifier: MIT

#ifndef DISMEC_GENERIC_H
#define DISMEC_GENERIC_H

#include "training/postproc.h"

namespace dismec::postproc {
    template<class T, class... Args>
    class GenericPostProcFactory : public PostProcessFactory {
    public:
        explicit GenericPostProcFactory(Args... args) : m_Args( std::move(args)...) {
        }

        [[nodiscard]] std::unique_ptr<PostProcessor> make_processor(const std::shared_ptr<objective::Objective>& objective) const override {
            return std::apply([&](const auto&... args){ return std::make_unique<T>(objective, args...); }, m_Args);
        }

        std::tuple<Args...> m_Args;
    };
}

#endif //DISMEC_GENERIC_H

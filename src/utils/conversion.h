// Copyright (c) 2021, Aalto University, developed by Erik Schultheis
// All rights reserved.
//
// SPDX-License-Identifier: MIT

#ifndef DISMEC_SRC_UTILS_CONVERSION_H
#define DISMEC_SRC_UTILS_CONVERSION_H

#include "throw_error.h"

namespace dismec {
    /// Convert the given value to `long`, throwing an error if the conversion is not possible.
    template<class T>
    constexpr long to_long(T value) {
        static_assert(std::is_integral_v<T>, "Can only convert between integral types");
        // I don't think that the if constexpr would be needed here, strictly speaking,
        // but handling the unsigned case separately prevent -Wsign-compare complaints.
        if constexpr(std::is_signed_v<T>) {
            if (value < std::numeric_limits<long>::max()) {
                return static_cast<long>(value);
            }
        } else {
            if (value < static_cast<unsigned long>(std::numeric_limits<long>::max())) {
                return static_cast<long>(value);
            }
        }
        THROW_EXCEPTION(std::range_error, "Value {} cannot be represented as long.", value);
    }

    /// Gets the `sizeof` of a type as a signed integer
    template<class T>
    constexpr std::ptrdiff_t calc_ssizeof() {
        return static_cast<std::ptrdiff_t>(sizeof(T));
    }

    /// Signed size of type `T`
    template<class T>
    constexpr std::ptrdiff_t ssizeof = calc_ssizeof<T>();

    /// signed size free function. Taken from https://en.cppreference.com/w/cpp/iterator/size
    template <class C>
    constexpr auto ssize(const C& c) -> std::common_type_t<std::ptrdiff_t, std::make_signed_t<decltype(c.size())>> {
        using R = std::common_type_t<std::ptrdiff_t, std::make_signed_t<decltype(c.size())>>;
        return static_cast<R>(c.size());
    }
}

#endif //DISMEC_SRC_UTILS_CONVERSION_H

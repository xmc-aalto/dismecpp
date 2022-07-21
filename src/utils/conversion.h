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
    long to_long(T value) {
        static_assert(std::is_integral_v<T>, "Can only convert between integral types");
        if(value < std::numeric_limits<long>::max()) {
            return static_cast<long>(value);
        }
        THROW_EXCEPTION(std::range_error, "Value {} cannot be represented as long.", value);
    }
}

#endif //DISMEC_SRC_UTILS_CONVERSION_H

// Copyright (c) 2021, Aalto University, developed by Erik Schultheis
// All rights reserved.
//
// SPDX-License-Identifier: MIT

#ifndef DISMEC_OPAQUE_INT_H
#define DISMEC_OPAQUE_INT_H

#include <cstdint>

template<class Tag, class T = std::int_fast32_t>
class opaque_int_type {
public:
    static_assert(std::is_integral_v<T>, "T needs to be an integral type");
    constexpr explicit opaque_int_type(T v) noexcept : m_Value(v) {}
    [[nodiscard]] T to_index() const { return m_Value; }

private:
    T m_Value;
};

#endif //DISMEC_OPAQUE_INT_H

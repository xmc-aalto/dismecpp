// Copyright (c) 2021, Aalto University, developed by Erik Schultheis
// All rights reserved.
//
// SPDX-License-Identifier: MIT

#ifndef DISMEC_OPAQUE_INT_H
#define DISMEC_OPAQUE_INT_H

#include <cstdint>
#include <type_traits>
#include <iosfwd>

namespace dismec {
    /*!
     * \brief  An integer-like type that represents categorical values.
     * \details The `opaque_int_type` differs from a normal integer in that it is not implicitly convertible
     * to any other type, and does not support arithmetic operators unless the user defines them. It does, however,
     * define comparison operators so that its values can be ordered. It is intended to be used to represent categorical
     * values.
     * \tparam Tag A tag type that allows to define different, non-interacting `opaque_int_type` types
     * \tparam T The integer type that is used to store the actual value.
     */
    template<class Tag, class T = std::int_fast32_t>
    class opaque_int_type {
    public:
        static_assert(std::is_integral_v<T>, "T needs to be an integral type");

        ///! Explicit constructor from an underlying int.
        constexpr explicit opaque_int_type(T v) noexcept: m_Value(v) {}

        ///! Explicitly convert to an integer.
        [[nodiscard]] constexpr T to_index() const { return m_Value; }

    protected:
        T m_Value;
    };

    // define comparison operators
    template<class Tag, class T>
    inline constexpr bool operator==(opaque_int_type<Tag, T> a, opaque_int_type<Tag, T> b) {
        return a.to_index() == b.to_index();
    }

    template<class Tag, class T>
    inline constexpr bool operator!=(opaque_int_type<Tag, T> a, opaque_int_type<Tag, T> b) {
        return a.to_index() != b.to_index();
    }

    template<class Tag, class T>
    inline constexpr bool operator<=(opaque_int_type<Tag, T> a, opaque_int_type<Tag, T> b) {
        return a.to_index() <= b.to_index();
    }

    template<class Tag, class T>
    inline constexpr bool operator<(opaque_int_type<Tag, T> a, opaque_int_type<Tag, T> b) {
        return a.to_index() < b.to_index();
    }

    template<class Tag, class T>
    inline constexpr bool operator>(opaque_int_type<Tag, T> a, opaque_int_type<Tag, T> b) {
        return a.to_index() > b.to_index();
    }

    template<class Tag, class T>
    inline constexpr bool operator>=(opaque_int_type<Tag, T> a, opaque_int_type<Tag, T> b) {
        return a.to_index() >= b.to_index();
    }

    template<class Tag, class T>
    std::ostream& operator<<(std::ostream& stream, opaque_int_type<Tag, T> a) {
        return stream << a.to_index();
    }
}

#endif //DISMEC_OPAQUE_INT_H

// Copyright (c) 2021, Aalto University, developed by Erik Schultheis
// All rights reserved.
//
// SPDX-License-Identifier: MIT

#ifndef DISMEC_THROW_ERROR_H
#define DISMEC_THROW_ERROR_H

#include "spdlog/fmt/fmt.h"

/// this is a utility macro that wraps the formatting
/// of an error message and throwing of `std::runtime_error`
/// into an immediately-executed lambda that is marked with the
/// cold attribute. This can help the optimizer.
#define THROW_EXCEPTION(exception_type, ...) [&]() __attribute__((cold, noreturn)) { \
    throw exception_type( fmt::format(__VA_ARGS__) ); \
}();

/// This macro adds a check that two values are equal. This check is added both in debug and release mode. If
/// the check fails, a std::invalid_argument exception is raised with the given message. The message should contain
/// two placeholders `{}` which will be filled with the two values.
#define ALWAYS_ASSERT_EQUAL(x, y, msg) \
{                                      \
auto v1=(x);                           \
auto v2=(y);                           \
if(v1 != v2) {                         \
    THROW_EXCEPTION(std::invalid_argument, msg, v1, v2); \
}}
#endif //DISMEC_THROW_ERROR_H

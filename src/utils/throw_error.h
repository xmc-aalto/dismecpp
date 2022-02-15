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

#endif //DISMEC_THROW_ERROR_H

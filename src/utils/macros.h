// Copyright (c) 2022, Aalto University, developed by Erik Schultheis
// All rights reserved.
//
// SPDX-License-Identifier: MIT

#ifndef DISMEC_SRC_MODEL_MACROS_H
#define DISMEC_SRC_MODEL_MACROS_H

#include <boost/predef.h>

// defining diagnostic helper macros. To disable a diagnostic, use `DIAGNOSTIC_IGNORE`.
// To disable only for GCC or only for Clang, use the respective suffixed macros
#if BOOST_COMP_GNUC
#define DO_PRAGMA(x) _Pragma (#x)
#define DIAGNOSTIC_PUSH _Pragma("GCC diagnostic push")
#define DIAGNOSTIC_POP _Pragma("GCC diagnostic pop")
#define DIAGNOSTIC_IGNORE(X) DO_PRAGMA(GCC diagnostic ignored X)
#define DIAGNOSTIC_IGNORE_GCC(X) DIAGNOSTIC_IGNORE(X)
#define DIAGNOSTIC_IGNORE_CLANG(X)
#elif BOOST_COMP_CLANG
#define DO_PRAGMA(x) _Pragma (#x)
#define DIAGNOSTIC_PUSH _Pragma("clang diagnostic push")
#define DIAGNOSTIC_POP _Pragma("clang diagnostic pop")
#define DIAGNOSTIC_IGNORE(X) DO_PRAGMA(clang diagnostic ignored X)
#define DIAGNOSTIC_IGNORE_GCC(X)
#define DIAGNOSTIC_IGNORE_CLANG(X) DIAGNOSTIC_IGNORE(X)
#endif

#endif //DISMEC_SRC_MODEL_MACROS_H

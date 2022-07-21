// Copyright (c) 2022, Aalto University, developed by Erik Schultheis
// All rights reserved.
//
// SPDX-License-Identifier: MIT

#ifndef DISMEC_SRC_PARALLEL_THREAD_ID_H
#define DISMEC_SRC_PARALLEL_THREAD_ID_H

#include "utils/opaque_int.h"

namespace dismec::parallel {
    /*!
     * \brief Strong typedef for an int to signify a thread id.
     * \details This value represents an id for a thread. These IDs
     * are only unique within a single run, and can be used (with the
     * `to_index()` method to manage thread-local data.
     * \internal Implemented as a subclass of `opaque_int_type` instead of a typedef
     * because this makes forward declarations easier.
     */
    class thread_id_t : public opaque_int_type<thread_id_t> {
        using opaque_int_type::opaque_int_type;
    };

    /*!
     * \brief Strong typedef for an int to signify a numa domain.
     */
    class numa_node_id_t : public opaque_int_type<numa_node_id_t> {
        using opaque_int_type::opaque_int_type;
    };

    /*!
     * \brief Strong typedef for an int to signify a (core of a) cpu
     */
    class cpu_id_t : public opaque_int_type<cpu_id_t> {
        using opaque_int_type::opaque_int_type;
    };
}

#endif //DISMEC_SRC_PARALLEL_THREAD_ID_H

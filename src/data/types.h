// Copyright (c) 2021, Aalto University, developed by Erik Schultheis
// All rights reserved.
//
// SPDX-License-Identifier: MIT

#ifndef DISMEC_TYPES_H
#define DISMEC_TYPES_H

#include "utils/opaque_int.h"

namespace dismec
{
    // forward declarations
    class DatasetBase;

    /*!
     * \brief Strong typedef for an int to signify a label id.
     * \details This value represents an id for a label.
     */
    class label_id_t : public opaque_int_type<label_id_t, std::int_fast32_t> {
    public:
        using opaque_int_type::opaque_int_type;

        inline label_id_t& operator++() {
            ++m_Value;
            return *this;
        }
    };

    inline std::ptrdiff_t operator-(label_id_t a, label_id_t b) {
        return a.to_index() - b.to_index();
    }

    inline label_id_t operator+(label_id_t a, std::ptrdiff_t b) {
        return label_id_t{a.to_index() + b};
    }
}

#endif //DISMEC_TYPES_H

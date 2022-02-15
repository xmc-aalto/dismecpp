// Copyright (c) 2021, Aalto University, developed by Erik Schultheis
// All rights reserved.
//
// SPDX-License-Identifier: MIT

#ifndef DISMEC_TYPES_H
#define DISMEC_TYPES_H

#include <cstdint>

// forward declarations
class DatasetBase;

/*!
 * \brief Strong typedef for an int to signify a label id.
 * \details This value represents an id for a label.
 */
class label_id_t {
public:
    explicit label_id_t(std::int_fast32_t v) : m_Value(v) {}
    [[nodiscard]] std::int_fast32_t to_index() const { return m_Value; }

    inline label_id_t& operator++() {
        ++m_Value;
        return *this;
    }

private:
    std::int_fast32_t m_Value;
};

inline bool operator==(label_id_t a, label_id_t b) {
    return a.to_index() == b.to_index();
}

inline bool operator!=(label_id_t a, label_id_t b) {
    return a.to_index() != b.to_index();
}

inline bool operator<=(label_id_t a, label_id_t b) {
    return a.to_index() <= b.to_index();
}

inline bool operator<(label_id_t a, label_id_t b) {
    return a.to_index() < b.to_index();
}

inline bool operator>(label_id_t a, label_id_t b) {
    return a.to_index() > b.to_index();
}

inline bool operator>=(label_id_t a, label_id_t b) {
    return a.to_index() >= b.to_index();
}

inline std::ptrdiff_t operator-(label_id_t a, label_id_t b) {
    return a.to_index() - b.to_index();
}

inline label_id_t operator+(label_id_t a, std::ptrdiff_t b) {
    return label_id_t{a.to_index() + b};
}

#endif //DISMEC_TYPES_H

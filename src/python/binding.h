// Copyright (c) 2021, Aalto University, developed by Erik Schultheis
// All rights reserved.
//
// SPDX-License-Identifier: MIT

#ifndef DISMEC_SRC_UTILS_BINDING_H
#define DISMEC_SRC_UTILS_BINDING_H

#include "pybind11/pybind11.h"
#include "pybind11/eigen.h"
#include "pybind11/stl.h"
#include "pybind11/stl/filesystem.h"
#include "fwd.h"

#include <memory>

// Utilities for pybind11 binding
/*!
 * \brief Utility class used to wrap all objects we provide to python
 * \tparam T
 * \details In order to enable shared ownership of objects between python and C++, we put them into a `shared_ptr`.
 * One option would be to specify the `HolderType` in the pybind bindings to use a `shared_ptr`. Instead, we have
 * introduced this wrapper class, for the following reasons:
 *  1) Converting constructor: The wrapper allows us to define a converting constructor that that allows us to construct
 *  the internal `shared_ptr` also from a derived class. This is very convenient.
 *  2) Null checking: This wrapper by default checks that the wrapped `shared_ptr` is not `nullptr`, and otherwise
 *  throws an exception. This means we generally need not worry about which access is safe (in the sense of
 *  raising an error on the python side instead of crashing the application) , and in the glue code the
 *  checks do not account for any relevant performance decrease.
 */
template<class T>
class PyWrapper {
public:
    template<class U>
    PyWrapper(U&& source, std::enable_if_t<std::is_convertible_v<U&, T&>>* a = nullptr) :
        m_Data(std::make_shared<U>(std::forward<U>(source))) {
    }
    PyWrapper(std::shared_ptr<T> d) : m_Data(std::move(d)) {};

    T* operator->() {
        return &access();
    }

    const T* operator->() const {
        return &access();
    }

    T& access() {
        if(m_Data) {
            return *m_Data;
        }
        throw std::runtime_error("Trying to access empty object");
    }

    [[nodiscard]] const T& access() const {
        if(m_Data) {
            return *m_Data;
        }
        throw std::runtime_error("Trying to access empty object");
    }

    [[nodiscard]] const std::shared_ptr<T>& ptr() const {
        return m_Data;
    }

    [[nodiscard]] std::shared_ptr<T>& ptr() {
        return m_Data;
    }
private:
    std::shared_ptr<T> m_Data;
};

// Move value into shared ptr
template<class T, class = typename std::enable_if<!std::is_lvalue_reference<T>::value>::type>
std::shared_ptr<T> wrap_shared(T&& source) {
    return std::make_shared<T>(source);
}

namespace py = pybind11;

using PyDataSet = std::shared_ptr<dismec::DatasetBase>;
using PyWeighting = std::shared_ptr<dismec::WeightingScheme>;
using PyModel = PyWrapper<dismec::model::Model>;

#endif //DISMEC_SRC_UTILS_BINDING_H

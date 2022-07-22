// Copyright (c) 2021, Aalto University, developed by Erik Schultheis
// All rights reserved.
//
// SPDX-License-Identifier: MIT

#ifndef DISMEC_HASH_VECTOR_H
#define DISMEC_HASH_VECTOR_H

#include <memory>
#include "matrix_types.h"

namespace dismec
{
    class VectorHash;

    /*! \class HashVector
     *  \brief An Eigen vector with versioning information, to implement simple caching of results.
     *  \details This class wraps an Eigen::Vector and adds an additional data field `m_UniqueID`.
     *  This field contains a unique number for each new HashVector created, and is updated whenever
     *  a potentially modifying operation is performed on the vector.
     *
     *  The unique id can be queried using the \ref hash() function, which returns a \ref VectorHash()
     *  object. The only purpose of this is to enable checks of whether a function is called with
     *  a vector argument that is the same as a previous call, without having to store and compare the
     *  contents of the vector. This works as follows (if `cached` doesn't need to be re-entrant):
     *  \code
     *  int cached(const HashVector& vec) {
     *      static VectorHash last_used{};
     *      static int last_result{};
     *      if(vec.hash() == last_used) {
     *          return last_result;
     *      }
     *      last_result = compute(vec.get());
     *      last_used = vec.hash();
     *  }
     *  \endcode
     *  The default-initialized \ref VectorHash compares unequal to all other hash values.
     *
     *  \note Technically, we are using an internal counter, which could overflow. On 64-bit systems, this should
     *  never occur (e.g. if you create a new HashVector every microsecond, it takes about 500k years). We don't
     *  really expect to run this code on hardware so old that std::size_t is less than 64 bit.
     */
    class HashVector {
    public:
        /*!
         * \brief Creates a new hash vector from the given DenseRealVector.
         * \details This moves the content from `data` into the newly created
         * `HashVector` and generates a new unique id.
         */
        explicit HashVector(DenseRealVector data);

        /*!
         * \brief Gets a constant reference to the data of this vector.
         * \details This is a non-modifying operation, so the id remains
         * unchanged.
         */
        [[nodiscard]] const DenseRealVector& get() const { return m_Data; }
        //! constant access to vector data.
        const DenseRealVector* operator->() const { return &m_Data; }

        /*!
         * \brief Gets the unique id of this vector.
         * \details This gets a unique id that is different from those of
         * all other \ref HashVector objects. This function returns the
         * same value until a modifying operation is performed with the vector.
         * Thus, for a given \ref HashVector `vec`:
         * \code
         * auto id = vec.hash();
         * auto twice = vec * 2;        // does not modify `vec`
         * assert(id == vec.hash());
         * vec = twice;                 // update value and id
         * assert(id != vec.hash());
         * \endcode
         * \return A \ref VectorHash() corresponding to this vector.
         */
        [[nodiscard]] VectorHash hash() const;

        /*!
         * \brief Update the contents of this vector.
         * \tparam Derived The actual type of the Eigen expression.By accepting this in a templated version, we make sure
         * that there are no unnecessary temporaries involved.
         * \param expr An eigen expression.
         * \return A reference to this.
         */
        template<class Derived>
        HashVector& operator=(const Eigen::EigenBase<Derived>& expr) {
            update_id();
            m_Data = expr;
            return *this;
        }

        /*!
         * \brief Gets non-const access to the underlying data.
         * \details Since this can change the contents of the underlying vector, this updates this vectors
         * unqiue id. However, note that you can store the returned reference, and modify the contents of
         * the vector later on. Such changes cannot be detected, and will result in invalid unique ids.
         */
        DenseRealVector& modify() {
            update_id();
            return m_Data;
        }
    private:
        void update_id();

        std::size_t m_UniqueID;
        DenseRealVector m_Data;
    };


    /*!
     * \brief A unique identifier for a \ref HashVector
     * \details This is the type that is returned by \ref HashVector::hash().
     * It can be used to verify whether two given vectors are guaranteed to have
     * the same content. In particular, it makes it possible to implement caching
     * of calculated values without having to store and compare a vector valued
     * argument, see the code example in \ref HashVector.
     */
    class VectorHash {
    public:
        /*!
         * \brief Default constructor. Initializes the `VectorHash` to special marker.
         * \details The default-constructed VectorHash contains a special marker value
         * as the id. This value never compares equal to any hash of a real `HashVector`.
         */
        VectorHash();

        /*!
         * \brief Checks that two hashes are equal.
         * \param other The other `VectorHash` to compare to.
         * \return Returns whether both refer to the same id, or both are empty.
         */
        bool operator==(const VectorHash& other) const {
            return m_UniqueID == other.m_UniqueID;
        }
        /// negation of the equality check
        bool operator!=(const VectorHash& other) const {
            return !(*this == other);
        }
    private:
        /*!
         * \brief Creates a `VectorHash` for the given ID.
         * \details This function is private and will only be called from \ref HashVector::hash()
         */
        VectorHash(std::size_t id) : m_UniqueID(id) {}
        std::size_t m_UniqueID;

        friend class HashVector;
    };

    // overload vector operators
    template<class T>
    auto operator+(const HashVector& vec, T&& other) {
        return vec.get() + other;
    }

    template<class T>
    auto operator+(T&& other, const HashVector& vec) {
        return other + vec.get() ;
    }

    template<class T>
    auto operator+=(T&& other, const HashVector& vec) {
        return other += vec.get() ;
    }

    template<class T>
    auto operator*(const HashVector& vec, T&& other) {
        return vec.get() * other;
    }

    template<class T>
    auto operator*(T&& other, const HashVector& vec) {
        return other * vec.get() ;
    }

    template<class T>
    auto operator*=(T&& other, const HashVector& vec) {
        return other *= vec.get() ;
    }

    class CacheHelper {
    public:
        explicit CacheHelper(Eigen::Index size) : m_Input(), m_Output(size) {}

        template<class F>
        const DenseRealVector& update(const HashVector& input, F&& function) {
            if(input.hash() == m_Input) {
                return m_Output;
            }
            function(input.get(), m_Output);
            m_Input = input.hash();
            return m_Output;
        }

        void invalidate() {
            m_Input = {};
        }
    private:
        VectorHash m_Input;
        DenseRealVector m_Output;
    };
}
#endif //DISMEC_HASH_VECTOR_H

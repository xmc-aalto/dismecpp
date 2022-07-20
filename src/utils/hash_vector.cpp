// Copyright (c) 2021, Aalto University, developed by Erik Schultheis
// All rights reserved.
//
// SPDX-License-Identifier: MIT

#include "hash_vector.h"
#include <atomic>

using namespace dismec;

/*!
 * \brief  Local namespace in which we define the counter used to create the unique ids for the hash vector.
 * \details  we have two options here -- either use an atomic counter, so that the results are consistent over all
 * threads, and have the chance that `next_id` is not lock-free and thus a potential performance problem, or use
 * thread_local counters. Given that `std::atomic<std::size_t>? should be lock-free on all reasonably modern systems
 * we run this code on, this is what we use.
 */
namespace {
    /// The counter for the ids. We expect `std::size_t` to have at least 64 bits, so that wrap-around is not an issue.
    std::atomic<std::size_t> next_id{0};
    static_assert(sizeof(std::size_t) >= 8, "std::size_t is expected to be at least 64 bit, so that we can be sure our unique ids never wrap around.");

    /*!
     * \brief Get the next valid id.
     * \details This increments the \ref next_id and returns the old value, in an atomic fashion so that even when
     * called from multiple threads, the return values will be unique.
     */
    std::size_t get_next_id()
    {
        // see also https://stackoverflow.com/questions/41206861/atomic-increment-and-return-counter
        return next_id++;
    }

}

HashVector::HashVector(DenseRealVector data) :
    m_UniqueID(get_next_id()),
    m_Data(std::move(data)) {
}

void HashVector::update_id() {
    m_UniqueID = get_next_id();
}

VectorHash HashVector::hash() const {
    return VectorHash(m_UniqueID);
}

VectorHash::VectorHash() : m_UniqueID(-1) {
}

#include "doctest.h"

/*!
 * \test This checks that construction and assignment result in the correct content in the vector.
 */
TEST_CASE("HashVector content") {
    HashVector hv(DenseRealVector::Zero(3));

    CHECK(hv->coeff(0) == 0.0);
    CHECK(hv->coeff(1) == 0.0);
    CHECK(hv->coeff(2) == 0.0);

    hv = DenseRealVector::Ones(3);
    CHECK(hv->coeff(0) == 1.0);
    CHECK(hv->coeff(1) == 1.0);
    CHECK(hv->coeff(2) == 1.0);
}

/*!
 * \test This checks that non-mutation operations leave the hash value as is, but mutation operations change the value.
 */
TEST_CASE("HashVector versioning") {
    VectorHash vh;
    HashVector hv(DenseRealVector::Zero(3));

    CHECK(vh != hv.hash());
    vh = hv.hash();
    CHECK(vh == hv.hash());

    // non-mutating operation
    auto content = hv.get();
    content = hv + DenseRealVector::Ones(3);
    CHECK(vh == hv.hash());
    CHECK(hv->coeff(0) == 0.0);

    hv = DenseRealVector::Ones(3);
    CHECK(vh != hv.hash());
}

/*!
 * \test This checks that two different HashVector objects always have different hash.
 */
TEST_CASE("HashVector uniqueness") {
    HashVector hv(DenseRealVector::Zero(3));
    HashVector hw(DenseRealVector::Zero(3));

    CHECK( hv.hash() != hw.hash() );
}

/*!
 * \test This checks that the uninitialized VectorHash compares unequal to the hash value of a vector.
 */
TEST_CASE("uninitialized VectorHash") {
    VectorHash vh;

    HashVector hv(DenseRealVector::Zero(3));
    CHECK_FALSE(vh == hv.hash());
    CHECK_FALSE(hv.hash() == vh);
}
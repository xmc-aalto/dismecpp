// Copyright (c) 2021, Aalto University, developed by Erik Schultheis
// All rights reserved.
//
// SPDX-License-Identifier: MIT

#ifndef DISMEC_FAST_SPARSE_ITER_H
#define DISMEC_FAST_SPARSE_ITER_H


namespace dismec
{
/*!
 * \brief This is an almost verbatim copy of the SparseFeatures::InnerIterator provided by Eigen.
 *
 * \details The only difference is that we do not check whether the matrix is compressed in the constructor, since we
 * already know this. This improves performance slightly.
 */
class FastSparseRowIter
{
public:
    using Scalar = SparseFeatures::Scalar;
    using Index = SparseFeatures::Index;
    using StorageIndex = SparseFeatures::StorageIndex;
    FastSparseRowIter(const SparseFeatures& mat, Index row)
            : m_values(mat.valuePtr()), m_indices(mat.innerIndexPtr()), m_outer(row)
    {
        m_id = mat.outerIndexPtr()[row];
        m_end = mat.outerIndexPtr()[row+1];
    }

    inline FastSparseRowIter& operator++() { m_id++; return *this; }

    [[nodiscard]] inline const Scalar& value() const { return m_values[m_id]; }
    inline Scalar& valueRef() { return const_cast<Scalar&>(m_values[m_id]); }

    [[nodiscard]] inline int index() const { return m_indices[m_id]; }
    [[nodiscard]] inline int outer() const { return m_outer; }
    [[nodiscard]] inline int row() const { return m_outer; }
    [[nodiscard]] inline int col() const { return index(); }

    inline explicit operator bool() const { return (m_id < m_end); }

protected:
    const Scalar* m_values;
    const StorageIndex* m_indices;
    const Index m_outer;
    Index m_id;
    Index m_end;
};



template<typename Scalar, int Options, typename StorageIndex, typename OtherDerived>
auto fast_dot(const Eigen::SparseMatrix<Scalar, Options, StorageIndex>& first, int row, const Eigen::MatrixBase<OtherDerived>& other) -> Scalar {
    auto values = first.valuePtr();
    auto indices = first.innerIndexPtr();
    auto id = first.outerIndexPtr()[row];
    auto end = first.outerIndexPtr()[row + 1];

    Scalar a(0);
    Scalar b(0);
    Scalar c(0);
    Scalar d(0);
    for (; id < end - 4; id += 4) {
        a += values[id] * other.coeff(indices[id]);
        b += values[id + 1] * other.coeff(indices[id + 1]);
        c += values[id + 2] * other.coeff(indices[id + 2]);
        d += values[id + 3] * other.coeff(indices[id + 3]);
    }
    Scalar res = (a + b) + (c + d);
    while (id != end) {
        res += values[id] * other.coeff(indices[id]);
        ++id;
    }
    return res;
}}


#endif //DISMEC_FAST_SPARSE_ITER_H

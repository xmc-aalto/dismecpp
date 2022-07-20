// Copyright (c) 2021, Aalto University, developed by Erik Schultheis
// All rights reserved.
//
// SPDX-License-Identifier: MIT

#ifndef DISMEC_EIGEN_GENERIC_H
#define DISMEC_EIGEN_GENERIC_H

#include <variant>
#include "utils/type_helpers.h"

namespace Eigen {
    template<typename Derived>
    class EigenBase;
}

#define EIGEN_VISITORS_IMPLEMENT_VISITOR(VISITOR, BASE, CALL)                           \
struct VISITOR {                                                                        \
    template<class Derived, class... Args>                                              \
    auto operator()(const Eigen::BASE<Derived>& source, Args&&... args) const {         \
        return source.CALL(std::forward<Args>(args)...);                                \
    }                                                                                   \
}

namespace dismec::eigen_visitors {
    EIGEN_VISITORS_IMPLEMENT_VISITOR(ColsVisitor, EigenBase, cols);
    EIGEN_VISITORS_IMPLEMENT_VISITOR(RowsVisitor, EigenBase, rows);
    EIGEN_VISITORS_IMPLEMENT_VISITOR(SizeVisitor, EigenBase, size);
}

#undef EIGEN_VISITORS_IMPLEMENT_VISITOR

namespace dismec::types {
    class VarWrapBase {};

    template<class... Types>
    class EigenVariantWrapper : public VarWrapBase {
    public:
        using variant_t = std::variant<Types...>;

        template<class T>
        explicit EigenVariantWrapper(T&& source) : m_Variant(std::forward<T>(source)) {}

        [[nodiscard]] auto size() const {
            return std::visit(eigen_visitors::SizeVisitor{}, m_Variant);
        }

        [[nodiscard]] auto rows() const {
            return std::visit(eigen_visitors::RowsVisitor{}, m_Variant);
        }

        [[nodiscard]] auto cols() const {
            return std::visit(eigen_visitors::ColsVisitor{}, m_Variant);
        }

        variant_t& unpack_variant() {
            return m_Variant;
        }

        const variant_t& unpack_variant() const {
            return m_Variant;
        }

        template<class T>
        T& get() {
            return std::get<T>(m_Variant);
        }

        template<class T>
        const T& get() const {
            return std::get<T>(m_Variant);
        }

    protected:
        variant_t m_Variant;
    };

    template<class T>
    constexpr bool is_variant_wrapper = std::is_base_of_v<VarWrapBase, std::decay_t<T>>;

    // TODO figure out the correct return type specification here
    template<class T>
    decltype(auto) unpack_variant_wrapper(T&& source, std::enable_if_t<!is_variant_wrapper<T>, void*> dispatch = nullptr) {
        return source;
    }

    template<class T>
    decltype(auto) unpack_variant_wrapper(T&& source, std::enable_if_t<is_variant_wrapper<T>, void*> dispatch = nullptr) {
        return source.unpack_variant();
    }


    template<class F, class... Variants>
    auto visit(F&& f, Variants&& ... variants) {
        return std::visit(std::forward<F>(f), unpack_variant_wrapper(std::forward<Variants>(variants))...);
    }

    template<class Dense, class Sparse>
    class GenericMatrix : public EigenVariantWrapper<Dense, Sparse> {
        public:
        using base_t = EigenVariantWrapper<Dense, Sparse>;
        using base_t::base_t;

        [[nodiscard]] const Dense& dense() const {
            return std::get<Dense>(this->m_Variant);
        }

        [[nodiscard]] Dense& dense() {
            return std::get<Dense>(this->m_Variant);
        }

        [[nodiscard]] const Sparse& sparse() const {
            return std::get<Sparse>(this->m_Variant);
        }

        [[nodiscard]] Sparse& sparse() {
            return std::get<Sparse>(this->m_Variant);
        }

        [[nodiscard]] bool is_sparse() const {
            return this->m_Variant.index() == 1;
        }
    };


    template<class... Types>
    class RefVariant : public EigenVariantWrapper<Eigen::Ref<Types>...> {
    public:
        using base_t = EigenVariantWrapper<Eigen::Ref<Types>...>;
        using base_t::base_t;
    };


    template<class T>
    class GenericVectorRef : public RefVariant<DenseVector<T>, SparseVector<T>> {
        public:
        using base_t = RefVariant<DenseVector<T>, SparseVector<T>>;
        using DenseRef = Eigen::Ref<DenseVector<T>>;
        using SparseRef = Eigen::Ref<SparseVector<T>>;

        explicit GenericVectorRef(const DenseVector<T>& m) : base_t(DenseRef(m)) {}
        explicit GenericVectorRef(const SparseVector<T>& m) : base_t(SparseRef(m)) {}

        [[nodiscard]] const DenseRef& dense() const {
            return this->template get<DenseRef>();
        }

        [[nodiscard]] DenseRef& dense() {
            return this->template get<DenseRef>();
        }

        [[nodiscard]] const SparseRef& sparse() const {
            return this->template get<SparseRef>();
        }

        [[nodiscard]] SparseRef& sparse() {
            return this->template get<SparseRef>();
        }
    };

    template<class T>
    class GenericMatrixRef : public RefVariant<DenseRowMajor<T>, DenseColMajor<T>, SparseRowMajor<T>, SparseColMajor<T>> {
    public:
        using base_t = RefVariant<DenseRowMajor<T>, DenseColMajor<T>, SparseRowMajor<T>, SparseColMajor<T>>;

        using DenseRowMajorRef = Eigen::Ref<DenseRowMajor<T>>;
        using DenseColMajorRef = Eigen::Ref<DenseColMajor<T>>;
        using SparseRowMajorRef = Eigen::Ref<SparseRowMajor<T>>;
        using SparseColMajorRef = Eigen::Ref<SparseColMajor<T>>;

        GenericMatrixRef(DenseRowMajorRef m) : base_t(std::move(m)) {}
        GenericMatrixRef(DenseColMajorRef m) : base_t(std::move(m)) {}
        GenericMatrixRef(SparseRowMajorRef m) : base_t(std::move(m)) {}
        GenericMatrixRef(SparseColMajorRef m) : base_t(std::move(m)) {}

        explicit GenericMatrixRef(const DenseRowMajor<T>& m) : base_t(DenseRowMajorRef(m)) {}
        explicit GenericMatrixRef(const DenseColMajor<T>& m) : base_t(DenseColMajorRef(m)) {}
        explicit GenericMatrixRef(const SparseRowMajor<T>& m) : base_t(SparseRowMajorRef(m)) {}
        explicit GenericMatrixRef(const SparseColMajor<T>& m) : base_t(SparseColMajorRef(m)) {}
    };
}

#endif //DISMEC_EIGEN_GENERIC_H

// Copyright (c) 2021, Aalto University, developed by Erik Schultheis
// All rights reserved.
//
// SPDX-License-Identifier: MIT

#ifndef DISMEC_REG_SQ_HINGE_H
#define DISMEC_REG_SQ_HINGE_H

#include "objective.h"
#include "matrix_types.h"
#include "hash_vector.h"
#include "linear.h"

namespace objective {

    /*! Objective for regularized Support Vector classification
     * with Squared Hinge loss. This is for binary labels, multilabel
     * problems can be generated trivially by having `l` independent
     * `Regularized_SquaredHingeSVC` objectives.
     */
    class Regularized_SquaredHingeSVC : public LinearClassifierImpBase<Regularized_SquaredHingeSVC> {
        using features_t = SparseFeatures;
    public:
        explicit Regularized_SquaredHingeSVC(std::shared_ptr<const GenericFeatureMatrix> X, std::unique_ptr<Objective> regularizer);

        [[nodiscard]] const features_t& features() const;

        void gradient_imp(const HashVector& location, Eigen::Ref<DenseRealVector> target);
        void gradient_at_zero_imp(Eigen::Ref<DenseRealVector> target);

        void hessian_times_direction_imp(const HashVector& location,
                                               const DenseRealVector &direction,
                                               Eigen::Ref<DenseRealVector> target);

        void diag_preconditioner_imp(const HashVector& location, Eigen::Ref<DenseRealVector> target);

        void gradient_and_pre_conditioner_imp(const HashVector& location, Eigen::Ref<DenseRealVector> gradient,
                                          Eigen::Ref<DenseRealVector> pre);

        template<typename OtherDerived>
        static real_t value_from_xTw(const DenseRealVector& cost, const BinaryLabelVector& labels, const Eigen::DenseBase<OtherDerived>& xTw)
        {
            real_t f = 0;
            assert(xTw.size() == labels.size());
            for(Eigen::Index i = 0; i < labels.size(); ++i) {
                real_t label = labels.coeff(i);
                real_t d = std::max(real_t{0}, real_t{1} - label * xTw.coeff(i));
                real_t factor = cost.coeff(i);
                f += factor * d * d;
            }

            return f;
        }
    private:
        void invalidate_labels() override;
        void margin_error(const HashVector& w);

        VectorHash m_Last_MV;
        // do not write to these directly, only `margin_error` is allowed to do that
        std::vector<int> m_MVPos;
        std::vector<real_t> m_MVVal;

        template<class T, class U>
        void gradient_and_pre_conditioner_tpl(const HashVector&location, T&& gradient, U&& pre); // __attribute__((hot));
    };
}

#endif //DISMEC_REG_SQ_HINGE_H

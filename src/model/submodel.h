// Copyright (c) 2021, Aalto University, developed by Erik Schultheis
// All rights reserved.
//
// SPDX-License-Identifier: MIT

#ifndef DISMEC_SUBMODEL_H
#define DISMEC_SUBMODEL_H

#include "model.h"

namespace model {
    template<class T>
    class SubModelWrapper : public Model {
        using model_t = std::remove_reference_t<decltype(*std::declval<T>())>;
        static_assert(std::is_convertible_v<std::remove_cv_t<model_t>&, Model&>, "T should be like a pointer to Model");
    public:
        SubModelWrapper(T original, label_id_t begin, label_id_t end) :
                Model(PartialModelSpec{begin, end - begin, original->num_labels()}), m_Original(original)
                {
        }

        [[nodiscard]] long num_features() const override { return m_Original->num_features(); }
        [[nodiscard]] bool has_sparse_weights() const override { return m_Original->has_sparse_weights(); }

        void get_weights_for_label_unchecked(label_id_t label, Eigen::Ref<DenseRealVector> target) const override {
            // we cannot directly call the _unchecked method, so we have to undo the label correction.
            return m_Original->get_weights_for_label(labels_begin() + label.to_index(), target);
        }
        void set_weights_for_label_unchecked(label_id_t label, const GenericInVector& weights) override {
            // we cannot directly call the _unchecked method, so we have to undo the label correction.
            if constexpr (std::is_const_v<model_t>) {
                throw std::logic_error("Cannot set weights for constant sub-model");
            } else {
                m_Original->set_weights_for_label(labels_begin() + label.to_index(), weights);
            }
        }

        void predict_scores_unchecked(const GenericInMatrix& instances, PredictionMatrixOut target) const override {
            throw std::logic_error("Cannot predict from model view");
        }
    private:
        T m_Original;
    };

    using SubModelView = SubModelWrapper<Model*>;
    using ConstSubModelView = SubModelWrapper<const Model*>;
}

#endif //DISMEC_SUBMODEL_H

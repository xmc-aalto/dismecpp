// Copyright (c) 2021, Aalto University, developed by Erik Schultheis
// All rights reserved.
//
// SPDX-License-Identifier: MIT

#ifndef DISMEC_SPARSE_H
#define DISMEC_SPARSE_H

#include "model.h"

namespace dismec::model {
    class SparseModel : public Model {
    public:
        SparseModel(long num_features, long num_labels);
        SparseModel(long num_features, PartialModelSpec partial);

        [[nodiscard]] long num_features() const override;

        [[nodiscard]] bool has_sparse_weights() const override { return true; }

        void predict_scores_unchecked(const FeatureMatrixIn& instances, PredictionMatrixOut target) const override;

        void get_weights_for_label_unchecked(label_id_t label, Eigen::Ref<DenseRealVector> target) const override;

        void set_weights_for_label_unchecked(label_id_t label, const WeightVectorIn& weights) override;

    private:
        std::vector<SparseRealVector> m_Weights;
        long m_NumFeatures;
    };
}

#endif //DISMEC_SPARSE_H

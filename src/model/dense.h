// Copyright (c) 2021, Aalto University, developed by Erik Schultheis
// All rights reserved.
//
// SPDX-License-Identifier: MIT

#ifndef DISMEC_DENSE_H
#define DISMEC_DENSE_H

#include "model/model.h"

namespace dismec::model {

    /*!
     * \brief Implementation of the \ref Model class that stores the weights as a single, dense matrix.
     */
    class DenseModel : public Model
    {
    public:
        using WeightMatrix = types::DenseColMajor<real_t>;
        using weight_matrix_ptr = std::shared_ptr<WeightMatrix>;

        /*!
         * \brief Creates a (complete) dense model with the given weight matrix.
         * \details If you want to create only a partial model, the
         * \ref DenseModel(std::shared_ptr<Eigen::MatrixXd> weights, PartialModelSpec partial)
         * constructor can be used.
         * \param weights Weights for each label.
         */
        explicit DenseModel(weight_matrix_ptr weights);

        /*!
         * \brief Creates a (potentially partial) dense model with the given weight matrix.
         * \param weights Weights for the labels as specified in `partial`. The number of columns needs to match
         * the number of weights as specified in `partial`.
         * \param partial Specifies where to place this partial model inside the complete model.
         * \throws If the number of weight vectors (columns of `weights`) does not match the specification in
         * `partial`, or if the given label range is invalid.
         */
        DenseModel(weight_matrix_ptr weights, PartialModelSpec partial);

        DenseModel(long num_features, long num_labels);

        DenseModel(long num_features, PartialModelSpec partial);

        //! A dense model doesn't have sparse weights.
        [[nodiscard]] bool has_sparse_weights() const final { return false; }

        [[nodiscard]] long num_features() const override;

        /// provides read-only access to the raw weight matrix.
        [[nodiscard]] const WeightMatrix& get_raw_weights() const { return *m_Weights; }

    private:
        void get_weights_for_label_unchecked(label_id_t label, Eigen::Ref<DenseRealVector> target) const override;

        void set_weights_for_label_unchecked(label_id_t label, const WeightVectorIn& weights) override;

        void predict_scores_unchecked(const FeatureMatrixIn& instances,
                                      PredictionMatrixOut target) const override;

        /*!
         * \brief The matrix of weights.
         * \details Each column in this matrix corresponds to the weight vector for one label.
         */
        weight_matrix_ptr m_Weights;
    };

}

#endif //DISMEC_DENSE_H

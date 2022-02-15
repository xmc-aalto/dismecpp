// Copyright (c) 2021, Aalto University, developed by Erik Schultheis
// All rights reserved.
//
// SPDX-License-Identifier: MIT

#ifndef DISMEC_MODEL_H
#define DISMEC_MODEL_H

#include <vector>
#include <memory>
#include <variant>
#include "matrix_types.h"
#include "data/types.h"

namespace model
{
    /*!
     * \brief Specifies how to interpret a weight matrix for a partial model.
     * \details This contains information about the first label for which there
     * are weights, the number of weights, and the number of total labels.
     */
    struct PartialModelSpec {
        label_id_t first_label;     //!< First label in the partial model
        long label_count;           //!< Number of labels in the partial model.
        long total_labels;          //!< Total number of labels.
    };

    /*!
     * \brief A model combines a set of weight with some meta-information about these weights.
     * \details The weights may be represented as a dense or as a sparse matrix. The model class allows access to model
     * meta-data through functions such as
     *  - \ref Model::num_labels()
     *  - \ref Model::num_features()
     *  - \ref Model::has_sparse_weights()
     *
     *  as well as access to the weights via
     *  - \ref Model::get_weights_for_label()
     *  - \ref Model::set_weights_for_label()
     *
     *  Finally, the class provides an interface for performing predictions. This is handled by
     *  the \ref Model::predict_scores function, which exists in a version that handles dense features and
     *  one version that handles sparse feature vectors.
     *
     *  The current design abstract away whether the weights are saved internally in a dense or sparse representation,
     *  but this comes at the cost that getting and setting weights has sub-optimal performance characteristics for both
     *  sparse and dense data format. However, these functions are mostly for 1. IO. In th case we expect the actual IO
     *  cost to be much more than the in-memory copies 2. setting weights after training. Same. In both cases, the
     *  solution is to have one buffer per thread, and only do a single read or write.
     *
     *  For performance reasons, the model functions do not return newly constructed vectors, but instead expect the
     *  caller to provide a buffer in which they will place their values. Where appropriate, input vectors are not taken
     *  as Eigen::Ref objects so they can bind to sub-matrices. In that case, the caller needs to make sure that the
     *  resulting sub-objects have an inner stride of one. This allows e.g. to parallelize the prediction process over
     *  the examples.
     *
     *  \todo what about per-label prediction? That would also be a valid axis for parallelizing.
     *
     *  Instances of the `Model` class can either represent an entire model, or a sub-model that only contains a
     *  contiguous subset of the weight vectors of the whole model. This is necessary e.g. to facility
     *  distributed-memory training and prediction. If only a subset of the weights are contained, then the labels can
     *  be queried by \ref Model::labels_begin() and \ref Model::labels_end().
     */
    class Model {
    public:
        using PredictionMatrixOut = Eigen::Ref<PredictionMatrix>;
        using FeatureMatrixIn = GenericInMatrix ;
        using WeightVectorIn = GenericInVector ;

        explicit Model(PartialModelSpec spec);

        virtual ~Model() = default;

        // general data shape
        /*!
         * \brief How many labels are in the underlying dataset.
         * \details If `is_partial_model()` is false, this is equal to the number of weights in this model.
         */
        [[nodiscard]] long num_labels() const noexcept { return m_NumLabels; };

        /// \brief How many weights are in each weight vector, i.e. how many features should the input have.
        [[nodiscard]] virtual long num_features() const = 0;

        /*!
         * \brief How many weights vectors are in this model.
         * \details If `is_partial_model` is false, this is equal to the number of labels.
         */
        [[nodiscard]] long num_weights() const noexcept { return labels_end() - labels_begin(); }

        /// \brief whether this model stores the weights in a sparse format, or a dense format.
        [[nodiscard]] virtual bool has_sparse_weights() const = 0;

        // submodel data
        /// returns true if this instance only stores part of the weights of an entire model
        [[nodiscard]] bool is_partial_model() const;

        /// If this is a partial model, returns the index of the first label for which weight vectors are available. For
        /// a complete model this always returns 0.
        [[nodiscard]] label_id_t labels_begin() const noexcept { return m_LabelsBegin; }

        /// If this is a partial, returns the first label index for which no weights are available. For a complete
        /// model this returns `num_labels()`
        [[nodiscard]] label_id_t labels_end() const noexcept { return m_LabelsEnd; }

        /// How many labels are in this submodel
        [[nodiscard]] long contained_labels() const noexcept { return m_LabelsEnd - m_LabelsBegin; }

        /*!
         * \brief Gets the weights for the given label as a dense vector.
         * \details Since we do not know whether the weights saved in the model are sparse or dense vectors, we cannot
         * simply return a const reference here. Instead, the user is required to provide a pre-allocated buffer
         * `target` into which the weights will be copied.
         * \throws If `target` does not have the correct size, or if `label` is invalid.
         */
        void get_weights_for_label(label_id_t label, Eigen::Ref<DenseRealVector> target) const;

        /*!
         * \brief Sets the weights for a label.
         * \details This assigns the given vector as weights for the `label`th label. For non-overlapping
         * `label` parameters this function can safely be called concurrently from different threads. This function can
         * be used for both dense and sparse internal weight representation. If the model internally uses a sparse
         * representation, the zeros will be filtered out.
         * \throws If `weights` does not have the correct size, or if `label` is invalid.
         */
        void set_weights_for_label(label_id_t label, const WeightVectorIn& weights);
        /*!
         * \brief Calculates the scores for all examples and all labels in this model.
         * \details This is just the matrix multiplication of the input instances and the weight matrix.
         * This function can be called safely from multiple threads. Note that Eigen::Ref requires that the passed
         * submatrix has an inner stride of 1, i.e. that features for a single instance are provided as contiguous
         * memory (in case of dense features), and the same for the pre-allocated buffer for the targets. This can be
         * achieved e.g. by using rows of a row-major matrix.
         * \param instances Feature vector of the instances for which we want to predict the scores. This is handled as
         * a `Eigen::Ref` parameter so that subsets of a large dataset can be passed without needing data to be copied.
         * Should have number of columns equal to the number of features. The `GenericInMatrix` allows different data
         * formats to be passed -- however, some data formats may be more efficient than others.
         * \param target This is the matrix to which the scores will be written. Has to have the correct size, i.e. the
         * same number of rows as `instances` and number of columns equal to the number of labels.
         * \throw If `instances` and `target` have different number of rows, or if the number of columns (rows) in
         * `instances` (`target`) does not match `get_num_features()` (`get_num_labels()`).
         */
        void predict_scores(const FeatureMatrixIn& instances, PredictionMatrixOut target) const;

    protected:
        /// this function verifies that `label` is a valid label, in `[labels_begin(), labels_end())`, and returns
        /// a zero-based label, i.e. it subtracts `labels_begin()`.
        [[nodiscard]] label_id_t adjust_label(label_id_t label) const;

    private:
        /*!
         * \brief Unchecked version of predict_scores().
         * \copydetails  Model::predict_scores()
         * \note This function is called from `predict_scores` and can assume that the shapes of `instances` and
         * `target` have been verified.
        */
        virtual void predict_scores_unchecked(const FeatureMatrixIn& instances,
                                              PredictionMatrixOut target) const = 0;

        /*!
         * \brief Unchecked version of get_weights_for_label().
         * \copydetails Model::get_weights_for_label()
         * \note This function is called from `get_weights_for_label` and can assume label is a valid label index that
         * has been corrected for partial models (i.e. such that the first label of the partial model will get the index
         * `0`). Target can be assumed to be of correct size.
        */
        virtual void get_weights_for_label_unchecked(label_id_t label, Eigen::Ref<DenseRealVector> target) const = 0;

        /*!
         * \brief Unchecked version of set_weights_for_label().
         * \copydetails Model::set_weights_for_label()
         * \note This function is called from `set_weights_for_label` and can assume label is a valid label index that
         * has been corrected for partial models (i.e. such that the first label of the partial model will get the index
         * `0`). `weights` can be assumed to be of correct size.
        */
        virtual void set_weights_for_label_unchecked(label_id_t label, const WeightVectorIn& weights) = 0;

        label_id_t m_LabelsBegin;
        label_id_t m_LabelsEnd;


        /*!
         * \brief Total number of labels of the complete model.
         * \details if this is a partial model, the information about the number of total labels cannot be extracted
         * from weights.
         */
        long m_NumLabels;
    };

}

#endif //DISMEC_MODEL_H

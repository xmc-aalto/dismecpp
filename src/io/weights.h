// Copyright (c) 2021, Aalto University, developed by Erik Schultheis
// All rights reserved.
//
// SPDX-License-Identifier: MIT

#ifndef DISMEC_WEIGHTS_H
#define DISMEC_WEIGHTS_H

#include <iosfwd>

namespace model {
    class Model;
}

namespace io::model
{
    using ::model::Model;

    /*!
     * \brief Saves the dense weights in a plain-text format.
     * \details Each row corresponds to one label, and each column to a feature.
     * \param target Output stream where the data is written.
     * \param model Reference to the model whose weights will be saved. If the model has sparse weights, the weight
     * vectors will be converted to a dense format.
     * \attention Note that this is transposed from the way liblinear saves weights, but it makes it easier to read
     * the weights for only a subset of the labels.
     */
    void save_dense_weights_txt(std::ostream& target, const Model& model);

    /*!
     * \brief Loads weights saved by io::model::save_dense_weights_txt.
     * \param source The stream from which to load the weights.
     * \param target Model whose weights to fill in. It is assumed that `target` is already
     * of the correct size.
     * \throws If there is an error when reading the weights, or if the number of labels in
     * `target` mismatches the number of lines in `source`.
     */
    void load_dense_weights_txt(std::istream& source, Model& target);

    /*!
     * \brief Saves the dense weights in a npy file.
     * \details The weights are saved in as a two dimensional array, with rows
     * corresponding to labels and columns corresponding to features. The data is
     * written in row-major format to allow loading a subset of the labels by reading
     * contiguous parts of the file. Since the output is binary, we operate directly
     * on a stream-buffer here.
     * \param target Output stream buffer where the data is written.
     * \param model Reference to the model whose weights will be saved. If the model has sparse weights, the weight
     * vectors will be converted to a dense format.
     */
    void save_dense_weights_npy(std::streambuf& target, const Model& model);

    /*!
     * \brief Loads dense weights from a npy file.
     * \details The npy file needs to contain a row-major array, with the number of rows corresponding to the number
     * of labels. The data type needs to be an exact match to the datatype used in the model, i.e. `real_t`.
     * \param target Input stream buffer from which the data is read.
     * \param target Model whose weights to fill in. It is assumed that `target` is already
     * of the correct size.
     */
    void load_dense_weights_npy(std::streambuf& target, Model& model);

    /*!
     * \brief Saves the weights in sparse plain-text format, culling small weights.
     * \param target Stream to which the weights are written.
     * \param model Reference to the model whose weights will be saved. Note that if the model already has sparse
     * weights, additional culling based in threshold will be performed over the nonzero weights. Each line in the
     * resulting file corresponds to the weight vector of one label.
     * \param threshold Threshold below which weights will be set to zero and omitted from the file.
     * \throw If `threshold < 0`. `target` remains unmodified in that case.
     */
    void save_as_sparse_weights_txt(std::ostream& target, const Model& model, double threshold);

    /*!
    * \brief Loads sparse weights from plain-text format.
    * \param source The stream from which to load the weights.
    * \param target Model whose weights to fill in. It is assumed that `target` is already
    * of the correct size.
    * \throws If there is an error when reading the weights, or if the number of labels in
    * `target` mismatches the number of lines in `source`.
    */
    void load_sparse_weights_txt(std::istream& source, Model& target);
}


#endif //DISMEC_WEIGHTS_H

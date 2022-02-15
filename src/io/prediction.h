// Copyright (c) 2021, Aalto University, developed by Erik Schultheis
// All rights reserved.
//
// SPDX-License-Identifier: MIT

#ifndef DISMEC_IO_PREDICTION_H
#define DISMEC_IO_PREDICTION_H

#include <filesystem>
#include "matrix_types.h"

namespace io::prediction
{
    using std::filesystem::path;
    /*!
     * \brief Saves sparse predictions as a text file.
     * \details The sparse predictions are expected to be available as two separate
     * arrays `indices` and `values`. Each is an `N x K` array, where `N` is the number
     * of prediction points and `K` denotes the number of entries for each instance.
     * The values are then interpreted as follows: For instance `i`, the predictions
     * are given for the labels `indices[i, j]`, with corresponding scores `values[i, k]`.
     * \param target Path to the file which will be created or overwritten, or output stream.
     * \param values The scores for the labels corresponding to `indices`.
     * \param indices The indices corresponding to the values.
     * \throws std::invalid_argument if the shape of `values` does not match `indices`.
     */
    void save_sparse_predictions(path target,
                                 const PredictionMatrix& values,
                                 const IndexMatrix& indices);

    /// \copydoc save_sparse_predictions()
    void save_sparse_predictions(std::ostream& target,
                                 const PredictionMatrix& values,
                                 const IndexMatrix& indices);

    /*!
     * \brief Reads sparse predictions as saved by `save_sparse_predictions()`.
     * \param source The stream from which the data is read. The first line is expected
     * the header that contains the number of instances and predictions, followed by the
     * prediction of one instance each line.
     * \todo document this format and provide a link.
     * \return A pair that contains the indices and the corresponding scores.
     */
    std::pair<IndexMatrix, PredictionMatrix> read_sparse_prediction(std::istream& source);

    /// \copydoc read_sparse_prediction()
    std::pair<IndexMatrix, PredictionMatrix> read_sparse_prediction(path source_file);

    /*!
     * \brief Saves predictions as a dense txt matrix.
     * \param target Path to the file which will be created or overwritten, or output stream.
     * \param values Matrix with the results. Each row corresponds to an instance and each column to a label.
     */
    void save_dense_predictions(path target, const PredictionMatrix& values);

    /// \copydoc save_dense_predictions()
    void save_dense_predictions(std::ostream& target, const PredictionMatrix& values);
}

#endif //DISMEC_IO_PREDICTION_H

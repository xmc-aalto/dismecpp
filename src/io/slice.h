// Copyright (c) 2021, Aalto University, developed by Erik Schultheis
// All rights reserved.
//
// SPDX-License-Identifier: MIT

#ifndef DISMEC_SLICE_H
#define DISMEC_SLICE_H

#include <filesystem>
#include <iosfwd>
#include "fwd.h"


/*! \page slice-data Slice data format
 * In this data format the input dataset is given in two files: One file with a (dense) representation of features
 * and one file with the (sparse) labels.
*/

namespace dismec::io {
    /*!
     * \brief reads a dataset given in slice format.
     * \details For a description of the data format, see \ref slice-data
     * \param features An input stream from which the feature data is read.
     * \param labels An input stream from which the labels will be read
     * \return The parsed multi-label dataset.
     * \throws std::runtime_error if the parser encounters an error in the data format.
     */
    MultiLabelData read_slice_dataset(std::istream& features, std::istream& labels);

    MultiLabelData read_slice_dataset(const std::filesystem::path& features, const std::filesystem::path& labels);
}

#endif //DISMEC_SLICE_H

// Copyright (c) 2021, Aalto University, developed by Erik Schultheis
// All rights reserved.
//
// SPDX-License-Identifier: MIT

#ifndef DISMEC_XMC_H
#define DISMEC_XMC_H

#include <filesystem>
#include <iosfwd>


class MultiLabelData;

/*! \page xmc-data XMC data format
This is the data format used e.g. here `http://manikvarma.org/downloads/XC/XMLRepository.html`. It supports multiple
labels per example, and encodes features and labels in a sparse format. Can be read using the \ref io::read_xmc_dataset
function.

\section Format Specification
Files start with a header line, which contains three positive integers that denote the number of instances
(i.e. number of lines following) as well as number of features and number of labels. This is followed
by one line for each instance, which has first a comma-separated list of label ids and then a space-separated
list of sparse features, where each sparse feature consists of an integer feature index and a real-valued
feature value, separated by a colon. The comma and colon need to follow the preceding number immediately
(i.e. without whitespace), but can themselves be followed by whitespace. An empty label list has to be indicated
by the line starting with a whitespace character. Both spaces and tabs are recognized as whitespace.
Empty lines are ignored, if you have an example without labels where all features are zero, you have to specify
one of the features with an explicit zero. We also skip any line whose first character is a `#`. Placing `#` at
any other location is an error.

An example file would be
```
3 10 5
# 3 instances, 10 features {0, ..., 9} and 5 labels {0, ..., 4}
2     4:1.0     5:-0.5
1, 4  2:1.0e-4
0,1   0:0.5     3:2.2
```

We also support reading files in which features and labels are indexed starting from one. In that case,
set `IndexMode` to \ref io::IndexMode::ONE_BASED.

\section Implementation Details
The functions for reading xmc data are defined in \ref xmc.cpp. In broad terms, the reading works as follows:
First, one quick pass is performed over the entire dataset, in which we count the number of occurrences of `:`
characters. This allows us to pre-allocate the buffers for the sparse feature matrix immediately at the correct
size, so that all insert operations will be `O(1)`. The second pass then does the actual number parsing. In that
case, I expect no disk read overhead, since the data should still be cached in RAM, but I have not verified this.
However, from a fast SSD, reading about 1.5GB of data file takes less than 15 seconds, so this is not a bottleneck.

To support both 0 and 1 based indexing, the internal reading method is templated over an `IndexOffset` integer
parameter, which is either one or zero. In that way, we get optimized code for the default (=0) setting, but can
still easily support 1-based indexing.

Why do we allow for spaces after the separators, but not before? The reason is simply that these spaces are
automatically skipped by the conversion functions from the standard library, so disallowing them would in fact be
extra work on our side. On the other hand, by not allowing spaces after the numbers, we can immediately check if
the feature list has ended (space) or will continue (,), without needing to look ahead.
*/


//! io namespace
//! TODO convert this code to use the faster <charconv> methods once gcc implements them for floats
namespace io
{
    /// Enum to decide whether indices in an xmc file are starting from 0 or from 1.
    enum class IndexMode {
        ZERO_BASED,     //!< labels and feature indices are 0, 1, ..., num - 1
        ONE_BASED       //!< labels and feature indices are 1, 2, ..., num
    };

    /*!
     * \brief Reads a dataset given in the extreme multilabel classification format.
     * \details For a description of the data format, see \ref xmc-data
     * \param source Path to the file which we want to load, or an input stream.
     * \param mode Whether indices are assumed to start from 0 (the default) or 1.
     * \return The parsed multi-label dataset.
     * \throws std::runtime_error , if the file cannot be opened,
     * or if the parser encounters an error in the data format.
     */
    MultiLabelData read_xmc_dataset(const std::filesystem::path& source, IndexMode mode=IndexMode::ZERO_BASED);

    /*!
     * \brief reads a dataset given in the extreme multilabel classification format.
     * \details For a description of the data format, see \ref xmc-data
     * \param source An input stream from which the data is read. Since the reader does two passes over the
     * data, this needs to be resettable to the beginning.
     * \param name What name to display in the logging status updates.
     * \param mode Whether indices are assumed to start from 0 (the default) or 1.
     * \return The parsed multi-label dataset.
     * \throws std::runtime_error if the parser encounters an error in the data format.
     */
    MultiLabelData read_xmc_dataset(std::istream& source, std::string_view name, IndexMode mode=IndexMode::ZERO_BASED);


    /*!
     * \brief Saves the given dataset in XMC format.
     * \param data The dataset to be saved. Only supports datasets with sparse features.
     * \param target The output stream where we will put the data.
     */
    void save_xmc_dataset(std::ostream& target, const MultiLabelData& data);

    void save_xmc_dataset(std::filesystem::path source, const MultiLabelData& data, int precision=4);
}

#endif //DISMEC_XMC_H

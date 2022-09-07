// Copyright (c) 2021, Aalto University, developed by Erik Schultheis
// All rights reserved.
//
// SPDX-License-Identifier: MIT

#ifndef DISMEC_NUMPY_H
#define DISMEC_NUMPY_H

#include <iosfwd>
#include <string>
#include <string_view>
#include "matrix_types.h"

namespace dismec::io {
    /*!
     * \brief Check whether the stream is a npy file.
     * \details This peeks at the next 6 bytes of target and checks whether they form the npy magic string.
     * In any case, the read pointer is set back to the original position.
     */
    bool is_npy(std::istream& target);

    /*!
     * \brief Writes the header for a npy file.
     * \details This write the npy header to `target`, with the data `description` already provided. This means that
     * this function writes the magic bytes and version number, pads `description` to achieve 64 bit alignment of data,
     * and writes the header length, description, and padding to `target. The header is terminated with a newline
     * character.
     * \param target The stream buffer to which the data is written. Note: If this is a file stream,
     * it should be in binary mode!
     * \param description The description of the data, as a string-formatted python dictionary.
     */
    void write_npy_header(std::streambuf& target, std::string_view description);

    /*!
     * \brief Creates a string with the data description dictionary for (1 dimensional) arrays.
     * \param dtype_desc Description string for the data element.
     * \param column_major Whether the format is column_major or row_major. Not really relevant for 1D I guess.
     * \param size The number of elements in the array.
     * \return A string containing a literal python dictionary.
     */
    std::string make_npy_description(std::string_view dtype_desc, bool column_major, std::size_t size);

    /*!
     * \brief Creates a string with the data description dictionary for matrices.
     * \param dtype_desc Description string for the data element.
     * \param column_major Whether the format is column_major or row_major.
     * \param rows The number of rows in the matrix.
     * \param cols The number of columns in the matrix.
     * \return A string containing a literal python dictionary.
     */
    std::string make_npy_description(std::string_view dtype_desc, bool column_major, std::size_t rows, std::size_t cols);

    /*!
     * \brief Contains the data of the header of a npy file with an array that has at most 2 dimensions.
     */
    struct NpyHeaderData {
        std::string DataType;       //!< The data type `descr`
        bool ColumnMajor;           //!< Whether the data is column major (Fortran)
        long Rows;                  //!< The number of rows in the data
        long Cols;                  //!< The number of columns in the data. This will be 0 if the data is a
                                    //!< one-dimensional array.
    };

    /*!
     * \brief Parses the header of the npy file given by `source`.
     * \details After calling this function, the read pointer of source will be positioned
       such that subsequent reads access the data portion of the npy file.
       \throws std::runtime_error If the magic bytes don't match, the version is unknown, or any other parsing
       error occurs.
    */
    NpyHeaderData parse_npy_header(std::streambuf& source);

    /// Given data type `S`, this returns the string representation used by numpy.
    /// For common data types, these are instantiated in `io/numpy.cpp`.
    template<class S>
    const char* data_type_string();

    /*!
     * \brief Generates the npy description string based on an Eigen matrix
     * \tparam Derived The derived type of the Eigen matrix
     * \param matrix Const reference to the eigen matrix.
     * \return A string for the description dict of the matrix.
     */
    template<class Derived>
    std::string make_npy_description(const Eigen::DenseBase<Derived>& matrix) {
        return make_npy_description(data_type_string<typename Derived::Scalar>(), !Derived::IsRowMajor, matrix.rows(), matrix.cols());
    }

    /*!
     * \brief Loads a matrix from a numpy array.
     */
    types::DenseRowMajor<real_t> load_matrix_from_npy(std::istream& source);
    types::DenseRowMajor<real_t> load_matrix_from_npy(const std::string& path);

    /*!
     * \brief Saves a matrix to a numpy array.
     */
    void save_matrix_to_npy(std::ostream& source, const types::DenseRowMajor<real_t>&);
    void save_matrix_to_npy(const std::string& path, const types::DenseRowMajor<real_t>&);
}

#endif //DISMEC_NUMPY_H

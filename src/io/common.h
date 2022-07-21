// Copyright (c) 2021, Aalto University, developed by Erik Schultheis
// All rights reserved.
//
// SPDX-License-Identifier: MIT

#ifndef DISMEC_COMMON_H
#define DISMEC_COMMON_H

#include <stdexcept>
#include "matrix_types.h"
#include "spdlog/fmt/fmt.h"
#include "utils/throw_error.h"

/*! \file
 * \brief building blocks for io procedures that are used by multiple io subsystems
 * \details This file defines functions that do some generic io procedures, such as parsing
 * or writing single vectors in dense or sparse text format.
 * \note This needs to be a macro, because THROW_EXCEPTION automatically captures all variables of the current
 * scope for use in the defined lambda expression.
 */
// NOLINTNEXTLINE(cppcoreguidelines-macro-usage)
#define THROW_ERROR(...) THROW_EXCEPTION(std::runtime_error, __VA_ARGS__)

namespace dismec::io {
    namespace detail {
        std::string print_char(char c);
    }

    /// Parses an integer using `std::strtol`. In contrast to the std function, the output parameter is const here.
    inline long parse_long(const char* string, const char** out) {
        char* out_ptr = nullptr;
        long result = std::strtol(string, &out_ptr, 10); // NOLINT(cppcoreguidelines-avoid-magic-numbers)
        *out = out_ptr;
        return result;
    }

    /*!
     * \brief parses sparse features given in index:value text format.
     * \details The `callback` is called with index and value of each feature. The features are expected for be
     * integers immediately followed by a colon `:`, followed by a floating point number (see e.g. \ref xmc-data).
     * \param feature_part Pointer to the part of the line where the features start, e.g. the return value of
     * `parse_labels`. Has to be `\0` terminated.
     * \param callback A function that takes two parameters, the first of type long which is the feature index, and
     * the second of type double which is the feature value.
     * \throws If number parsing fails, or the format is not as expected.
     */
    template<class F>
    void parse_sparse_vector_from_text(const char* feature_part, F&& callback) {
        const char* start = feature_part;
        while(*feature_part) {
            const char* result = nullptr;
            errno = 0;
            long index = parse_long(feature_part, &result);
            if (result == feature_part) {
                // parsing failed -- either, wrong format, or we have reached some trailing spaces
                // we verify this here explicitly, again using IELF to keep this out of the hot code path
                bool is_error = [&](){
                    for(const char* scan = feature_part; *scan; ++scan) {
                        if(!std::isspace(*scan)) {
                            return true;
                        }
                    }
                    return false;
                }();
                if(!is_error) {
                    return;
                }
                THROW_ERROR("Error parsing feature. Missing feature index.");
            } else if(*result != ':') {
                THROW_ERROR("Error parsing feature index. Expected ':' at position {}, got '{}'", (result - start), detail::print_char(*result));
            } else if(errno != 0) {
                THROW_ERROR("Error parsing feature index. Errno={}: '{}'", errno, strerror(errno));
            }

            errno = 0;
            char* after_feature = nullptr;
            double value = std::strtod(result+1, &after_feature);
            if(result + 1 == after_feature) {
                THROW_ERROR("Error parsing feature: Missing feature value.");
            } else if(errno != 0) {
                THROW_ERROR("Error parsing feature value. Errno={}: '{}'", errno, strerror(errno));
            }

            feature_part = after_feature;
            callback(index, value);
        }
    }

    /*!
     * \brief Writes the given vector as space-separated human-readable numbers
     * \details This function does not check if the writing was successful.
     * \return For convenience, this function returns a reference to the stream.
     */
    std::ostream& write_vector_as_text(std::ostream& stream, const Eigen::Ref<const DenseRealVector>& data);

    /*!
     * \brief Reads the given vector as space-separated human-readable numbers
     * \details This function expects that `data` is already of the correct size, and tries to read as many items
     * as this specifies.
     * \return For convenience, this function returns a reference to the stream.
     */
    std::istream& read_vector_from_text(std::istream& stream, Eigen::Ref<DenseRealVector> data);

    // binary read and write functions
    template<class T>
    void binary_dump(std::streambuf& target, const T* begin, const T* end) {
        static_assert(std::is_pod_v<T>, "Can only binary dump POD types");
        auto num_bytes = (end - begin) * sizeof(T);
        auto wrote = target.sputn(reinterpret_cast<const char*>(begin), (end - begin) * sizeof(T));
        if(num_bytes != wrote) {
            THROW_ERROR("Expected to write {} bytes, but wrote only {}", num_bytes, wrote);
        }
    }

    template<class T>
    void binary_load(std::streambuf& target, T* begin, T* end) {
        static_assert(std::is_pod_v<T>, "Can only binary load POD types");
        auto num_bytes = (end - begin) * sizeof(T);
        auto read = target.sgetn(reinterpret_cast<char*>(begin), num_bytes);
        if(num_bytes != read) {
            THROW_ERROR("Expected to read {} bytes, but got only {}", num_bytes, read);
        }
    }

    /// \brief Collects the rows and columns parsed from a plain-text matrix file
    struct MatrixHeader {
        long NumRows;
        long NumCols;
    };

    /// Given a string containing a matrix header, parses it into rows and columns.
    /// The input string should contain exactly two positive integers, otherwise and exception will be thrown.
    MatrixHeader parse_header(const std::string& content);

    /*!
     * \brief Binary Sparse Matrix in List-of-Lists format
     * \details Since the matrix is binary, we do not store the actual values, only the location of the non-zeros.
     */
    struct LoLBinarySparse {
        long NumRows;
        long NumCols;
        std::vector<std::vector<long>> NonZeros;
    };

    /// Reads a sparse binary matrix file in the format index:1.0 as a list-of-list of the non-zero entries.
    /// The first line of the file should be the shape of the matrix.
    LoLBinarySparse read_binary_matrix_as_lil(std::istream& source);
}

#endif //DISMEC_COMMON_H

// Copyright (c) 2021, Aalto University, developed by Erik Schultheis
// All rights reserved.
//
// SPDX-License-Identifier: MIT

#include "common.h"

using namespace dismec;

std::string io::detail::print_char(char c) {
    std::string result;
    if(std::isprint(c) != 0) {
        result.push_back(c);
        return result;
    }
    result.push_back('\\');
    result.append(std::to_string((int)c));
    return result;
}

std::ostream& io::write_vector_as_text(std::ostream& stream, const Eigen::Ref<const DenseRealVector>& data)
{
    if(data.size() == 0) {
        return stream;
    }

    // size is > 0, so -1 is safe
    for(int i = 0; i < data.size() - 1; ++i) {
        stream << data.coeff(i) << ' ';
    }
    // no trailing space
    stream << data.coeff(data.size() - 1);

    return stream;
}

std::istream& io::read_vector_from_text(std::istream& stream, Eigen::Ref<DenseRealVector> data) {
    for (int j = 0; j < data.size(); ++j) {
        stream >> data.coeffRef(j);
    }

    if (stream.bad()) {
        THROW_ERROR("Error while reading a {} element dense vector from text data", data.size());
    }

    return stream;
}

io::MatrixHeader io::parse_header(const std::string& content) {
    std::stringstream parse_header{content};
    long NumRows = -1;
    long NumCols = -1;

    parse_header >> NumRows >> NumCols;
    if (parse_header.fail()) {
        THROW_ERROR("Error parsing header: '{}'", content);
    }

    // check validity of numbers
    if(NumRows <= 0) {
        THROW_ERROR("Invalid number of rows {} specified in header '{}'", NumRows, content);
    }
    if(NumCols <= 0) {
        THROW_ERROR("Invalid number of rows {} specified in header '{}'", NumCols, content);
    }

    std::string rest;
    parse_header >> rest;
    if(!rest.empty()) {
        THROW_ERROR("Found additional text '{}' in header '{}'", rest, content);
    }

    return {NumRows, NumCols};
}

io::LoLBinarySparse io::read_binary_matrix_as_lil(std::istream& source) {
    // for now, labels are assumed to come from a text file
    std::string line_buffer;
    std::getline(source, line_buffer);
    auto header = parse_header(line_buffer);

    std::vector<std::vector<long>> label_data;
    label_data.resize(header.NumCols);

    long example = 0;
    long num_rows = header.NumRows;
    long num_cols = header.NumCols;

    while (std::getline(source, line_buffer)) {
        if (line_buffer.empty())
            continue;
        if (line_buffer.front() == '#')
            continue;

        if(example >= num_rows) {
            THROW_ERROR("Encountered row {:5} but only expected {:5} rows.", example, num_rows);
        }

        try {
            io::parse_sparse_vector_from_text(line_buffer.c_str(), [&](long index, double value) {
                long adjusted_index = index;
                if (adjusted_index >= num_cols || adjusted_index < 0) {
                    THROW_ERROR("Encountered index {:5}. Number of columns "
                                "was specified as {}.", index, num_cols);
                }
                // filter out explicit zeros
                if (value != 1) {
                    THROW_ERROR("Encountered value {} at index {}.", value, index);
                }
                label_data[adjusted_index].push_back(example);
            });
        } catch (std::runtime_error& e) {
            THROW_ERROR("Error reading example {}: {}.", example + 1, e.what());
        }
        ++example;
    }
    return {num_rows, num_cols, std::move(label_data)};
}

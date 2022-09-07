// Copyright (c) 2021, Aalto University, developed by Erik Schultheis
// All rights reserved.
//
// SPDX-License-Identifier: MIT

#include "numpy.h"
#include "io/common.h"
#include <ostream>
#include <fstream>
#include <cstdint>
#include "spdlog/fmt/fmt.h"
#include "spdlog/spdlog.h"

using namespace dismec;

namespace {
    constexpr const char MAGIC[] = {'\x93', 'N', 'U', 'M', 'P', 'Y', '\x03', '\x00'};
    constexpr const int MAGIC_SIZE = 6;
    constexpr const unsigned NPY_PADDING = 64u;
}

bool io::is_npy(std::istream& source) {
    char buffer[MAGIC_SIZE];
    auto current_position = source.tellg();
    if(auto num_read = source.readsome(buffer, sizeof(buffer)); num_read != sizeof(buffer)) {
        THROW_ERROR("Error when trying to read magic bytes. Read on {} bytes.", num_read);
    }
    source.seekg(current_position);
    return (std::memcmp(buffer, MAGIC, sizeof(buffer)) == 0);
}

void io::write_npy_header(std::streambuf& target, std::string_view description) {
    target.sputn(MAGIC, sizeof(MAGIC));
    std::size_t total_length = sizeof(MAGIC) + sizeof(std::uint32_t) + description.size() + 1;
    unsigned padding = NPY_PADDING - total_length % NPY_PADDING;

    std::uint32_t header_length = description.size() + padding + 1;
    target.sputn(reinterpret_cast<const char*>(&header_length), sizeof(header_length));
    target.sputn(description.data(), description.size());
    for(unsigned i = 0; i < padding; ++i) {
        target.sputc('\x20');
    }
    if(target.sputc('\n') != '\n') {
        THROW_ERROR("Could not write terminating newline to npy header");
    }
}

std::string io::make_npy_description(std::string_view dtype_desc, bool column_major, std::size_t size) {
    return fmt::format(R"({{"descr": "{}", "fortran_order": {}, "shape": ({},)}})", dtype_desc, column_major ? "True" : "False", size);
}

std::string io::make_npy_description(std::string_view dtype_desc, bool column_major, std::size_t rows, std::size_t cols) {
    return fmt::format(R"({{"descr": "{}", "fortran_order": {}, "shape": ({}, {})}})", dtype_desc, column_major ? "True" : "False", rows, cols);
}

#define REGISTER_DTYPE(TYPE, STRING)        \
template<>                                  \
const char* data_type_string<TYPE>() {      \
    return STRING;                          \
}

namespace dismec::io {
    REGISTER_DTYPE(float, "<f4");
    REGISTER_DTYPE(double, "<f8");
    REGISTER_DTYPE(std::int32_t, "<i4");
    REGISTER_DTYPE(std::int64_t, "<i8");
    REGISTER_DTYPE(std::uint32_t, "<u4");
    REGISTER_DTYPE(std::uint64_t, "<u8");
}

namespace {
    std::uint32_t read_header_length(std::streambuf& source) {
        auto read_raw = [&](auto& target){
            auto num_read = source.sgetn(reinterpret_cast<char*>(&target), sizeof(target));
            if(num_read != sizeof(target)) {
                THROW_ERROR("Unexpected end of data while reading header length");
            }
        };

        int major = source.sbumpc();
        int minor = source.sbumpc();

        if(major == 2 || major == 3) {
            std::uint32_t header_length;
            read_raw(header_length);
            return header_length;
        }
        if (major == 1) {
            std::uint16_t short_header_length;
            read_raw(short_header_length);
            return short_header_length;
        }
        THROW_ERROR("Unknown npy file format version {}.{} -- {}", major, minor, source.pubseekoff(0, std::ios_base::cur, std::ios_base::in));
    };

    long skip_whitespace(std::string_view source, long position) {
        while(std::isspace(source[position]) != 0 && position < ssize(source)) {
            ++position;
        }

        return position;
    }

    /// This function parses a single element from a python dict literal
    std::pair<std::string_view, std::string_view> read_key_value(std::string_view source) {
        auto source_end = ssize(source);

        // skip all initial whitespace
        long position = skip_whitespace(source, 0);
        if(position == source_end) {
            THROW_ERROR("received only whitespace");
        }

        char open_quote = source[position];
        long key_start = position;
        // next, we should get a dictionary key. This is indicated by quotation marks
        if(open_quote != '"' && open_quote != '\'') {
            THROW_ERROR("Expected begin of string ' or \" for parsing dictionary key. Got {}.", open_quote);
        }

        std::size_t key_end = source.find(open_quote, key_start + 1);
        if(key_end == std::string_view::npos) {
            THROW_ERROR("Could not find matching closing quotation mark `{}` for key string", open_quote);
        }

        // next, we expect a colon to separate the value
        position = skip_whitespace(source, to_long(key_end) + 1);
        if(position == source_end) {
            THROW_ERROR("Could not find : that separates key and value");
        }

        if(source[position] != ':') {
            THROW_ERROR("Expected : to separate key and value, got {}", source[position]);
        }

        position = skip_whitespace(source, position + 1);
        if(position == source_end) {
            THROW_ERROR("Missing feature");
        }

        const char openers[] = {'"', '\'', '(', '[', '{'};
        const char closers[] = {'"', '\'', ')', ']', '}'};

        // to keep the code simple, we do not support nesting or escaping of
        // delimiters. For the intended use case, that should be enough, but
        // it means that this function cannot be used for general npy files

        long value_start = position;
        char expect_close = 0;
        while(position < source_end) {
            char current = source[position];
            if(expect_close == 0) {
                // if we are not in a nested expression, the end of the current value is reached if we find a comma
                // or closing brace
                if(current == ',' || current == '}') {
                    return {{source.begin() + key_start + 1, key_end - key_start - 1},
                            {source.begin() + value_start, std::size_t(position - value_start)}};
                }

                // if we are opening a nested expression, figure out which char we are waiting for next
                switch (current) {
                    case '"':
                    case '\'':
                    case '(':
                    case '[':
                    case '{':
                    {
                        for(int i = 0; i < to_long(sizeof(openers)); ++i) {
                            if(openers[i] == current) {
                                expect_close = closers[i];
                            }
                        }
                    }
                    default: break;
                }
            } else if(current == expect_close) {
                expect_close = 0;
            }
            ++position;
        }

        if(expect_close != 0) {
            THROW_ERROR("Expected closing {}, but reached end of input", expect_close);
        }
        THROW_ERROR("Expected } or , to signal end of input");
    }

    io::NpyHeaderData parse_description(std::string_view view) {
        view.remove_prefix(1);

        io::NpyHeaderData result;

        bool has_descr = false;
        bool has_order = false;
        bool has_shape = false;
        for(int i = 0; i < 3; ++i) {
            // can't use structured bindings here, because apparently they cannot be captured in the THROW_ERROR lambda
            auto kv = read_key_value(view);
            auto key = kv.first;
            auto value = kv.second;
            view = view.substr(value.end() - view.begin() + 1);

            if(key == "descr") {
                if(value.front() != '\'' && value.front() != '"') {
                    THROW_ERROR("expected string for descr, got '{}'", value);
                }
                result.DataType = value.substr(1, value.size() - 2);
                has_descr = true;
            } else if (key == "fortran_order") {
                if(value == "False" || value == "0") {
                    result.ColumnMajor = false;
                } else if (value == "True" || value == "1") {
                    result.ColumnMajor = true;
                } else {
                    std::string val_str{value};
                    THROW_ERROR("unexpected value '{}' for fortran_order", val_str);
                }
                has_order = true;
            } else if(key == "shape") {
                if(value.at(0) != '(') {
                    THROW_ERROR("expected ( to start tuple for shape");
                }
                auto sep = value.find(',');
                if(sep == std::string::npos) {
                    THROW_ERROR("Expected comma in tuple definition");
                }

                const char* endptr = nullptr;
                errno = 0;
                result.Rows = io::parse_long( value.begin() + 1, &endptr);
                if(errno != 0 || endptr == value.begin() + 1) {
                    THROW_ERROR("error while trying to parse number for size");
                }
                if(result.Rows < 0) {
                    THROW_ERROR("Number of rows cannot be negative. Got {}", result.Rows);
                }

                result.Cols = io::parse_long( value.begin() + sep + 1, &endptr);
                if(errno != 0) {
                    THROW_ERROR("error while trying to parse number for size");
                }

                if(result.Cols < 0) {
                    THROW_ERROR("Number of rows cannot be negative. Got {}", result.Cols);
                }

                has_shape = true;
            } else {
                std::string key_str{key};
                THROW_ERROR("unexpected key '{}'", key_str);
            }
        }

        bool closed_dict = false;
        for(const auto& c : view) {
            if(std::isspace(c) != 0) continue;
            if(c == '}' && !closed_dict) {
                closed_dict = true;
                continue;
            }
            THROW_ERROR("Trailing '{}'", c);
        }

        if(!has_descr) {
            THROW_ERROR("Missing 'descr' entry in dict");
        }

        if(!has_order) {
            THROW_ERROR("Missing 'fortran_order' entry in dict");
        }

        if(!has_shape) {
            THROW_ERROR("Missing 'shape' entry in dict");
        }

        return result;
    }
}
#include <iostream>
io::NpyHeaderData io::parse_npy_header(std::streambuf& source) {
    std::array<char, MAGIC_SIZE> magic{};
    source.sgetn(magic.data(), MAGIC_SIZE);
    for(int i = 0; i < MAGIC_SIZE; ++i) {
        if(magic[i] != MAGIC[i]) {
            THROW_ERROR("Magic bytes mismatch");
        }
    }

    std::uint32_t header_length = read_header_length(source);

    std::string header_buffer(header_length, '\0');
    if(auto num_read = source.sgetn(header_buffer.data(), header_length); num_read != header_length) {
        THROW_ERROR("Expected to read a header of size {}, but only got {} elements", header_length, num_read);
    }

    // OK, now for the actual parsing of the dict.
    if(header_buffer.at(0) != '{') {
        THROW_ERROR("Expected data description dict to start with '{{', got '{}'. Header is: {}", header_buffer.at(0), header_buffer );
    }

    if(header_buffer.back() != '\n') {
        THROW_ERROR("Expected newline \\n at end of header \"{}\"", header_buffer );
    }

    return parse_description(header_buffer);
}

namespace {
    template<class T>
    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> load_matrix_from_npy_imp(std::streambuf& source) {
        auto header = io::parse_npy_header(source);
        if(header.DataType != io::data_type_string<T>()) {
            THROW_ERROR("Unsupported data type {}", header.DataType);
        }
        if(header.ColumnMajor) {
            THROW_ERROR("Currently, only row-major npy files can be read");
        }

        // load the matrix row-by-row, to make sure this works even if Eigen decides to include padding
        Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> target(header.Rows, header.Cols);
        for(int row = 0; row < target.rows(); ++row) {
            auto row_data = target.row(row);
            io::binary_load(source, row_data.data(), row_data.data() + row_data.size());
        }

        return target;
    }

    template<class T>
    void save_matrix_to_npy_imp(std::streambuf& target, const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>& matrix) {
        io::write_npy_header(target, io::make_npy_description(matrix));

        // save the matrix row-by-row, to make sure this works even if Eigen decides to include padding
        for(int row = 0; row < matrix.rows(); ++row) {
            const auto& row_data = matrix.row(row);
            io::binary_dump(target, row_data.data(), row_data.data() + row_data.size());
        }
    }

}

Eigen::Matrix<real_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> io::load_matrix_from_npy(std::istream& source) {
    return load_matrix_from_npy_imp<real_t>(*source.rdbuf());
}

Eigen::Matrix<real_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> io::load_matrix_from_npy(const std::string& path) {
    std::ifstream file(path);
    if(!file.is_open()) {
        THROW_ERROR("Could not open file {} for reading.", path)
    }
    return load_matrix_from_npy(file);
}

void io::save_matrix_to_npy(std::ostream& source,
                            const Eigen::Matrix<real_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>& matrix) {
    save_matrix_to_npy_imp(*source.rdbuf(), matrix);
}

void io::save_matrix_to_npy(const std::string& path,
                            const Eigen::Matrix<real_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>& matrix) {
    std::ofstream file(path);
    if(!file.is_open()) {
        THROW_ERROR("Could not open file {} for writing.", path)
    }
    return save_matrix_to_npy(file, matrix);
}


#include "doctest.h"
#include <sstream>

TEST_CASE("numpy header with given description") {
    std::stringstream target;
    std::string description = "{'descr': '<f8', 'fortran_order': False, 'shape': (3,), }";
    std::string ground_truth("\x93NUMPY\x03\x00\x74\x00\x00\x00{'descr': '<f8', 'fortran_order': False, 'shape': (3,), }                                                          \n", 128);
    io::write_npy_header(*target.rdbuf(), description);

    std::string new_str = target.str();
    CHECK(new_str == ground_truth);
}

TEST_CASE("header length test") {
    std::stringstream src;

    SUBCASE("read valid length v2/3") {
        src.str(std::string("\x03\x00s\x00\x00\x00", MAGIC_SIZE));
        CHECK(read_header_length(*src.rdbuf()) == 's');
        CHECK(src.rdbuf()->pubseekoff(0, std::ios_base::cur, std::ios_base::in) == 6);
    }

    SUBCASE("read valid length v1") {
        src.str(std::string("\x01\x00s\x00\x00\x00", MAGIC_SIZE));
        CHECK(read_header_length(*src.rdbuf()) == 's');
        CHECK(src.rdbuf()->pubseekoff(0, std::ios_base::cur, std::ios_base::in) == 4);
    }

    SUBCASE("invalid version") {
        src.str(std::string("\x04\x00s\x00\x00\x00", MAGIC_SIZE));
        CHECK_THROWS(read_header_length(*src.rdbuf()));
    }

    SUBCASE("end of data") {
        src.str(std::string("\x03\x00s\x00", 4));
        CHECK_THROWS(read_header_length(*src.rdbuf()));
    }
}

TEST_CASE("read key value error check") {
    CHECK_THROWS(read_key_value("  "));
    CHECK_THROWS(read_key_value(" key'"));
    CHECK_THROWS(read_key_value("'key'   "));
    CHECK_THROWS(read_key_value("'key':  "));
    CHECK_THROWS(read_key_value("'key  "));
    CHECK_THROWS(read_key_value("'key' error:"));
    CHECK_THROWS(read_key_value("'key': 'value"));
    CHECK_THROWS(read_key_value("'key': (1, 2]"));
}

TEST_CASE("read key value test") {
    std::string input;
    std::string key;
    std::string value;

    SUBCASE("double quotes") {
        input = "{\"key\": value}";
        key = "key";
        value = "value";
    }

    SUBCASE("single quotes") {
        input = "{'key': value}";
        key = "key";
        value = "value";
    }

    SUBCASE("tuple value") {
        input = "{'key': (1, 2, 3)}";
        key = "key";
        value = "(1, 2, 3)";
    }

    SUBCASE("multiple entries") {
        input = "{'key': a, \"other key\": b}";
        key = "key";
        value = "a";
    }

    SUBCASE("nested quotes") {
        input = "{\"key_with'\":  value}";
        key = "key_with'";
        value = "value";
    }

    SUBCASE("quoted value") {
        input = "{'key': 'a value that contains } and \" and ) and ]'}";
        key = "key";
        value = "'a value that contains } and \" and ) and ]'";
    }

    std::string_view dict_contents = input;
    dict_contents.remove_prefix(1);
    auto [got_key, got_value] = read_key_value(dict_contents);
    CHECK(got_key == key);
    CHECK(got_value == value);
}

TEST_CASE("parse description -- valid") {
    SUBCASE("f8 c order vector") {
        auto data = parse_description("{'descr': '<f8', 'fortran_order': False, 'shape': (3,), }");
        CHECK(data.ColumnMajor == false);
        CHECK(data.Rows == 3);
        CHECK(data.Cols == 0);
        CHECK(data.DataType == "<f8");
    }
    SUBCASE("reordered") {
        auto data = parse_description("{'fortran_order': False, 'shape': (3,), 'descr': '<f8'}");
        CHECK(data.ColumnMajor == false);
        CHECK(data.Rows == 3);
        CHECK(data.Cols == 0);
        CHECK(data.DataType == "<f8");
    }
    SUBCASE("i4 f order matrix no trailing comma") {
        auto data = parse_description("{'descr': \"<i4\", 'fortran_order': 1, 'shape': (5 , 7)}");
        CHECK(data.ColumnMajor == true);
        CHECK(data.Rows == 5);
        CHECK(data.Cols == 7);
        CHECK(data.DataType == "<i4");
    }
    SUBCASE("f8 c order matrix no whitespace") {
        auto data = parse_description("{'descr':'<f8','fortran_order':0,'shape':(5,7)}");
        CHECK(data.ColumnMajor == false);
        CHECK(data.Rows == 5);
        CHECK(data.Cols == 7);
        CHECK(data.DataType == "<f8");
    }
}

TEST_CASE("parse description -- errors") {
    SUBCASE("wrong value") {
        CHECK_THROWS(parse_description("{'descr': '<f8', 'fortran_order': Unknown, 'shape': (3,), }"));
        CHECK_THROWS(parse_description("{'descr': (5, 4), 'fortran_order': False, 'shape': (3,), }"));
        CHECK_THROWS(parse_description("{'descr': '<f8', 'fortran_order': False, 'shape': 8 }"));
        CHECK_THROWS(parse_description("{'descr': 5, 'fortran_order': False, 'shape': (3,) }"));
        CHECK_THROWS(parse_description("{'descr': '<f8', 'fortran_order': False, 'shape': (3) }"));
        CHECK_THROWS(parse_description("{'descr': '<f8', 'fortran_order': False, 'shape': (a,) }"));
    }

    SUBCASE("missing key") {
        CHECK_THROWS(parse_description("{'fortran_order':0,'shape':(5,7)}"));
        CHECK_THROWS(parse_description("{'descr':'<f8','shape':(5,7)}"));
        CHECK_THROWS(parse_description("{'descr':'<f8','fortran_order':0"));
    }

    SUBCASE("unexpected key") {
        CHECK_THROWS(parse_description("{'descr':'<f8','fortran_order':0,'shape':(5,7), 'other': 'value'}"));
        CHECK_THROWS(parse_description("{'descr':'<f8','other': 'value', 'fortran_order':0,'shape':(5,7)}"));
    }
}

TEST_CASE("make description") {
    CHECK(io::make_npy_description("<f8", false, 5) == "{\"descr\": \"<f8\", \"fortran_order\": False, \"shape\": (5,)}");
    CHECK(io::make_npy_description(">i4", true, 17) == "{\"descr\": \">i4\", \"fortran_order\": True, \"shape\": (17,)}");
    CHECK(io::make_npy_description("<f8", false, 7, 5) == "{\"descr\": \"<f8\", \"fortran_order\": False, \"shape\": (7, 5)}");
}

TEST_CASE("save/load round trip") {
    std::ostringstream save_stream;
    types::DenseRowMajor<real_t> matrix = types::DenseRowMajor<real_t>::Random(4, 5);
    io::save_matrix_to_npy(save_stream, matrix);

    std::istringstream load_stream;
    load_stream.str(save_stream.str());
    auto ref = io::load_matrix_from_npy(load_stream);

    CHECK( matrix == ref );
}
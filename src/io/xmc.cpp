// Copyright (c) 2021, Aalto University, developed by Erik Schultheis
// All rights reserved.
//
// SPDX-License-Identifier: MIT

#include "io/xmc.h"
#include "io/common.h"
#include "data/data.h"
#include <fstream>
#include "spdlog/spdlog.h"
#include "spdlog/fmt/fmt.h"
#include "spdlog/stopwatch.h"

using namespace dismec;

namespace {
    /// \brief Collects the data from the header of an xmc file \ref xmc-data.
    struct XMCHeader {
        long NumExamples;
        long NumFeatures;
        long NumLabels;
    };

    /*!
     * \brief Parses the header (number of examples, features, labels) of an XMC dataset file.
     * \details parses the given line as the header of an xmc dataset. The header is expected to consist of
     * three whitespace separated positive integers in the order `#examples #features #labels`.
     * \throws If any of the parsed numbers is non-positive, or if the parsing itself fails.
     * \todo throw if there is more data on the line.
     */
    XMCHeader parse_xmc_header(const std::string& content) {
        std::stringstream parse_header{content};
        long NumExamples = -1;
        long NumFeatures = -1;
        long NumLabels = -1;

        parse_header >> NumExamples >> NumFeatures >> NumLabels;
        if (parse_header.fail()) {
            THROW_ERROR("Error parsing dataset header: '{}'", content);
        }

        // check validity of numbers
        if(NumExamples <= 0) {
            THROW_ERROR("Invalid number of examples {} in specified in header '{}'", NumExamples, content);
        }
        if(NumFeatures <= 0) {
            THROW_ERROR("Invalid number of features {} in specified in header '{}'", NumFeatures, content);
        }
        if(NumLabels <= 0) {
            THROW_ERROR("Invalid number of labels {} in specified in header '{}'", NumLabels, content);
        }

        std::string rest;
        parse_header >> rest;
        if(!rest.empty()) {
            THROW_ERROR("Found additional text '{}' in header '{}'", rest, content);
        }

        return {NumExamples, NumFeatures, NumLabels};
    }

    /*!
     * \brief Extracts number of nonzero features for each instance.
     * \details This iterates over the lines in `source` and extracts the number of nonzero features
        for each line. You can optionally supply the number of examples expected, which will be
        used to reserve memory in the counter buffer. Completely empty lines are ignored,
        as are lines that start with # (see \ref xmc-data). This function does not validate that the data is given
        in the correct format. It just counts the number of occurences of the colon `:` character, which in correctly
        formatted lines corresponds to the number of labels.
        \param source The stream from which to read. Should not contain the header.
        \param num_examples Number of examples to expect. This is used to reserve space in the result vector. Optional,
        but if not given may result in additional allocations being performed and/or too much memory being used.
        \todo also count number of labels based on `,`, then we can reserve also the label vector
    */
    std::vector<long> count_features_per_example(std::istream& source, std::size_t num_examples = 100'000)
    {
        std::string line_buffer;
        std::vector<long> features_per_example;
        features_per_example.reserve(num_examples);

        // next, we iterate over the entire dataset once to gather
        // more statistics so we can pre-allocate the corresponding buffers
        while (std::getline(source, line_buffer))
        {
            // we don't parse empty lines or comment lines
            if (line_buffer.empty())
                continue;
            if(line_buffer.front() == '#')
                continue;

            // features are denoted by index:value, so the number of features is equal to the
            // number of colons in the string
            long num_ftr = std::count(begin(line_buffer), end(line_buffer), ':');
            features_per_example.push_back(num_ftr);
        }

        return features_per_example;
    }

    /*!
     * \brief parses the labels part of a xmc dataset line.
     * \details Returns a pointer to where the label part ends and feature parsing should start.
     * Each labels is parsed as an integer number (with possibly leading spaces), followed by either
     * a comma, indicating more labels, or a whitespace indicating this was the last label. If the first character
     * is a white space, this is interpreted as the absence of labels. This function expects that comments and empty
     * lines have already been skipped.
     * \param line Pointer to a null-terminated string.
     * \param callback A function that takes a single parameter of type long, which will be called for each label that
     * is encountered.
     * \throws If number parsing fails, or the format is not as expected.
     * \return Pointer into the string given by `line` from which the feature parsing should start.
     */
    template<class F>
    const char* parse_labels(const char* line, F&& callback) {
        const char *last = line;
        if (!std::isspace(*line)) {
            // then read as many integers as we can, always skipping exactly one character between. If an integer
            // is followed by a colon, it was a feature id
            while (true) {
                const char *result = nullptr;
                errno = 0;
                long read = dismec::io::parse_long(last, &result);
                // was there a number to read?
                if(result == last) {
                    THROW_ERROR("Error parsing label. Expected a number.");
                } else if(errno != 0) {
                    THROW_ERROR("Error parsing label. Errno={}: '{}'", errno, strerror(errno));
                }
                if (*result == ',') {
                    // fine, more labels to come
                } else if (std::isspace(*result) != 0 || *result == '\0') {
                    // fine, this was the last label
                    callback(read);
                    return result;
                } else {
                    // everything else is not accepted
                    THROW_ERROR("Error parsing label. Expected ',', got '{}', '{}'", errno, *result ? *result : '0', line);
                }
                callback(read);
                last = result + 1;
            }
        }
        return last;
    }

    /*!
     * \brief iterates over the lines in `source` and puts the corresponding features and labels into the given buffers.
     * \details These are expected to be pre-allocated with the correct size. This means that `feature_buffer` has to
     * be an empty sparse matrix of dimensions `num_examples x num_features`, and `label_buffer` should be a
     * vector (of empty vectors) of size `num_labels`. To speed up reading, it is advisable to reserve the
     * appropriate amount of space in the buffers, though this is not technically necessary.
     * \tparam IndexOffsetThe template-parameter `IndexOffset` is used to switch between 0-based and 1-based indexing.
     * Internally, we always use 0-based indexing, so if `IndexOffset != 0` we subtract the offset from the
     * indices that are read from the file.
     * \param source istream from which the lines are read. If the file has a header, this has to be skipped before calling
     * `read_into_buffers`.
     * \param feature_buffer Shared pointer to an empty sparse matrix where rows correspond to examples and columns correspond
     * to features.
     * \param label_buffer Vector of vectors, where the inner vectors will list the indices of the examples in which the
     * label (as given by the outer index) is present. The outer vector has to be of size `num_label`.
     * \throws If feature, label or example index are out of bounds.
     */
    template<long IndexOffset>
    void read_into_buffers(std::istream& source,
                           SparseFeatures& feature_buffer,
                           std::vector<std::vector<long>>& label_buffer)
    {
        std::string line_buffer;
        auto num_labels = ssize(label_buffer);
        auto num_features = feature_buffer.cols();
        auto num_examples = feature_buffer.rows();
        long example = 0;

        while (std::getline(source, line_buffer)) {
            if (line_buffer.empty())
                continue;
            if (line_buffer.front() == '#')
                continue;

            if(example >= num_examples) {
                THROW_ERROR("Encountered example number index {:5} but buffers only expect {:5} examples.", example, num_examples);
            }

            try {
                auto label_end = parse_labels(line_buffer.data(), [&](long lbl) {
                    long adjusted_label = lbl - IndexOffset;
                    if (adjusted_label >= num_labels || adjusted_label < 0) {
                        THROW_ERROR("Encountered label {:5}, but number of labels "
                                    "was specified as {}.", lbl, num_labels);
                    }
                    label_buffer[adjusted_label].push_back(example);
                });

                dismec::io::parse_sparse_vector_from_text(label_end, [&](long index, double value) {
                    long adjusted_index = index - IndexOffset;
                    if (adjusted_index >= num_features || adjusted_index < 0) {
                        THROW_ERROR("Encountered feature index {:5} with value {}. Number of features "
                                    "was specified as {}.", index, value, num_features);
                    }
                    // filter out explicit zeros
                    if (value != 0) {
                        if(std::isnan(value)) {
                            THROW_ERROR("Encountered feature index {:5} with value {}.", index, value);
                        }
                        feature_buffer.insert(example, adjusted_index) = static_cast<real_t>(value);
                    }
                });
            } catch (std::runtime_error& e) {
                THROW_ERROR("Error reading example {}: {}.", example + 1, e.what());
            }
            ++example;
        }
    }
}

dismec::MultiLabelData dismec::io::read_xmc_dataset(const std::filesystem::path& source_path, IndexMode mode) {
    std::fstream source(source_path, std::fstream::in);
    if (!source.is_open()) {
        throw std::runtime_error(fmt::format("Cannot open input file {}", source_path.c_str()));
    }

    return read_xmc_dataset(source, source_path.c_str(), mode);
}

dismec::MultiLabelData dismec::io::read_xmc_dataset(std::istream& source, std::string_view name, IndexMode mode) {
    // for now, do what the old code does: iterate twice, once to count and once to read
    std::string line_buffer;
    spdlog::stopwatch timer;

    std::getline(source, line_buffer);
    XMCHeader header = parse_xmc_header(line_buffer);

    spdlog::info("Loading dataset '{}' with {} examples, {} features and {} labels.",
                 name, header.NumExamples, header.NumFeatures, header.NumLabels);

    std::vector<long> features_per_example = count_features_per_example(source, header.NumExamples);
    if (ssize(features_per_example) != header.NumExamples) {
        THROW_EXCEPTION(std::runtime_error, "Dataset '{}' declared {} examples, but {} where found!",
                                             name, header.NumExamples, features_per_example.size());
    }

    // reset to beginning
    source.clear();
    source.seekg(0);

    // reserve space for all the features
    SparseFeatures x(header.NumExamples, header.NumFeatures);
    x.reserve(features_per_example);

    std::vector<std::vector<long>> label_data;
    label_data.resize(header.NumLabels);
    // TODO reserve space for correct number of labels

    // skip header this time
    std::getline(source, line_buffer);

    if(mode == IndexMode::ZERO_BASED) {
        read_into_buffers<0>(source, x, label_data);
    } else {
        read_into_buffers<1>(source, x, label_data);
    }

    x.makeCompressed();

    // remove excess memory
    for (auto& instance_list : label_data) {
        instance_list.shrink_to_fit();
    }

    spdlog::info("Finished loading dataset '{}' in {:.3}s.", name, timer);

    return {x.markAsRValue(), std::move(label_data)};
}

namespace {
    std::ostream& write_label_list(std::ostream& stream, const std::vector<int>& labels)
    {
        if(labels.empty()) {
            return stream;
        }

        // size is > 0, so this code is safe
        auto all_but_one = ssize(labels) - 1;
        for(int i = 0; i < all_but_one; ++i) {
            stream << labels[i] << ',';
        }
        // no trailing space
        stream << labels.back();

        return stream;
    }
}

void dismec::io::save_xmc_dataset(std::ostream& target, const MultiLabelData& data) {
    //! \todo insert proper checks that data is sparse
    target << data.num_examples() << " " << data.num_features() << " " << data.num_labels() << "\n";
    // for efficient saving, we need the labels in sparse row format, but for training we have them
    // in sparse column format.
    /// TODO handle this in a CSR format instead of LoL
    std::vector<std::vector<int>> all_labels(data.num_examples());
    for(label_id_t label{0}; label.to_index() < data.num_labels(); ++label) {
        for(const auto& instance : data.get_label_instances(label)) {
            all_labels[instance].push_back(label.to_index());
        }
    }

    if(!data.get_features()->is_sparse()) {
        throw std::runtime_error(fmt::format("XMC format requires sparse labels"));
    }
    const auto& feature_ptr = data.get_features()->sparse();

    for(int example = 0; example < data.num_examples(); ++example) {
        // first, write the label list
        write_label_list(target, all_labels[example]);
        // then, write the sparse features
        for (SparseFeatures::InnerIterator it(feature_ptr, example); it; ++it) {
            target << ' ' << it.col() << ':' << it.value();
        }
        target << '\n';
    }
}

void dismec::io::save_xmc_dataset(const std::filesystem::path& target_path, const MultiLabelData& data, int precision) {
    std::fstream target(target_path, std::fstream::out);
    if (!target.is_open()) {
        throw std::runtime_error(fmt::format("Cannot open output file {}", target_path.c_str()));
    }

    target.setf(std::fstream::fmtflags::_S_fixed, std::fstream::floatfield);
    target.precision(precision);
    save_xmc_dataset(target, data);
}

#include "doctest.h"

//! \test Checks that valid XMC headers are parsed correctly
TEST_CASE("parse valid header") {
    std::string input;
    SUBCASE("minimal") {
        input = "12 54 43";
    }
    SUBCASE("trailing space") {
        input = "12 54 43 ";
    }
    SUBCASE("tab separated") {
        input = "12\t54 \t 43 ";
    }
    auto valid = parse_xmc_header(input);
    CHECK(valid.NumExamples == 12);
    CHECK(valid.NumFeatures == 54);
    CHECK(valid.NumLabels == 43);
}


/// \test Check that invalid XMC headers are causing an exception. The headers are invalid if either the number of
/// data does not match, of if any of the supplied counts are non-positive.
TEST_CASE("parse invalid header") {
    // check number of arguments
    CHECK_THROWS(parse_xmc_header("6 1"));
    CHECK_THROWS(parse_xmc_header("6 1 5 1"));

    // we also know that something is wrong if any of the counts are <= 0
    CHECK_THROWS(parse_xmc_header("0 5 5"));
    CHECK_THROWS(parse_xmc_header("5 0 5"));
    CHECK_THROWS(parse_xmc_header("5 5 0"));
    CHECK_THROWS(parse_xmc_header("-1 5 5"));
    CHECK_THROWS(parse_xmc_header("5 -1 5"));
    CHECK_THROWS(parse_xmc_header("5 5 -1"));
}

/*!
 * \test Checks that the number of features are counted correctly. Since feature counting is only a very simple
 * approximation, that assumes correctly formatted input data, the only thing we test here is that 1) the counts are
 * correct and 2) empty lines and comments are skipped as they are supposed to.
 */
TEST_CASE("count features") {
    auto do_test = [](const std::string& source) {
        std::stringstream sstr(source);
        auto count = count_features_per_example(sstr, 10);
        REQUIRE(count.size() == 3);
        CHECK(count[0] == 2);
        CHECK(count[1] == 1);
        CHECK(count[2] == 4);
    };

    SUBCASE("minimal") {
        std::string source = R"(12 5:5.3 6:34
    4 6:4
    1 3:4  5:1 10:43 5:3)";
        do_test(source);
    }

    SUBCASE("comment") {
        std::string source = R"(12 5:5.3 6:34
    4 6:4
#   65:4
    1 3:4  5:1 10:43 5:3)";
        do_test(source);
    }

    SUBCASE("empty line") {
        std::string source = R"(12 5:5.3 6:34
    4 6:4

    1 3:4  5:1 10:43 5:3)";
        do_test(source);
    }

}

/*!
 * \test This test checks that XMC label parsing of incorrectly formatted lines results in errors.
 * Note that some formatting problems (such as a space before a comma: `5 ,1 10:3.0`) are perfectly
 * legal from a label parsing point of view, but will result in an error when the subsequent feature
 * parsing happens. We currently check the following error conditions:
 *  - trailing comma
 *  - not-a-number
 *  - not an integer
 *  - wrong separator between labels
*/
TEST_CASE("parse labels errors") {
    // trailing comma
    CHECK_THROWS(parse_labels("5,1, 5:2.0", [&](long v) {}));
    // not a number
    CHECK_THROWS(parse_labels("5, x", [&](long v) {}));
    // floating point
    CHECK_THROWS(parse_labels("5.5,1 10:3.0", [&](long v) {}));
    // wrong separator
    CHECK_THROWS(parse_labels("5;1 10:3.0", [&](long v) {}));
    // wrong spacing
    // an error like this will only be problematic for the subsequent feature parsing
    // the label parsing will already stop after the 5
    // CHECK_THROWS(parse_labels("5 ,1 10:3.0", [&](long v) {}));
}


/*!
 * \test This test checks that XMC label parsing of correctly formatted lines gives the
 * desired results. We check that whitespace is robust to space/tab, and that the empty
 * labels situation is correctly recognised.
 * \todo we should also check that the returned pointer is valid.
 */
TEST_CASE("parse labels") {
    auto run_test = [&](std::string source, const std::vector<long>& expect){
        int pos = 0;
        CAPTURE(source);
        try {
            parse_labels(source.data(), [&](long v) {
                CHECK(expect.at(pos) == v);
                ++pos;
            });
        } catch (std::runtime_error& err) {
            FAIL("parsing failed");
        }
        CHECK(expect.size() == pos);
    };

    SUBCASE("simple valid line") {
        run_test("1,3,4 12:4", {1, 3, 4});
    }
    SUBCASE("with space") {
        run_test("1, 3,\t4 12:4", {1, 3, 4});
    }
    SUBCASE("leading +") {
        run_test("+1, 3,\t4 12:4", {1, 3, 4});
    }
    SUBCASE("separated by space") {
        run_test("1,3,4\t12:4", {1, 3, 4});
    }
    SUBCASE("empty labels space") {
        run_test(" 12:4", {});
    }
    SUBCASE("empty labels tab") {
        run_test("\t12:4", {});
    }
    SUBCASE("missing features") {
        run_test("5, 1", {5, 1});
    }
}


/*!
 * \test This test verifies that `read_into_buffers` performs correct bounds checking
 * for its features, labels and examples. We check that both overflow (for examples, labels, features)
 * and underflow (for labels, features) are detected, and that these take into account whether indexing
 * is one-based or zero-based
*/
TEST_CASE("read into buffers bounds checks") {
    auto x = std::make_shared<SparseFeatures>(2, 3);

    std::vector<std::vector<long>> labels;
    labels.resize(2);
    std::stringstream source;

    SUBCASE("invalid feature") {
        source.str("1 2:0.5 3:0.5");
        SUBCASE("zero-base") {
            CHECK_THROWS(read_into_buffers<0>(source, *x, labels));
        }
        SUBCASE("one-base") {
            CHECK_NOTHROW(read_into_buffers<1>(source, *x, labels));
        }
    }

    SUBCASE("negative feature") {
        source.str("1 -1:0.5 1:0.5");
        SUBCASE("zero-base") {
            CHECK_THROWS(read_into_buffers<0>(source, *x, labels));
        }
        SUBCASE("one-base") {
            CHECK_THROWS(read_into_buffers<1>(source, *x, labels));
        }
    }

    SUBCASE("invalid label") {
        source.str("2 2:0.5");
        SUBCASE("zero-base") {
            CHECK_THROWS(read_into_buffers<0>(source, *x, labels));
        }
        SUBCASE("one-base") {
            CHECK_NOTHROW(read_into_buffers<1>(source, *x, labels));
        }
    }

    SUBCASE("negative  label") {
        source.str("-1 2:0.5");
        SUBCASE("zero-base") {
            CHECK_THROWS(read_into_buffers<0>(source, *x, labels));
        }
        SUBCASE("one-base") {
            CHECK_THROWS(read_into_buffers<1>(source, *x, labels));
        }
    }

    SUBCASE("invalid example") {
        source.str("0 0:0.5\n0 0:0.5\n0 0:0.5");
        SUBCASE("zero-base") {
            CHECK_THROWS(read_into_buffers<0>(source, *x, labels));
        }
        SUBCASE("one-base") {
            CHECK_THROWS(read_into_buffers<1>(source, *x, labels));
        }
    }

    SUBCASE("invalid zero label in one-based indexing") {
        source.str("0 2:0.5 2:0.5");
        SUBCASE("zero-base") {
            CHECK_NOTHROW(read_into_buffers<0>(source, *x, labels));
        }
        SUBCASE("one-base") {
            CHECK_THROWS(read_into_buffers<1>(source, *x, labels));
        }
    }

    SUBCASE("invalid zero feature in one-based indexing") {
        source.str("1 0:0.5 2:0.5");
        SUBCASE("zero-base") {
            CHECK_NOTHROW(read_into_buffers<0>(source, *x, labels));
        }
        SUBCASE("one-base") {
            CHECK_THROWS(read_into_buffers<1>(source, *x, labels));
        }
    }
}
// Copyright (c) 2021, Aalto University, developed by Erik Schultheis
// All rights reserved.
//
// SPDX-License-Identifier: MIT

#include "common.h"

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

#include "doctest.h"

/*!
 * \test This checks that vector writing works. It in particular verifies the special cases of 0 and 1-element vectors
 */
TEST_CASE("check write dense vector")
{
    std::stringstream target;
    SUBCASE("empty") {
        DenseRealVector v(0);
        CHECK(&io::write_vector_as_text(target, v) == &target);
        CHECK(target.str().empty());
    }

    SUBCASE("one element") {
        DenseRealVector v(1);
        v << 2.5;
        CHECK(&io::write_vector_as_text(target, v) == &target);
        CHECK(target.str() == "2.5");
    }

    SUBCASE("three elements") {
        DenseRealVector v(3);
        v << 2.5, -1.0, 8e12;
        CHECK(&io::write_vector_as_text(target, v) == &target);
        CHECK(target.str() == "2.5 -1 8e+12");
    }
}

/*!
 * \test This checks that reading a dense vector from text works.
 */
TEST_CASE("read dense vector from text") {
    std::stringstream source;
    SUBCASE("empty") {
        DenseRealVector v(0);
        CHECK(&io::read_vector_from_text(source, v) == &source);
    }

    SUBCASE("one element") {
        source.str("2.5");
        DenseRealVector v(1);
        CHECK(&io::read_vector_from_text(source, v) == &source);
        CHECK(v.coeff(0) == 2.5);
    }

    SUBCASE("three elements") {
        DenseRealVector v(3);
        source.str("2.5 -1 8e+12");
        CHECK(&io::read_vector_from_text(source, v) == &source);
        CHECK(v.coeff(0) == 2.5);
        CHECK(v.coeff(1) == -1);
        CHECK(v.coeff(2) == doctest::Approx(8e+12));
    }
}

/*!
 * \test This test checks that XMC feature parsing of valid lines gives the correct result.
 * We check robustness to
 *  - leading spaces
 *  - space vs tab
 *  - scientific notation
 *  - space after separator
 *  - space after last feature
 */
TEST_CASE("parse sparse vector") {
    /// TODO move and rename this test case, it is no longer feature specific.
    std::vector<long> expect_ids;
    std::vector<double> expect_vals;

    auto run_test = [&](std::string source) {
        REQUIRE(expect_ids.size() == expect_vals.size());
        int pos = 0;
        CAPTURE(source);
        try {
            io::parse_sparse_vector_from_text(source.data(), [&](long i, double v) {
                CHECK(expect_ids.at(pos) == i);
                CHECK(expect_vals.at(pos) == v);
                ++pos;
            });
        } catch (std::runtime_error &err) {
            FAIL("parsing failed");
        }
        CHECK(expect_ids.size() == pos);
    };

    SUBCASE("no leading space") {
        expect_ids = {12, 7};
        expect_vals = {2.6, 4.4};
        run_test("12:2.6 7:4.4");
    }

    SUBCASE("simple valid features") {
        expect_ids = {12, 7};
        expect_vals = {2.6, 4.4};
        run_test(" 12:2.6 7:4.4");
    }

    SUBCASE("leading space in front of values") {
        expect_ids = {12, 7};
        expect_vals = {2.6, 4.4};
        run_test(" 12: 2.6 7: 4.4");
    }

    SUBCASE("tab separation") {
        expect_ids = {12, 7};
        expect_vals = {2.6, 4.4};
        run_test("\t12:2.6\t7:4.4");
    }

    SUBCASE("scientific notation") {
        expect_ids = {12, 7};
        expect_vals = {2.6e-5, 4.4e4};
        run_test(" 12:2.6e-5 7:4.4e4");
    }

    SUBCASE("ends with space") {
        expect_ids = {12, 7};
        expect_vals = {2, 4};
        run_test(" 12:2 7:4\t ");
    }
}

/*!
 * \test This test checks that XMC sparse vector parsing of invalid lines results in an error. We check the following
 * formatting errors:
 *  - non-integer index
 *  - not-a-number
 *  - missing value with and without space/with and without colon `:`
 *  - wrong separator
 *  - space in front of colon.
 *  - missing feature index
 */
TEST_CASE("parse sparse vector errors") {
    CHECK_THROWS(io::parse_sparse_vector_from_text(" 5.4:2.0", [&](long i, double v) {}));
    CHECK_THROWS(io::parse_sparse_vector_from_text(" x:2.0", [&](long i, double v) {}));
    CHECK_THROWS(io::parse_sparse_vector_from_text(" 5:2.x", [&](long i, double v) {}));
    CHECK_THROWS(io::parse_sparse_vector_from_text(" 5:", [&](long i, double v) {}));
    CHECK_THROWS(io::parse_sparse_vector_from_text(" 5: ", [&](long i, double v) {}));
    CHECK_THROWS(io::parse_sparse_vector_from_text(" 5", [&](long i, double v) {}));
    CHECK_THROWS(io::parse_sparse_vector_from_text(" 5 ", [&](long i, double v) {}));
    CHECK_THROWS(io::parse_sparse_vector_from_text(" 5-4", [&](long i, double v) {}));
    CHECK_THROWS(io::parse_sparse_vector_from_text(" 5 : 2.0", [&](long i, double v) {}));
    CHECK_THROWS(io::parse_sparse_vector_from_text(":2.0", [&](long i, double v) {}));
}

/*!
 * \test This test case checks that binary dump and load round-trip data
 */
TEST_CASE("binary dump/load") {
    std::stringbuf buffer;
    std::vector<float> data = {4.0, 2.0, 8.0, -2.0};
    io::binary_dump(buffer, &*(data.begin()), &*(data.end()));
    CHECK(buffer.pubseekoff(0, std::ios_base::cur, std::ios_base::out) == sizeof(float) * data.size());
    buffer.pubseekpos(0);
    std::vector<float> load(data.size());
    io::binary_load(buffer, &*(load.begin()), &*(load.end()));
    for(int i = 0; i < data.size(); ++i) {
        CHECK(data[i] == load[i]);
    }
}
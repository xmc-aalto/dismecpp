// Copyright (c) 2022, Aalto University, developed by Erik Schultheis
// All rights reserved.
//
// SPDX-License-Identifier: MIT

#include "utils/macros.h"
// GCC 11 emits `maybe-uninitialized` from inside Eigen with one of the test cases.
DIAGNOSTIC_IGNORE_GCC("-Wmaybe-uninitialized")

#include "doctest.h"
#include "dense.h"
#include "utils/eigen_generic.h"


using namespace dismec;
using namespace dismec::model;

/*! \test We check the that the following result in an exception when getting the label weights:
 *      - supplying a target vector of non-matching size
 *      - accessing an invalid label
 */
TEST_CASE("get dense weights errors")
{
    auto test_mat = std::make_shared<DenseModel::WeightMatrix>(DenseModel::WeightMatrix::Zero(4, 3));
    DenseModel model{test_mat};

    DenseRealVector target(10);
    CHECK_THROWS(model.get_weights_for_label(label_id_t{0}, target));

    // OK, is the size matches we're good to go.
    target = DenseRealVector::Ones(4);
    REQUIRE_NOTHROW(model.get_weights_for_label(label_id_t{0}, target));

    // check error on wrong label
    CHECK_THROWS(model.get_weights_for_label(label_id_t{-1}, target));
    CHECK_THROWS(model.get_weights_for_label(label_id_t{3}, target));
}

/*! \test We check the that the following result in an exception when setting the label weights:
 *      - supplying a target vector of non-matching size
 *      - accessing an invalid label
 */
TEST_CASE("set dense weights errors") {
    auto test_mat = std::make_shared<DenseModel::WeightMatrix>(DenseModel::WeightMatrix::Zero(4, 3));
    DenseModel model{test_mat};

    DenseRealVector source(10);
    CHECK_THROWS(model.set_weights_for_label(label_id_t{0}, Model::WeightVectorIn{source}));

    // OK, is the size matches we're good to go.
    source = DenseRealVector::Ones(4);
    REQUIRE_NOTHROW(model.set_weights_for_label(label_id_t{0}, Model::WeightVectorIn{source}));

    // check error on wrong label
    CHECK_THROWS(model.set_weights_for_label(label_id_t{-1}, Model::WeightVectorIn{source}));
    CHECK_THROWS(model.set_weights_for_label(label_id_t{3}, Model::WeightVectorIn{source}));
}

/*! \test This test verifies that setting and getting weights for labels round-trips.
 */
TEST_CASE("get/set dense weights round-trip") {
    auto test_mat = std::make_shared<DenseModel::WeightMatrix>(DenseModel::WeightMatrix::Zero(4, 3));
    DenseModel model{test_mat};

    DenseRealVector source = DenseRealVector::Ones(4);
    source.coeffRef(2) = 2.0;
    model.set_weights_for_label(label_id_t{1}, DenseModel::WeightVectorIn{source});

    DenseRealVector target(4);
    model.get_weights_for_label(label_id_t{1}, target);

    for(int i = 0; i < 4; ++i) {
        CHECK(source[i] == target[i]);
    }
}

/*! \test This test verifies that `Model::predict_scores()` throws errors if the matrices have wrong shapes
 */
TEST_CASE("predict_scores checks") {
    DenseModel model{4, 3};

    PredictionMatrix t1(3, 7);
    PredictionMatrix t2(3, 6);
    PredictionMatrix t3(4, 6);

    CHECK_THROWS(model.predict_scores(GenericInMatrix::DenseRowMajorRef(Eigen::MatrixXf(4, 6)), t1));        // mismatched rows
    CHECK_THROWS(model.predict_scores(GenericInMatrix::DenseColMajorRef(Eigen::MatrixXf(3, 6)), t2));        // wrong number of features
    CHECK_THROWS(model.predict_scores(GenericInMatrix::DenseColMajorRef(Eigen::MatrixXf(4, 6)), t3));        // wrong number of labels
}

/*! \test This test verifies the partial model interface part of the Model base class. We use DenseModel as the
 * instantiation. This doubles as a test for the `DenseModel` constructors.
 */
TEST_CASE("partial model") {
    DenseModel full(4, 5);
    CHECK_FALSE(full.is_partial_model());
    CHECK(full.labels_begin() == label_id_t{0});
    CHECK(full.labels_end() == label_id_t{5});
    CHECK(full.num_labels() == 5);
    CHECK(full.num_weights() == 5);

    DenseModel partial(4, PartialModelSpec{label_id_t{1}, 3, 5});
    CHECK(partial.is_partial_model());
    CHECK(partial.labels_begin() == label_id_t{1});
    CHECK(partial.labels_end() == label_id_t{4});
    CHECK(partial.num_labels() == 5);
    CHECK(partial.num_weights() == 3);
}

/*! \test This test verifies that the constructor of DenseModel throws an error if the model specifications
 * are impossible.
 */
TEST_CASE("DenseModel ctor consistency") {
    auto build = [](long first_label, long label_count, long total_labels) {
        DenseModel model(4, PartialModelSpec{label_id_t{first_label}, label_count, total_labels});
    };
    // label range exceeding total number of labels
    SUBCASE("invalid range") {
        CHECK_THROWS(build(4, 3, 5));
        CHECK_THROWS(build(1, 5, 5));
    }

    // negative numbers or zero are invalid
    SUBCASE("non positive") {
        // first label can be positive
        CHECK_THROWS(build(-1, 5, 5));
        CHECK_THROWS(build(3, 0, 5));
        CHECK_THROWS(build(3, -1, 5));
        CHECK_THROWS(build(3, 2, 0));
        CHECK_THROWS(build(3, 2, -1));

        CHECK_THROWS(DenseModel(0, 5));
        CHECK_THROWS(DenseModel(-1, 5));
    }

    SUBCASE("data mismatch") {
        auto matrix = std::make_shared<DenseModel::WeightMatrix>(4, 3);
        // claim four labels, but matrix only has 3
        CHECK_THROWS(DenseModel(matrix, PartialModelSpec{label_id_t{0}, 4, 5}));
    }
}

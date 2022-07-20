// Copyright (c) 2021, Aalto University, developed by Erik Schultheis
// All rights reserved.
//
// SPDX-License-Identifier: MIT

#include "reg_sq_hinge_detail.h"
#include "reg_sq_hinge.h"
#include "doctest.h"
#include "utils/test_utils.h"
#include "regularizers_imp.h"

using namespace dismec;
using namespace dismec::l2_reg_sq_hinge_detail;
using dismec::objective::Regularized_SquaredHingeSVC;

TEST_CASE("hessian_times_direction_sum") {
    int num_ftr = 1000;
    int num_rows = 500;

    std::vector<int> indices = {1, 5, 34, 54, 125, 499};

    SparseFeatures features = make_uniform_sparse_matrix(num_rows, num_ftr, 5);

    DenseRealVector direction = DenseRealVector::Random(num_ftr);
    DenseRealVector costs = DenseRealVector::Random(num_ftr);

    // simple reference implementation
    DenseRealVector reference(num_ftr);
    reference.setZero();
    htd_sum_naive(indices, reference, features, costs, direction);

    DenseRealVector output(num_ftr);
    output.setZero();

    htd_sum(indices, output, features, costs, direction);

    if(output != reference) {
        for(int i = 0; i < num_ftr; ++i) {
            DOCTEST_CAPTURE(i);
            REQUIRE(output.coeff(i) == reference.coeff(i));
        }
    }
}

/*
TEST_CASE("line restriction") {
    Eigen::SparseMatrix<real_t> x(3, 5);
    x.insert(0, 3) = 1.0;
    x.insert(1, 0) = 2.0;
    x.insert(2, 1) = 1.0;
    x.insert(2, 2) = 1.0;

    BinaryLabelVector y(3);
    y << -1, 1, -1;


    auto loss = Regularized_SquaredHingeSVC(std::make_shared<SparseFeatures>(x), std::make_unique<objective::SquaredNormRegularizer>());
    loss.get_label_ref() = y;

    DenseRealVector weights = DenseRealVector::Random(5);
    DenseRealVector dir = DenseRealVector::Random(5);

    loss.project_to_line(HashVector{weights}, dir);

    for(real_t a = 0; a < 2.0; a += 0.1) {
        real_t ground_truth = loss.value(HashVector{weights + a*dir});
        real_t fast_approx = loss.lookup_on_line(a);
        CHECK(fast_approx == doctest::Approx(ground_truth));
    }
}


TEST_CASE("L2 regularized squared hinge") {
    Eigen::SparseMatrix<real_t> x(3, 5);
    x.insert(0, 3) = 1.0;
    x.insert(1, 0) = 2.0;
    x.insert(2, 1) = 1.0;
    x.insert(2, 2) = 1.0;

    Eigen::Matrix<std::int8_t, Eigen::Dynamic, 1> y(3);
    y << -1, 1, -1;


    auto loss = Regularized_SquaredHingeSVC(std::make_shared<SparseFeatures>(x), std::make_unique<objective::SquaredNormRegularizer>());
    loss.get_label_ref() = y;

    DenseRealVector weights(5);
    weights << 1.0, 2.0, 0.0, -1.0, 2.0;

    auto do_check = [&](real_t factor){
        // z = (-1, 2, 2)
        // 1 - yz = 0, -1, 3
        CHECK_MESSAGE(loss.value(HashVector{weights}) == doctest::Approx(factor * 9.0 + 5), "wrong value");

        //
        DenseRealVector grad(5);
        loss.gradient(HashVector{weights}, grad);
        // dl/dz = 0, 0, 2*3
        // dl/dx = 6*(0.0, 1.0, 1.0, 0.0, 0.0) + 0.5*weights
        DenseRealVector i(5);
        i << 0.0, 1.0, 1.0, 0.0, 0.0;
        DenseRealVector r = factor * 6 * i + weights;
        CHECK_MESSAGE(grad == r, "wrong gradient");

        // also check numerically
        real_t old_val = loss.value(HashVector{weights});
        DenseRealVector nw = weights + grad * 1e-4;
        real_t new_val = loss.value(HashVector{nw});
        CHECK (new_val - old_val == doctest::Approx(grad.squaredNorm() * 1e-4).epsilon(1e-4));
    };

    // since the positive example is correct with margin,
    // re-weighting positives does not change the outcome,
    // whereas negatives change the result by a constant factor
    SUBCASE("unweighted") {
        do_check(1.0);
    }
    SUBCASE("positive-reweighted") {
        loss.set_positive_cost(2.0);
        do_check(1.0);
    }
    SUBCASE("negative-reweighted") {
        loss.set_negative_cost(2.0);
        do_check(2.0);
    }
}

 */

/// TODO gradient_at_zero test

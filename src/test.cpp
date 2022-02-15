// Copyright (c) 2021, Aalto University, developed by Erik Schultheis
// All rights reserved.
//
// SPDX-License-Identifier: MIT

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"
/*
#include "solver/minimizer.h"
#include "solver/cg.h"
#include "hash_vector.h"
#include "objectives/l2_reg_sq_hinge.h"


TEST_CASE("eurlex at w=0") {
    auto problem = read_liblinear_dataset("test_data/liblin_eurlex_train.txt");
    auto objective = Regularized_SquaredHingeSVC(problem.get_features(), problem.get_labels(0));

    Eigen::VectorXd w = Eigen::VectorXd::Zero(objective.get_num_variables());
    Eigen::VectorXd reference_gradient = Eigen::VectorXd::Zero(objective.get_num_variables());
    Eigen::VectorXd reference_m = Eigen::VectorXd::Zero(objective.get_num_variables());
    {
        std::fstream file_g("test_data/g.txt", std::fstream::in);
        REQUIRE(file_g.is_open());
        for (int i = 0; i < objective.get_num_variables(); ++i) {
            file_g >> reference_gradient.coeffRef(i);
        }
        REQUIRE(file_g.good());

        std::fstream file_m("test_data/m.txt", std::fstream::in);
        REQUIRE(file_m.is_open());
        for (int i = 0; i < objective.get_num_variables(); ++i) {
            file_m >> reference_m.coeffRef(i);
        }
        REQUIRE(file_m.good());

        std::fstream weight_file("test_data/w.txt", std::fstream::in);
        REQUIRE(weight_file.is_open());
        for (int i = 0; i < objective.get_num_variables(); ++i) {
            weight_file >> w.coeffRef(i);
        }
        REQUIRE(weight_file.good());
    }

    SUBCASE("gradient") {
        auto g = objective.gradient(HashVector{w});
        for (int i = 0; i < objective.get_num_variables(); ++i) {
            CHECK(reference_gradient.coeff(i) == doctest::Approx(g[i]).epsilon(1e-10));
        }
    }

    SUBCASE("hessian times direction") {
        Eigen::VectorXd d(objective.get_num_variables());
        std::fstream file_d("test_data/d.txt", std::fstream::in);
        REQUIRE(file_d.is_open());
        for (int i = 0; i < objective.get_num_variables(); ++i) {
            file_d >> d.coeffRef(i);
        }
        REQUIRE(file_d.good());

        auto Hd = objective.hessian_times_direction(HashVector{w}, d);

        std::cout << "d " << d[0] << "\n";

        std::fstream reference("test_data/Hd.txt", std::fstream::in);
        for (int i = 0; i < objective.get_num_variables(); ++i) {
            double ground_truth;
            reference >> ground_truth;
            CHECK(ground_truth == doctest::Approx(Hd[i]).epsilon(1e-14));
        }
    }

    SUBCASE("preconditioning") {
        CGMinimizer cg(objective.get_num_variables());
        Eigen::VectorXd M = objective.get_diag_preconditioner(HashVector{w});
        for (int i = 0; i < objective.get_num_variables(); ++i) {
            CHECK(reference_m.coeff(i) == doctest::Approx(M[i]).epsilon(1e-14));
        }
    }

    SUBCASE("cg") {
        CGMinimizer cg(objective.get_num_variables());
        // regularize the preconditioner: M = aI + (1-a)M
        Eigen::VectorXd M = (1 - 0.01) + (reference_m * 0.01).array();

        cg.minimize([&](const Eigen::VectorXd& d, Eigen::Ref<Eigen::VectorXd> o) {
            o = objective.hessian_times_direction(HashVector{w}, d);
        }, reference_gradient, M);

        std::fstream reference("test_data/s.txt", std::fstream::in);
        for (int i = 0; i < objective.get_num_variables(); ++i) {
            double ground_truth;
            reference >> ground_truth;
            REQUIRE(ground_truth == doctest::Approx(cg.get_solution()[i]));
        }
    }
}

TEST_CASE("heart_scale") {
    auto problem = read_liblinear_dataset("test_data/heart_scale");

    auto objective = Regularized_SquaredHingeSVC(problem.get_features(), problem.get_labels(0));
    auto minimizer = NewtonWithLineSearch();

    Eigen::VectorXd w = Eigen::VectorXd::Zero(problem.num_features());
    auto res = minimizer.minimize(objective, w);
    REQUIRE(res.Outcome == MinimizerStatus::SUCCESS);


    std::array<double, 13> liblinear_results = {
            0.10478334183618995,
            0.2310677614517834,
            0.42808388718499868,
            0.26794147077324199,
            -0.01253218890573427,
            -0.16433603637371605,
            0.12597353419494489,
            -0.26426183971657191,
            0.12586162713774524,
            0.070475901031940236,
            0.16194342067257222,
            0.44104006346498714,
            0.25983806499730233,
    };

    for(int i = 0; i < 13; ++i) {
        CHECK(w[i] == doctest::Approx(liblinear_results[i]));
    }
}
*/
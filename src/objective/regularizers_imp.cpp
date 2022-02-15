// Copyright (c) 2021, Aalto University, developed by Erik Schultheis
// All rights reserved.
//
// SPDX-License-Identifier: MIT

#include "regularizers_imp.h"
#include "regularizers.h"
#include "hash_vector.h"

using objective::SquaredNormRegularizer;
using objective::HuberRegularizer;
using objective::ElasticNetRegularizer;

SquaredNormRegularizer::SquaredNormRegularizer(real_t scale, bool ignore_bias) :
        PointWiseRegularizer(scale, ignore_bias) {

}

void SquaredNormRegularizer::project_to_line_unchecked(const HashVector& location, const DenseRealVector& direction) {

    m_LsCache_w02 = location->squaredNorm();
    m_LsCache_d2 = direction.squaredNorm();
    m_LsCache_dTw = location->dot(direction);
    assert(std::isfinite(m_LsCache_w02));
    assert(std::isfinite(m_LsCache_d2));
    assert(std::isfinite(m_LsCache_dTw));

    if(dont_regularize_bias()) {
        real_t ll = location->coeff(location->size() - 1);
        real_t ld = direction.coeff(direction.size() - 1);
        m_LsCache_w02 -= ll * ll;
        m_LsCache_d2 -= ld * ld;
        m_LsCache_dTw -= ll * ld;
    }
}

real_t SquaredNormRegularizer::lookup_on_line(real_t a) {
    return real_t{0.5} * scale() * (m_LsCache_w02 + 2*a*m_LsCache_dTw + a*a*m_LsCache_d2);
}

real_t objective::SquaredNormRegularizer::value_unchecked(const HashVector& location) {
    return real_t{0.5} * PointWiseRegularizer::value_unchecked(location);
}

real_t objective::SquaredNormRegularizer::point_wise_value(real_t x) {
    return x * x;
}

real_t objective::SquaredNormRegularizer::point_wise_grad(real_t x) {
    return x;
}

real_t objective::SquaredNormRegularizer::point_wise_quad(real_t x) {
    return 1;
}

HuberRegularizer::HuberRegularizer(real_t epsilon, real_t scale, bool ignore_bias) :
        PointWiseRegularizer(scale, ignore_bias), m_Epsilon(epsilon) {
    if(m_Epsilon <= 0) {
        THROW_EXCEPTION(std::invalid_argument, "Epsilon has to be positive. Got {}", m_Epsilon);
    }
}

real_t objective::HuberRegularizer::point_wise_value(real_t x) const {
    if(x > m_Epsilon) return x - m_Epsilon/2;
    if(x < -m_Epsilon) return -x - m_Epsilon/2;
    return real_t{0.5} * x*x / m_Epsilon;
}

real_t objective::HuberRegularizer::point_wise_grad(real_t x) const {
    if(x > m_Epsilon) {
        return 1.0;
    } else if(x < -m_Epsilon) {
        return -1.0;
    } else {
        return x / m_Epsilon;
    }
}

real_t objective::HuberRegularizer::point_wise_quad(real_t x) const {
    if(x > m_Epsilon) return real_t{1.0} / x;
    if(x < -m_Epsilon) return -real_t{1.0} / x;
    return real_t{0.5} / m_Epsilon;
}

ElasticNetRegularizer::ElasticNetRegularizer(real_t epsilon, real_t scale, real_t interp, bool ignore_bias)
    : PointWiseRegularizer(scale, ignore_bias), m_Epsilon(epsilon), m_L1_Factor(1 - interp), m_L2_Factor(interp)
{
    if(m_Epsilon <= 0) {
        THROW_EXCEPTION(std::invalid_argument, "Epsilon has to be positive. Got {}", m_Epsilon);
    }

    if(interp < 0 || interp > 1) {
        THROW_EXCEPTION(std::invalid_argument, "Interpolation needs to be in [0, 1]. Got {}", interp);
    }
}

real_t objective::ElasticNetRegularizer::point_wise_value(real_t x) const {
    real_t x2 = x*x;
    if(x > m_Epsilon) return m_L1_Factor*(x - m_Epsilon/2) + real_t{0.5} * m_L2_Factor * x2;
    if(x < -m_Epsilon) return m_L1_Factor*(-x - m_Epsilon/2) + real_t{0.5} * m_L2_Factor * x2;
    return real_t{0.5} * (m_L1_Factor / m_Epsilon + m_L2_Factor) * x2;
}

real_t objective::ElasticNetRegularizer::point_wise_grad(real_t x) const {
    if(x > m_Epsilon) {
        return m_L1_Factor + m_L2_Factor * x;
    } else if(x < -m_Epsilon) {
        return -m_L1_Factor + m_L2_Factor * x;
    } else {
        return m_L1_Factor * x / m_Epsilon + m_L2_Factor * x;
    }
}

real_t objective::ElasticNetRegularizer::point_wise_quad(real_t x) const {
    if(x > m_Epsilon) return m_L1_Factor / x + m_L2_Factor;
    if(x < -m_Epsilon) return -m_L1_Factor / x + m_L2_Factor;
    return real_t{0.5} / m_Epsilon * m_L1_Factor + m_L2_Factor;
}

using objective::Objective;
// The factory functions
std::unique_ptr<Objective> objective::make_regularizer(const SquaredNormConfig& config) {
    return std::make_unique<SquaredNormRegularizer>(config.Strength, config.IgnoreBias);
}

std::unique_ptr<Objective> objective::make_regularizer(const HuberConfig& config) {
    return std::make_unique<HuberRegularizer>(config.Epsilon, config.Strength, config.IgnoreBias);
}

std::unique_ptr<Objective> objective::make_regularizer(const ElasticConfig& config) {
    return std::make_unique<ElasticNetRegularizer>(config.Epsilon, config.Strength, config.Interpolation, config.IgnoreBias);
}


#include "doctest.h"

namespace {
    DenseRealVector make_vec(std::initializer_list<real_t> values) {
        DenseRealVector vec(values.size());
        auto it = begin(values);
        for(int i = 0; i < values.size(); ++i) {
            vec.coeffRef(i) = *it;
            ++it;
        }
        return vec;
    }
}

TEST_CASE("l2-reg") {
    SquaredNormRegularizer reg(1.0);
    DenseRealVector loc = make_vec({1.0, 2.0, -3.0, 0.0});
    HashVector hl{loc};

    // check value
    CHECK(reg.value(hl) == 0.5*(1+4+9));

    // check gradient: should be equal to location
    DenseRealVector target(loc.size());
    reg.gradient(hl, target);
    for(int i = 0; i < target.size(); ++i) {
        CHECK(loc.coeff(i) == target.coeff(i));
    }

    reg.gradient_at_zero(target);
    CHECK(target.squaredNorm() == 0);

    // check hessian: should be equal to probe direction
    DenseRealVector probe = make_vec({1.0, 2.0, 1.0, -1.0});
    reg.hessian_times_direction(hl, probe, target);
    for(int i = 0; i < target.size(); ++i) {
        CHECK(probe.coeff(i) == target.coeff(i));
    }

    // check preconditioner
    reg.diag_preconditioner(hl, target);
    for(int i = 0; i < target.size(); ++i) {
        CHECK(target.coeff(i) == 1.0);
    }
}

void verify_line_search(objective::Objective& reg) {
    DenseRealVector loc = make_vec({1.0, 2.0, -3.0, 0.0});
    DenseRealVector dir = make_vec({3.0, -1.0, 2.0, 1.0});

    reg.project_to_line(HashVector{loc}, dir);

    for(real_t t : {-1.2, 0.1, 0.5, 0.8, 2.5}) {
        real_t predict = reg.lookup_on_line(t);
        real_t actual = reg.value(HashVector{loc + t*dir});
        CHECK(predict == doctest::Approx(actual));
    }
}

void verify_bias(objective::Objective& full, objective::Objective& no_bias) {
    DenseRealVector loc = make_vec({1.0, 0.05, -3.0, 0.0});
    DenseRealVector dir = make_vec({3.0, -1.0, 2.0, 1.0});
    HashVector hl{loc};

    // short versions
    HashVector short_loc{loc.topRows(3)};
    DenseRealVector short_dir{dir.topRows(3)};

    // short on full objective
    real_t reference = full.value(short_loc);
    CHECK(no_bias.value(hl) == doctest::Approx(reference));

    DenseRealVector target = DenseRealVector::Random(4);
    DenseRealVector short_target(3);

    full.gradient(short_loc, short_target);
    no_bias.gradient(hl, target);
    for(int i = 0; i < 3; ++i) {
        CHECK(target.coeff(i) == short_target.coeff(i));
    }
    CHECK(target.coeff(3) == 0);

    full.hessian_times_direction(short_loc, short_dir, short_target);
    target = DenseRealVector::Random(4);
    no_bias.hessian_times_direction(hl, dir, target);
    for(int i = 0; i < 3; ++i) {
        CHECK(target.coeff(i) == short_target.coeff(i));
    }
    CHECK(target.coeff(3) == 0);
}

TEST_CASE("l2 line-search") {
    bool ignore_bias = false;

    SUBCASE("ignore bias") {
        ignore_bias = true;
    }

    SUBCASE("full weights") {
        ignore_bias = false;
    }

    SquaredNormRegularizer reg(1.0, ignore_bias);
    verify_line_search(reg);
}

TEST_CASE("l2 bias") {
    SquaredNormRegularizer full(1.0);
    SquaredNormRegularizer bias(1.0, true);
    verify_bias(full, bias);
}

TEST_CASE("huber-reg") {
    HuberRegularizer absreg(1);
    DenseRealVector loc(4);
    loc << 1, 5.0, -3.0, 0.0;
    HashVector hl{loc};
    CHECK(absreg.value(hl) == 9.0 - 1.5);

    DenseRealVector grad(4);
    absreg.gradient(hl, grad);

    CHECK(grad.coeff(0) == 1.0);
    CHECK(grad.coeff(1) == 1.0);
    CHECK(grad.coeff(2) == -1.0);
    CHECK(grad.coeff(3) == 0.0);

    absreg.gradient_at_zero(grad);
    CHECK(grad.squaredNorm() == 0);
}

TEST_CASE("huber line-search") {
    bool ignore_bias = false;

    SUBCASE("ignore bias") {
        ignore_bias = true;
    }

    SUBCASE("full weights") {
        ignore_bias = false;
    }

    HuberRegularizer reg(1.0, 1.0, ignore_bias);
    verify_line_search(reg);
}


TEST_CASE("huber bias") {
    HuberRegularizer full(1.0, 1.0);
    HuberRegularizer bias(1.0, 1.0, true);
    verify_bias(full, bias);
}
#include "spdlog/spdlog.h"
TEST_CASE("elastic-net") {
    ElasticNetRegularizer reg(1.0, 1.0, 0.4, false);
    HuberRegularizer l1_part(1.0, 0.6, false);
    SquaredNormRegularizer l2_part(0.4, false);

    DenseRealVector loc(4);
    loc << 1, 0.5, -3.0, 0.0;
    HashVector hl{loc};

    auto check_vector_valued = [&](auto&& f) {
        DenseRealVector elastic(4);
        DenseRealVector l1(4);
        DenseRealVector l2(4);
        f(reg, elastic);
        f(l1_part, l1);
        f(l2_part, l2);
        CHECK(elastic.coeff(0) == doctest::Approx(l1.coeff(0) + l2.coeff(0)));
        CHECK(elastic.coeff(1) == doctest::Approx(l1.coeff(1) + l2.coeff(1)));
        CHECK(elastic.coeff(2) == doctest::Approx(l1.coeff(2) + l2.coeff(2)));
        CHECK(elastic.coeff(3) == doctest::Approx(l1.coeff(3) + l2.coeff(3)));
    };

    SUBCASE("value") {
        CHECK(reg.value(hl) == l1_part.value(hl) + l2_part.value(hl));
    }

    SUBCASE("gradient") {
        check_vector_valued([&](auto&& ref, auto&& vec){
            ref.gradient(hl, vec);
        });
    }

    SUBCASE("gradient_at_zero") {
        check_vector_valued([&](auto&& ref, auto&& vec){
            ref.gradient_at_zero(vec);
        });
    }

    SUBCASE("hessian_times_direction") {
        DenseRealVector dir = DenseRealVector::Random(4);
        check_vector_valued([&](auto&& ref, auto&& vec){
            ref.hessian_times_direction(hl, dir, vec);
        });
    }

    DenseRealVector grad(4);
    reg.gradient(hl, grad);


}

TEST_CASE("elastic line-search") {
    bool ignore_bias = false;

    SUBCASE("ignore bias") {
        ignore_bias = true;
    }

    SUBCASE("full weights") {
        ignore_bias = false;
    }

    ElasticNetRegularizer reg(1.0, 1.0, 0.5, ignore_bias);
    verify_line_search(reg);
}


TEST_CASE("elastic bias") {
    ElasticNetRegularizer full(1.0, 1.0, 0.7);
    ElasticNetRegularizer bias(1.0, 1.0, 0.7,true);
    verify_bias(full, bias);
}

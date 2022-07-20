// Copyright (c) 2021, Aalto University, developed by Erik Schultheis
// All rights reserved.
//
// SPDX-License-Identifier: MIT

#include "objective.h"
#include "utils/hash_vector.h"
#include "utils/throw_error.h"
#include "stats/timer.h"

using namespace dismec;
using namespace dismec::objective;

namespace {
    stats::stat_id_t STAT_PERF_VALUE{0};
    stats::stat_id_t STAT_PERF_PRECONDITIONER{1};
    stats::stat_id_t STAT_PERF_GRAD_AT_ZERO{2};
    stats::stat_id_t STAT_PERF_GRADIENT{3};
    stats::stat_id_t STAT_PERF_HESSIAN{4};
    stats::stat_id_t STAT_PERF_GRAD_AND_PRE{5};
    stats::stat_id_t STAT_PERF_PROJ_TO_LINE{6};
}

Objective::Objective() {
    declare_stat(STAT_PERF_VALUE, {"perf_value", "µs"});
    declare_stat(STAT_PERF_PRECONDITIONER, {"perf_preconditioner", "µs"});
    declare_stat(STAT_PERF_GRAD_AT_ZERO, {"perf_grad_at_zero", "µs"});
    declare_stat(STAT_PERF_GRADIENT, {"perf_gradient", "µs"});
    declare_stat(STAT_PERF_HESSIAN, {"perf_hessian", "µs"});
    declare_stat(STAT_PERF_GRAD_AND_PRE, {"perf_grad_and_pre", "µs"});
    declare_stat(STAT_PERF_PROJ_TO_LINE, {"perf_proj_to_line", "µs"});
}


real_t Objective::value(const HashVector& location) {
    auto timer = make_timer(STAT_PERF_VALUE);
    if(num_variables() > 0) {
        ALWAYS_ASSERT_EQUAL(location->size(), num_variables(), "location size {} differs from num_variables {}");
    }
    return value_unchecked(location);
}

void Objective::diag_preconditioner(const HashVector& location, Eigen::Ref<DenseRealVector> target) {
    auto timer = make_timer(STAT_PERF_PRECONDITIONER);
    if(num_variables() > 0) {
        ALWAYS_ASSERT_EQUAL(location->size(), num_variables(), "location size {} differs from num_variables {}");
        ALWAYS_ASSERT_EQUAL(target.size(), num_variables(), "target size {} differs from num_variables {}");
    } else {
        ALWAYS_ASSERT_EQUAL(target.size(), location->size(), "target size {} differs from location size {}");
    }
    diag_preconditioner_unchecked(location, target);
}

void Objective::diag_preconditioner_unchecked(const HashVector& location, Eigen::Ref<DenseRealVector> target) {
    target.setOnes();
}

void Objective::gradient_and_pre_conditioner(const HashVector& location,
                                             Eigen::Ref<DenseRealVector> gradient,
                                             Eigen::Ref<DenseRealVector> pre) {
    auto timer = make_timer(STAT_PERF_GRAD_AND_PRE);
    if(num_variables() > 0) {
        ALWAYS_ASSERT_EQUAL(location->size(), num_variables(), "location size {} differs from num_variables {}");
        ALWAYS_ASSERT_EQUAL(gradient.size(), num_variables(), "gradient size {} differs from num_variables {}");
        ALWAYS_ASSERT_EQUAL(pre.size(), num_variables(), "pre size {} differs from num_variables {}");
    } else {
        ALWAYS_ASSERT_EQUAL(gradient.size(), location->size(), "gradient size {} differs from location size {}");
        ALWAYS_ASSERT_EQUAL(pre.size(), location->size(), "pre size {} differs from location size {}");
    }

    gradient_and_pre_conditioner_unchecked(location, gradient, pre);
}

void Objective::gradient_and_pre_conditioner_unchecked(
        const HashVector& location,
        Eigen::Ref<DenseRealVector> gradient,
        Eigen::Ref<DenseRealVector> pre) {
    gradient_unchecked(location, gradient);
    diag_preconditioner_unchecked(location, pre);
}

void Objective::gradient_at_zero(Eigen::Ref<DenseRealVector> target) {
    auto timer = make_timer(STAT_PERF_GRAD_AT_ZERO);
    if(num_variables() > 0) {
        ALWAYS_ASSERT_EQUAL(target.size(), num_variables(), "target size {} differs from num_variables {}");
    }
    gradient_at_zero_unchecked(target);
}

void Objective::gradient_at_zero_unchecked(Eigen::Ref<DenseRealVector> target) {
    // we can call gradient_unchecked directly, because the first argument here is
    // guaranteed to match the second, which has already been validated.
    gradient_unchecked(HashVector(DenseRealVector::Zero(target.size())), target);
}

void Objective::gradient(const HashVector& location, Eigen::Ref<DenseRealVector> target) {
    auto timer = make_timer(STAT_PERF_GRADIENT);
    if(num_variables() > 0) {
        ALWAYS_ASSERT_EQUAL(location->size(), num_variables(), "location size {} differs from num_variables {}");
        ALWAYS_ASSERT_EQUAL(target.size(), num_variables(), "target size {} differs from num_variables {}");
    } else {
        ALWAYS_ASSERT_EQUAL(target.size(), location->size(), "target size {} differs from location size {}");
    }
    gradient_unchecked(location, target);
}

void Objective::hessian_times_direction(
        const HashVector& location,
        const DenseRealVector& direction,
        Eigen::Ref<DenseRealVector> target) {
    auto timer = make_timer(STAT_PERF_HESSIAN);
    if(num_variables() > 0) {
        ALWAYS_ASSERT_EQUAL(location->size(), num_variables(), "location size {} differs from num_variables {}");
        ALWAYS_ASSERT_EQUAL(target.size(), num_variables(), "target size {} differs from num_variables {}");
        ALWAYS_ASSERT_EQUAL(direction.size(), num_variables(), "direction size {} differs from num_variables {}");
    } else {
        ALWAYS_ASSERT_EQUAL(target.size(), location->size(), "target size {} differs from location size {}");
        ALWAYS_ASSERT_EQUAL(direction.size(), location->size(), "direction size {} differs from location size {}");
    }

    hessian_times_direction_unchecked(location, direction, target);
}

void Objective::project_to_line(const HashVector& location, const DenseRealVector& direction) {
    auto timer = make_timer(STAT_PERF_PROJ_TO_LINE);
    if(num_variables() > 0) {
        ALWAYS_ASSERT_EQUAL(location->size(), num_variables(), "location size {} differs from num_variables {}");
        ALWAYS_ASSERT_EQUAL(direction.size(), num_variables(), "direction size {} differs from num_variables {}");
    } else {
        ALWAYS_ASSERT_EQUAL(direction.size(), location->size(), "direction size {} differs from location size {}");
    }
    project_to_line_unchecked(location, direction);
}

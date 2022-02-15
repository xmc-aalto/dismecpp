// Copyright (c) 2021, Aalto University, developed by Erik Schultheis
// All rights reserved.
//
// SPDX-License-Identifier: MIT

#include "objective.h"
#include "hash_vector.h"
#include "utils/throw_error.h"
#include "stats/timer.h"

using namespace objective;

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
        if (location->size() != num_variables()) {
            THROW_EXCEPTION(std::invalid_argument, "location size {} differs from num_variables {}", location->size(),
                            num_variables());
        }
    }
    return value_unchecked(location);
}

void Objective::diag_preconditioner(const HashVector& location, Eigen::Ref<DenseRealVector> target) {
    auto timer = make_timer(STAT_PERF_PRECONDITIONER);
    if(num_variables() > 0) {
        if(location->size() != num_variables()) {
            THROW_EXCEPTION(std::invalid_argument, "location size {} differs from num_variables {}", location->size(), num_variables());
        }
        if(target.size() != num_variables()) {
            THROW_EXCEPTION(std::invalid_argument, "target size {} differs from num_variables {}", target.size(), num_variables());
        }
    } else {
        if(target.size() != location->size()) {
            THROW_EXCEPTION(std::invalid_argument, "target size {} differs from location size {}", target.size(), location->size());
        }
    }
    target.setOnes();
}

void Objective::diag_preconditioner_unchecked(const HashVector& location, Eigen::Ref<DenseRealVector> target) {
    target.setOnes();
}

void Objective::gradient_and_pre_conditioner(const HashVector& location,
                                             Eigen::Ref<DenseRealVector> gradient,
                                             Eigen::Ref<DenseRealVector> pre) {
    auto timer = make_timer(STAT_PERF_GRAD_AND_PRE);
    if(num_variables() > 0) {
        if(location->size() != num_variables()) {
            THROW_EXCEPTION(std::invalid_argument, "location size {} differs from num_variables {}", location->size(), num_variables());
        }
        if(gradient.size() != num_variables()) {
            THROW_EXCEPTION(std::invalid_argument, "gradient size {} differs from num_variables {}", gradient.size(), num_variables());
        }
        if(pre.size() != num_variables()) {
            THROW_EXCEPTION(std::invalid_argument, "pre size {} differs from num_variables {}", pre.size(), num_variables());
        }
    } else {
        if(gradient.size() != location->size()) {
            THROW_EXCEPTION(std::invalid_argument, "gradient size {} differs from location size {}", gradient.size(), location->size());
        }
        if(pre.size() != location->size()) {
            THROW_EXCEPTION(std::invalid_argument, "pre size {} differs from location size {}", pre.size(), location->size());
        }
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
        if (target.size() != num_variables()) {
            THROW_EXCEPTION(std::invalid_argument, "target size {} differs from num_variables {}", target.size(),
                            num_variables());
        }
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
        if(location->size() != num_variables()) {
            THROW_EXCEPTION(std::invalid_argument, "location size {} differs from num_variables {}", location->size(), num_variables());
        }
        if(target.size() != num_variables()) {
            THROW_EXCEPTION(std::invalid_argument, "target size {} differs from num_variables {}", target.size(), num_variables());
        }
    } else {
        if(target.size() != location->size()) {
            THROW_EXCEPTION(std::invalid_argument, "target size {} differs from location size {}", target.size(), location->size());
        }
    }
    gradient_unchecked(location, target);
}

void Objective::hessian_times_direction(
        const HashVector& location,
        const DenseRealVector& direction,
        Eigen::Ref<DenseRealVector> target) {
    auto timer = make_timer(STAT_PERF_HESSIAN);
    if(num_variables() > 0) {
        if(location->size() != num_variables()) {
            THROW_EXCEPTION(std::invalid_argument, "location size {} differs from num_variables {}",
                            location->size(), num_variables());
        }
        if(target.size() != num_variables()) {
            THROW_EXCEPTION(std::invalid_argument, "target size {} differs from num_variables {}",
                            target.size(), num_variables());
        }
        if(direction.size() != num_variables()) {
            THROW_EXCEPTION(std::invalid_argument, "direction size {} differs from num_variables {}",
                            direction.size(), num_variables());
        }
    } else {
        if(target.size() != location->size()) {
            THROW_EXCEPTION(std::invalid_argument, "target size {} differs from location size {}",
                            target.size(), location->size());
        }
        if(direction.size() != location->size()) {
            THROW_EXCEPTION(std::invalid_argument, "direction size {} differs from location size {}",
                            direction.size(), location->size());
        }
    }

    hessian_times_direction_unchecked(location, direction, target);
}

void Objective::project_to_line(const HashVector& location, const DenseRealVector& direction) {
    auto timer = make_timer(STAT_PERF_PROJ_TO_LINE);
    if(num_variables() > 0) {
        if (location->size() != num_variables()) {
            THROW_EXCEPTION(std::invalid_argument, "location size {} differs from num_variables {}",
                            location->size(), num_variables());
        }
        if(direction.size() != num_variables()) {
            THROW_EXCEPTION(std::invalid_argument, "direction size {} differs from num_variables {}",
                            direction.size(), num_variables());
        }
    }
    project_to_line_unchecked(location, direction);
}

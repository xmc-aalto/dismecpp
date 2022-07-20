// Copyright (c) 2021, Aalto University, developed by Erik Schultheis
// All rights reserved.
//
// SPDX-License-Identifier: MIT

#ifndef DISMEC_SRC_OBJECTIVE_MARGIN_LOSSES_H
#define DISMEC_SRC_OBJECTIVE_MARGIN_LOSSES_H

namespace dismec::objective {
    struct SquaredHingePhi {
        [[nodiscard]] real_t value(real_t margin) const {
            real_t value = std::max(real_t{0}, real_t{1.0} - margin);
            return value * value;
        }

        [[nodiscard]] real_t grad(real_t margin) const {
            real_t value = std::max(real_t{0}, real_t{1.0} - margin);
            return -real_t{2} * value;
        }

        [[nodiscard]] real_t quad(real_t margin) const {
            real_t value = real_t{1.0} - margin;
            return value > 0 ? real_t{2} : real_t{0};
        }
    };

    struct HuberPhi {
        [[nodiscard]] real_t value(real_t margin) const {
            real_t value = std::max(real_t{0}, real_t{1.0} - margin);
            if(value > Epsilon) return value - Epsilon/2;
            return real_t{0.5} * value*value / Epsilon;
        }

        [[nodiscard]] real_t grad(real_t margin) const {
            real_t value = std::max(real_t{0}, real_t{1} - margin);
            if(value > Epsilon) {
                return -real_t{1};
            } else if(value == real_t{0}) {
                return real_t{0};
            } else {
                return -value / Epsilon;
            }
        }

        [[nodiscard]] real_t quad(real_t margin) const {
            real_t value = std::max(real_t{0}, real_t{1.0} - margin);
            if(value > Epsilon) return real_t{1.0} / value;
            if(value == 0) return real_t{0};
            return real_t{1} / Epsilon;
        }

        real_t Epsilon = 1;
    };

    struct LogisticPhi {
        [[nodiscard]] real_t value(real_t margin) const {
            real_t exp_part = std::exp(-margin);
            if(std::isfinite(exp_part)) {
                return std::log1p(exp_part);
            } else {
                return -margin;
            }
        }

        [[nodiscard]] real_t grad(real_t margin) const {
            real_t exp_part = std::exp(margin);
            if(std::isfinite(exp_part)) {
                return -real_t{1} / (real_t{1} + exp_part);
            } else {
                return 0;
            }
        }

        [[nodiscard]] real_t quad(real_t margin) const {
            real_t exp_part = std::exp(margin);
            if(std::isfinite(exp_part)) {
                return exp_part / std::pow(1 + exp_part, real_t{2});
            } else {
                return 0;
            }
        }
    };
}

#endif //DISMEC_SRC_OBJECTIVE_MARGIN_LOSSES_H

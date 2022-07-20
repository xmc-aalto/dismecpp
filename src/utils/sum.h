// Copyright (c) 2021, Aalto University, developed by Erik Schultheis
// All rights reserved.
//
// SPDX-License-Identifier: MIT

#ifndef DISMEC_SRC_UTILS_SUM_H
#define DISMEC_SRC_UTILS_SUM_H

namespace dismec {
    // https://en.wikipedia.org/wiki/Kahan_summation_algorithm
    /*!
     * \brief Implements a numerically stable sum algorithm
     * \tparam Float Floating point type to use for the accumulator.
     * \details Implements Kahan Summation to sum up a large stream of floating point numbers while compensating
     * for numerical instability. Typically, `Float` should be `double`.
     */
    template<class Float>
    class KahanAccumulator {
    public:
        KahanAccumulator() = default;

        Float value() const { return Sum; }

        KahanAccumulator& operator+=(Float value) {
            if (value > Sum) {
                accumulate(value, Correction, Sum);
                Sum = value;
            } else {
                accumulate(Sum, Correction, value);
            }
            return *this;
        }

    private:
        static void accumulate(Float& accumulator, Float& correction, Float addition) {
            Float y = addition - correction;
            volatile Float t = accumulator + y;
            volatile Float z = t - accumulator;
            correction = z - y;
            accumulator = t;
        }

        Float Sum = Float{0};
        Float Correction = Float{0};
    };
}
#endif //DISMEC_SRC_UTILS_SUM_H

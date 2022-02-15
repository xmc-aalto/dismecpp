// Copyright (c) 2021, Aalto University, developed by Erik Schultheis
// All rights reserved.
//
// SPDX-License-Identifier: MIT

#include <utility>

#include "training/postproc.h"
#include "solver/minimizer.h"
#include "hash_vector.h"
#include "training/postproc/generic.h"
#include "stats/collection.h"
#include "stats/timer.h"

namespace {
    stats::stat_id_t STAT_CUTOFF{0};
    stats::stat_id_t STAT_NNZ{1};
    stats::stat_id_t STAT_BINARY_SEARCH_STEPS{2};
    stats::stat_id_t STAT_INITIAL_STEPS{3};
    stats::stat_id_t STAT_DURATION{4};
};


namespace postproc {
    class Sparsify : public PostProcessor {
    public:
        Sparsify(std::shared_ptr<objective::Objective> objective, real_t tolerance) :
            m_Objective(std::move(objective)),
            m_Tolerance(tolerance),
            m_WorkingVector(DenseRealVector(m_Objective->num_variables())) {

            declare_stat(STAT_CUTOFF, {"cutoff", {}});
            declare_stat(STAT_NNZ, {"nnz", "%"});
            declare_stat(STAT_BINARY_SEARCH_STEPS, {"binary_search_steps", {}});
            declare_stat(STAT_INITIAL_STEPS, {"initial_steps", {}});
            declare_stat(STAT_DURATION, {"duration", "µs"});
        }
    private:
        void process(label_id_t label_id, DenseRealVector& weight_vector, solvers::MinimizationResult& result) override;

        std::shared_ptr<objective::Objective> m_Objective;
        real_t m_Tolerance;
        HashVector m_WorkingVector;

        static int make_sparse(DenseRealVector& target, const DenseRealVector& source, real_t cutoff) {
            int nnz = 0;
            for(int i = 0; i < target.size(); ++i) {
                auto w_i = source.coeff(i);
                bool is_small = abs(w_i) < cutoff;
                target.coeffRef(i) = is_small ? 0 : w_i;
                if(!is_small) ++nnz;
            }
            return nnz;
        }

        struct BoundData {
            real_t Cutoff;
            long NNZ;
            real_t Loss;
        };

        struct UpperBoundResult {
            BoundData LowerBound;
            BoundData UpperBound;
        };



        UpperBoundResult find_initial_bounds(DenseRealVector& weight_vector, real_t tolerance, real_t initial_lower);

        real_t m_NumValues = 1;
        real_t m_SumLogVal = std::log(0.02);
        real_t m_SumSqrLog = std::log(0.02) * std::log(0.02);
    };



    void Sparsify::process(label_id_t label_id, DenseRealVector& weight_vector, solvers::MinimizationResult& result) {
        auto timer = make_timer(STAT_DURATION);
        m_WorkingVector = weight_vector;
        real_t tolerance = (1 + m_Tolerance) * result.FinalValue + real_t{1e-5};

        auto [lower, upper] = find_initial_bounds(weight_vector, tolerance, result.FinalValue);

        // now we can do a binary search
        int count = 0;
        while( (lower.NNZ - upper.NNZ) > upper.NNZ / 10 + 1 ) {
            real_t middle = (upper.Cutoff + lower.Cutoff) / 2;
            int nnz = make_sparse(m_WorkingVector.modify(), weight_vector, middle);
            auto new_score = m_Objective->value(m_WorkingVector);
            if(new_score > tolerance) {
                upper.Cutoff = middle;
                upper.NNZ = nnz;
                upper.Loss = new_score;
            } else {
                lower.Cutoff = middle;
                lower.NNZ = nnz;
                lower.Loss = new_score;
            }
            ++count;
        }
        record(STAT_BINARY_SEARCH_STEPS, count);

        // finally, apply the culling to the actual weight vector
        int nnz = make_sparse(weight_vector, weight_vector, lower.Cutoff);

        m_NumValues += 1;
        real_t log_cutoff = std::log(lower.Cutoff);
        m_SumLogVal += log_cutoff;
        m_SumSqrLog += log_cutoff*log_cutoff;

        record(STAT_CUTOFF, lower.Cutoff);
        record(STAT_NNZ, float(100 * nnz) / weight_vector.size());
    }

    Sparsify::UpperBoundResult Sparsify::find_initial_bounds(DenseRealVector& weight_vector, real_t tolerance, real_t initial_lower)
    {
        real_t mean_log = m_SumLogVal / m_NumValues;
        real_t std_log = std::sqrt(m_SumSqrLog / m_NumValues - mean_log*mean_log + real_t{1e-5});

        int step_count = 0;

        auto check_bound = [&](real_t log_cutoff) {
            real_t cutoff = std::exp(log_cutoff);
            int nnz = make_sparse(m_WorkingVector.modify(), weight_vector, cutoff);
            auto score = m_Objective->value(m_WorkingVector);
            ++step_count;
            return BoundData{cutoff, nnz, score};
        };

        // we assume that [exp(mean_log - 2std_var), exp(mean_log + 2std_var)] is a good interval
        auto at_mean = check_bound( mean_log );
        if(at_mean.Loss > tolerance) {
            // ok, mean is an upper bound
            // let's try the lower bound then
            BoundData minus_std = check_bound(mean_log - std_log);
            if(minus_std.Loss > tolerance) {
                record(STAT_INITIAL_STEPS, step_count);
                return {{0, weight_vector.size(), initial_lower}, minus_std};
            } else {
                record(STAT_INITIAL_STEPS, step_count);
                return {minus_std, at_mean};
            }
        } else {
            // ok, mean is a lower bound
            BoundData plus_std = check_bound(mean_log + std_log);
            if(plus_std.Loss > tolerance) {
                record(STAT_INITIAL_STEPS, step_count);
                return {at_mean, plus_std};
            } else {
                // one more naive trial:
                BoundData plus_3_std = check_bound(mean_log + 3 * std_log);
                if(plus_3_std.Loss > tolerance) {
                    record(STAT_INITIAL_STEPS, step_count);
                    return {plus_std, plus_3_std};
                } else {
                    BoundData at_max = check_bound( std::log(weight_vector.maxCoeff()) );
                    record(STAT_INITIAL_STEPS, step_count);
                    return {plus_3_std, at_max};
                }
            }
        }
    }
}

std::shared_ptr<postproc::PostProcessFactory> postproc::create_sparsify(real_t tolerance) {
    return std::make_shared<GenericPostProcFactory<Sparsify, real_t>>(tolerance);
}
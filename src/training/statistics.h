// Copyright (c) 2021, Aalto University, developed by Erik Schultheis
// All rights reserved.
//
// SPDX-License-Identifier: MIT

#ifndef DISMEC_STATISTICS_H
#define DISMEC_STATISTICS_H

#include <unordered_map>
#include <memory>
#include <mutex>
#include <vector>
#include "fwd.h"
#include "matrix_types.h"
#include "stats/tracked.h"
#include <nlohmann/json_fwd.hpp>

namespace dismec
{
    class ResultStatsGatherer : public stats::Tracked {
    public:
        ResultStatsGatherer();
        virtual ~ResultStatsGatherer();
        virtual void record_result(const DenseRealVector& weights, const solvers::MinimizationResult& result) = 0;
        virtual void start_label(label_id_t label) = 0;
        virtual void start_training(const DenseRealVector& init_weights) = 0;
    };

    class TrainingStatsGatherer {
        using thread_id_t = dismec::parallel::thread_id_t;
    public:
        TrainingStatsGatherer(std::string source, std::string target_file);
        ~TrainingStatsGatherer();

        /// NOTE: these functions will be called concurrently
        void setup_minimizer(thread_id_t thread, stats::Tracked& minimizer);
        void setup_initializer(thread_id_t thread, stats::Tracked& initializer);
        void setup_objective(thread_id_t thread, stats::Tracked& objective);
        void setup_postproc(thread_id_t thread, stats::Tracked& objective);
        std::unique_ptr<ResultStatsGatherer> create_results_gatherer(thread_id_t thread, const std::shared_ptr<const TrainingSpec>& spec);

        void finalize();

        nlohmann::json to_json() const;
    private:
        struct StatData {
            stats::StatisticMetaData Meta;
            std::unique_ptr<stats::Statistics> Stat;
        };
        std::unordered_map<std::string, StatData> m_Merged;
        using collection_ptr_t = std::shared_ptr<stats::StatisticsCollection>;
        // we need to have this data per thread to 1) correctly associate per-thread tags and 2) ensure consistent order when merging.
        std::vector<std::unordered_map<std::string, collection_ptr_t>> m_PerThreadCollections;

        std::mutex m_Lock;

        std::string m_TargetFile;

        void add_accu(const std::string& key, thread_id_t thread, const std::shared_ptr<stats::StatisticsCollection>& accumulator);

        std::unique_ptr<nlohmann::json> m_Config;
    };
}

#endif //DISMEC_STATISTICS_H

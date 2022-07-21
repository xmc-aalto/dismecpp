// Copyright (c) 2021, Aalto University, developed by Erik Schultheis
// All rights reserved.
//
// SPDX-License-Identifier: MIT

#include "statistics.h"
#include "solver/minimizer.h"
#include "stats/collection.h"
#include "stats/statistics.h"
#include "parallel/task.h"  // for thread_id_t
#include "initializer.h"
#include "spec.h"
#include "data/data.h"
#include <fstream>
#include <iomanip>
#include <nlohmann/json.hpp>

using namespace dismec;

TrainingStatsGatherer::TrainingStatsGatherer(std::string source, std::string target_file) :
    m_TargetFile(std::move(target_file)) {
    if(source.empty()) {
        m_Config = std::make_unique<nlohmann::json>();
    } else {
        std::fstream source_stream(source, std::fstream::in);
        m_Config = std::make_unique<nlohmann::json>(nlohmann::json::parse(source_stream));
    }
}

void TrainingStatsGatherer::setup_minimizer(thread_id_t thread, stats::Tracked& minimizer) {
    add_accu("minimizer", thread, minimizer.get_stats());
}

void TrainingStatsGatherer::setup_initializer(thread_id_t thread, stats::Tracked& initializer) {
    add_accu("init", thread, initializer.get_stats());
}

void TrainingStatsGatherer::setup_objective(thread_id_t thread, stats::Tracked& objective) {
    add_accu("objective", thread,  objective.get_stats());
}
void TrainingStatsGatherer::setup_postproc(thread_id_t thread, stats::Tracked& post) {
    add_accu("post", thread, post.get_stats());
}

void TrainingStatsGatherer::finalize() {
    // Outer loop iterates over thread indices.
    for(auto&& entries : m_PerThreadCollections) {
        // inner loop iterates over map keys
        for (auto&& accu : entries) {
            for (auto&& meta : accu.second->get_statistics_meta()) {
                if (!accu.second->is_enabled_by_name(meta.Name)) continue;
                std::string qualified_name = accu.first + '.'+ meta.Name;
                if (m_Merged.count(qualified_name) == 0) {
                    m_Merged[qualified_name] = {meta, accu.second->get_stat(meta.Name).clone()};
                }

                m_Merged.at(qualified_name).Stat->merge(accu.second->get_stat(meta.Name));
            }
        }
    }
}

TrainingStatsGatherer::~TrainingStatsGatherer() {
    if(!m_TargetFile.empty()) {
        nlohmann::json result = to_json();
        std::fstream target(m_TargetFile, std::fstream::out);
        target << std::setw(2) << result << "\n";
    }
}

nlohmann::json TrainingStatsGatherer::to_json() const {
    nlohmann::json result;
    for(const auto& stat : m_Merged) {
        auto raw = stat.second.Stat->to_json();
        if(!stat.second.Meta.Unit.empty())
            raw["Unit"] = stat.second.Meta.Unit;
        result[stat.first] = std::move(raw);
    }
    return result;
}


ResultStatsGatherer::ResultStatsGatherer()  = default;
ResultStatsGatherer::~ResultStatsGatherer() = default;

namespace {
    constexpr const stats::stat_id_t STAT_FINAL_LOSS{0};
    constexpr const stats::stat_id_t STAT_FINAL_GRAD{1};
    constexpr const stats::stat_id_t STAT_INIT_LOSS{2};
    constexpr const stats::stat_id_t STAT_INIT_GRAD{3};
    constexpr const stats::stat_id_t STAT_NUM_ITERS{4};
    constexpr const stats::stat_id_t STAT_DURATION{5};
    constexpr const stats::stat_id_t STAT_WEIGHT_VECTOR{6};
    constexpr const stats::stat_id_t STAT_LABEL_ID{7};
    constexpr const stats::stat_id_t STAT_LABEL_FREQ{8};
    constexpr const stats::stat_id_t STAT_INIT_VECTOR{9};
    constexpr const stats::stat_id_t STAT_TRAINING_SHIFT{10};

    constexpr const stats::tag_id_t  TAG_LABEL_ID{0};
    constexpr const stats::tag_id_t  TAG_LABEL_FREQ{1};

    class DefaultGatherer: public ResultStatsGatherer {
    public:

        DefaultGatherer(const TrainingSpec& spec) {
            declare_stat(STAT_FINAL_LOSS, {"final_loss", "|g|"});
            declare_stat(STAT_FINAL_GRAD, {"final_grad", "loss"});
            declare_stat(STAT_INIT_LOSS, {"initial_loss", "loss"});
            declare_stat(STAT_INIT_GRAD, {"initial_grad", "|g|"});
            declare_stat(STAT_NUM_ITERS, {"iters", "#iters"});
            declare_stat(STAT_DURATION, {"duration", "duration [ms]"});
            declare_stat(STAT_WEIGHT_VECTOR, {"weights"});
            declare_stat(STAT_LABEL_ID, {"label_id"});
            declare_stat(STAT_LABEL_FREQ, {"label_freq"});
            declare_stat(STAT_INIT_VECTOR, {"initial_weights"});
            declare_stat(STAT_TRAINING_SHIFT, {"training_shift"});
            declare_tag(TAG_LABEL_ID, "label");
            declare_tag(TAG_LABEL_FREQ, "label_freq");
            m_Data = &spec.get_data();
        }


        void start_label(label_id_t label) override {
            int pos = m_Data->num_positives(label);
            set_tag(TAG_LABEL_ID, label.to_index());
            set_tag(TAG_LABEL_FREQ, pos);
            record(STAT_LABEL_ID, label.to_index());
            record(STAT_LABEL_FREQ, pos);
        }

        void start_training(const DenseRealVector& init_weights) override {
            record(STAT_INIT_VECTOR, init_weights);
            // if we want to record the shift in the weight vector, we need to cache it here
            if(get_stats()->is_enabled(STAT_TRAINING_SHIFT)) {
                if(m_InitWeightsCache)
                    *m_InitWeightsCache = init_weights;
                else
                    m_InitWeightsCache = std::make_unique<DenseRealVector>(init_weights);
            }
        }

        void record_result(const DenseRealVector& weights, const solvers::MinimizationResult& result) override {
            record(STAT_FINAL_LOSS, real_t(result.FinalValue));
            record(STAT_FINAL_GRAD, real_t(result.FinalGrad));
            record(STAT_INIT_LOSS, real_t(result.InitialValue));
            record(STAT_INIT_GRAD, real_t(result.InitialGrad));
            record(STAT_NUM_ITERS, result.NumIters);
            record(STAT_DURATION, result.Duration.count());
            record(STAT_WEIGHT_VECTOR, weights);

            record(STAT_TRAINING_SHIFT, [&]() -> DenseRealVector {
                return weights - *m_InitWeightsCache;
            });
        }

        const DatasetBase* m_Data;

        std::unique_ptr<DenseRealVector> m_InitWeightsCache;
    };
}

std::unique_ptr<ResultStatsGatherer> TrainingStatsGatherer::create_results_gatherer(parallel::thread_id_t thread, const std::shared_ptr<const TrainingSpec>& spec) {
    auto gather = std::make_unique<DefaultGatherer>(*spec);
    add_accu("result", thread, gather->get_stats());
    return gather;
}

void TrainingStatsGatherer::add_accu(std::string key, parallel::thread_id_t thread, const std::shared_ptr<stats::StatisticsCollection>& accumulator) {
    std::lock_guard<std::mutex> lck{m_Lock};
    if(thread.to_index() >= m_PerThreadCollections.size()) {
        m_PerThreadCollections.resize(thread.to_index() + 1);
    }
    // Iterate over all existing collections. We need the two nested loops to first iterate over
    // the names, and then
    for(auto& entry : m_PerThreadCollections.at(thread.to_index())) {
        accumulator->provide_tags(*entry.second);
        entry.second->provide_tags(*accumulator);
    }
    auto result = m_PerThreadCollections.at(thread.to_index()).emplace(key, accumulator);
    assert(result.second);

    if(m_Config->contains(key)) {
        for (auto& entry : m_Config->at(key).items()) {
            if(!accumulator->has_stat(entry.key())) {
                spdlog::warn("Statistics {} has been defined in json, but has not been declared", entry.key());
                continue;
            }
            accumulator->register_stat(entry.key(), stats::make_stat_from_json(entry.value()));
        }
    }
}

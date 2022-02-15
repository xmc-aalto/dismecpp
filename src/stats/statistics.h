// Copyright (c) 2021, Aalto University, developed by Erik Schultheis
// All rights reserved.
//
// SPDX-License-Identifier: MIT

#ifndef DISMEC_STATS_STATISTICS_H
#define DISMEC_STATS_STATISTICS_H

#include "stats_base.h"
#include <unordered_map>
#include <functional>

namespace stats {

    class CounterStat final : public StatImplBase<CounterStat> {
    public:
        ~CounterStat() override = default;
        void record(long integer) override;

        [[nodiscard]] std::unique_ptr<Statistics> clone() const override;
        void merge(const CounterStat& other);
        [[nodiscard]] nlohmann::json to_json() const override;
    private:
        long m_Counter = 0;
    };

    class BasicStat final : public StatImplBase<BasicStat> {
    public:
        ~BasicStat() override = default;
        void record(long value) override;
        void record(real_t value) override;

        [[nodiscard]] std::unique_ptr<Statistics> clone() const override;
        void merge(const BasicStat& other);
        [[nodiscard]] nlohmann::json to_json() const override;
    private:
        long m_Counter = 0;
        double m_Sum = 0;
        double m_SumSquared = 0;
    };

    class TaggedStat final : public StatImplBase<TaggedStat> {
    public:
        TaggedStat(std::string tag, int max_tag, std::string transform_name = {}, std::function<double(double)> transform = {});
        ~TaggedStat() override = default;
        void record(long value) override;
        void record(real_t value) override;

        [[nodiscard]] std::unique_ptr<Statistics> clone() const override;
        void merge(const TaggedStat& other);
        [[nodiscard]] nlohmann::json to_json() const override;

        void setup(const StatisticsCollection& source) override;
    private:
        std::vector<long> m_Counters;
        std::vector<double> m_Sums;
        std::vector<double> m_SumsSquared;

        TagContainer m_Tag;
        int m_MaxTag = -1;

        std::function<double(double)> m_Transform;
        std::string m_TransformName;
    };

    class MultiStat final : public StatImplBase<MultiStat> {
    public:
        MultiStat(std::unordered_map<std::string, std::unique_ptr<Statistics>> ss);
        ~MultiStat() override = default;
        void record(long value) override;
        void record(real_t value) override;
        void record(const DenseRealVector& vector) override;

        [[nodiscard]] std::unique_ptr<Statistics> clone() const override;
        void merge(const MultiStat& other);
        [[nodiscard]] nlohmann::json to_json() const override;

        void setup(const StatisticsCollection& source) override;
    private:
        using stats_map_t = std::unordered_map<std::string, std::unique_ptr<Statistics>>;
        stats_map_t m_SubStats;

        template<class T>
        void do_record(T&&);
    };

    class FullRecordStat final: public StatImplBase<FullRecordStat> {
    public:
        FullRecordStat() = default;
        ~FullRecordStat() override = default;

        void record(long value) override;
        void record(real_t value) override;
        [[nodiscard]] std::unique_ptr<Statistics> clone() const override;
        void merge(const FullRecordStat& other);
        [[nodiscard]] nlohmann::json to_json() const override;

    private:
        std::vector<real_t> m_Data;
    };

    class VectorReductionStat final: public StatImplBase<VectorReductionStat> {
    public:
        VectorReductionStat(std::unique_ptr<Statistics> stat, std::string reduction);
        ~VectorReductionStat() override = default;

        // this is needed so that the default implementation for vector recording in StatImplBase does not cause an error.
        using Statistics::record;
        void record(const DenseRealVector& value) override;
        [[nodiscard]] std::unique_ptr<Statistics> clone() const override;
        void merge(const VectorReductionStat& other);
        [[nodiscard]] nlohmann::json to_json() const override;

    private:
        std::unique_ptr<Statistics> m_Target;
        std::function<real_t(const DenseRealVector&)> m_Reduction;
        std::string m_ReductionName;
    };
}

#endif //DISMEC_STATS_STATISTICS_H

// Copyright (c) 2021, Aalto University, developed by Erik Schultheis
// All rights reserved.
//
// SPDX-License-Identifier: MIT

#include "statistics.h"
#include "collection.h"
#include <nlohmann/json.hpp>

using namespace stats;

void CounterStat::record(long integer) {
    m_Counter += integer;
}

std::unique_ptr<Statistics> CounterStat::clone() const {
    return std::make_unique<CounterStat>();
}

void CounterStat::merge(const CounterStat& other) {
    m_Counter += other.m_Counter;
}

nlohmann::json CounterStat::to_json() const {
    return {{"Counter", m_Counter}, {"Type", "Counter"}};
}

void BasicStat::record(long value) {
    record(real_t(value));
}

void BasicStat::record(real_t value)  {
    ++m_Counter;
    m_Sum += value;
    m_SumSquared += value*value;
}

std::unique_ptr<Statistics> BasicStat::clone() const {
    return std::make_unique<BasicStat>();
}

void BasicStat::merge(const BasicStat& other) {
    m_Counter += other.m_Counter;
    m_Sum += other.m_Sum;
    m_SumSquared += other.m_SumSquared;
}

nlohmann::json BasicStat::to_json() const {
    return {{"Counter", m_Counter}, {"Sum", m_Sum}, {"SumSquared", m_SumSquared}, {"Type", "Basic"},
            {"Mean", m_Sum / double(m_Counter)}};
}

TaggedStat::TaggedStat(std::string tag, int max_tag, std::string transform_name, std::function<double(double)> transform ) :
    m_Tag( TagContainer::create_empty_container(std::move(tag)) ),
    m_MaxTag(max_tag),
    m_Transform( std::move(transform) ),
    m_TransformName( std::move(transform_name) )
    {

}
void TaggedStat::record(long value) {
    record(real_t(value));
}

void TaggedStat::record(real_t value)  {
    int tag = m_Tag.get_value();
    if(tag < 0)
        throw std::logic_error("Missing tag!");
    if(tag > m_MaxTag && m_MaxTag >= 0)
        tag = m_MaxTag;

    if(tag >= m_Counters.size()) {
        m_Counters.resize(tag + 1);
        m_Sums.resize(tag + 1);
        m_SumsSquared.resize(tag + 1);
    }
    ++m_Counters[tag];
    if(m_Transform) {
        value = m_Transform(value);
    }
    m_Sums[tag] += value;
    m_SumsSquared[tag] += value * value;
}

std::unique_ptr<Statistics> TaggedStat::clone() const {
    return std::make_unique<TaggedStat>(m_Tag.get_name(), m_MaxTag, m_TransformName, m_Transform);
}

void TaggedStat::merge(const TaggedStat& other) {
    int other_size = other.m_Counters.size();
    if(other_size > m_Counters.size()) {
        m_Counters.resize(other_size);
        m_Sums.resize(other_size);
        m_SumsSquared.resize(other_size);
    }

    for(int i = 0; i < other_size; ++i) {
        m_Counters[i] += other.m_Counters[i];
        m_Sums[i] += other.m_Sums[i];
        m_SumsSquared[i] += other.m_SumsSquared[i];
    }
}

nlohmann::json TaggedStat::to_json() const {
    return {{"Counters", m_Counters}, {"Sums", m_Sums}, {"SumsSquared", m_SumsSquared},
            {"Type", "BasicTagged"}, {"Transform", m_TransformName}};
}

void TaggedStat::setup(const StatisticsCollection& source) {
    m_Tag = source.get_tag_by_name(m_Tag.get_name());
}

MultiStat::MultiStat(std::unordered_map<std::string, std::unique_ptr<Statistics>> ss) : m_SubStats(std::move(ss)) {

}

void MultiStat::record(long value) {
    do_record(value);
}
void MultiStat::record(real_t value) {
    do_record(value);
}
void MultiStat::record(const DenseRealVector& vector) {
    do_record(vector);
}

template<class T>
void MultiStat::do_record(T&& value) {
    for(const auto& entry : m_SubStats) {
        entry.second->record(std::forward<T>(value));
    }
}

std::unique_ptr<Statistics> MultiStat::clone() const {
    stats_map_t new_map;
    for(const auto& entry : m_SubStats) {
        new_map.emplace(entry.first, entry.second->clone());
    }
    return std::make_unique<MultiStat>(std::move(new_map));
}

void MultiStat::merge(const MultiStat& other) {
    for(const auto& entry : m_SubStats) {
        entry.second->merge( *other.m_SubStats.at(entry.first) );
    }
}

nlohmann::json MultiStat::to_json() const {
    nlohmann::json result;
    result["Type"] = "Multi";
    nlohmann::json data;
    for(const auto& entry : m_SubStats) {
        data[entry.first] = entry.second->to_json();
    }
    result["Data"] = std::move(data);
    return result;
}

void MultiStat::setup(const StatisticsCollection& source) {
    for(const auto& entry : m_SubStats) {
        entry.second->setup(source);
    }
}

void FullRecordStat::record(real_t value) {
    m_Data.push_back(value);
}

void FullRecordStat::record(long value) {
    m_Data.push_back(value);
}

std::unique_ptr<Statistics> FullRecordStat::clone() const {
    return std::make_unique<FullRecordStat>();
}
void FullRecordStat::merge(const FullRecordStat& other) {
    m_Data.reserve(m_Data.size() + other.m_Data.size());
    m_Data.insert(end(m_Data), begin(other.m_Data), end(other.m_Data));
}

nlohmann::json FullRecordStat::to_json() const {
    return {{"Type", "Full"}, {"Values", m_Data}};
}

VectorReductionStat::VectorReductionStat(std::unique_ptr<Statistics> stat, std::string reduction) :
    m_Target( std::move(stat) ), m_ReductionName(std::move(reduction)) {
    if(m_ReductionName == "L1") {
        m_Reduction = [](const DenseRealVector& v) -> real_t { return v.lpNorm<1>(); };
    } else if(m_ReductionName == "L2") {
        m_Reduction = [](const DenseRealVector& v) -> real_t { return v.norm(); };
    } else if(m_ReductionName == "L2Squared") {
        m_Reduction = [](const DenseRealVector& v) -> real_t { return v.squaredNorm(); };
    } else if(m_ReductionName == "Linf") {
        m_Reduction = [](const DenseRealVector& v) -> real_t { return v.lpNorm<Eigen::Infinity>(); };
    } else {
        throw std::runtime_error("Unknown reduction operation");
    }
};

void VectorReductionStat::record(const DenseRealVector& value) {
    m_Target->record(real_t{m_Reduction(value)});
}

std::unique_ptr<Statistics> VectorReductionStat::clone() const {
    return std::make_unique<VectorReductionStat>(m_Target->clone(), m_ReductionName);
}

void VectorReductionStat::merge(const VectorReductionStat& other) {
    m_Target->merge(*other.m_Target);
}

nlohmann::json VectorReductionStat::to_json() const {
    return m_Target->to_json();
}

#include "histogram.h"
std::unique_ptr<stats::Statistics> stats::make_stat_from_json(const nlohmann::json& source) {
    auto type = source.at("type").get<std::string>();
    if(type == "Basic") {
        return std::make_unique<BasicStat>();
    } else if (type == "Counter") {
        return std::make_unique<CounterStat>();
    } else if (type == "Tagged") {
        std::function<double(double)> transform;
        std::string transform_name = "lin";
        if(source.contains("transform")) {
            transform_name= source.at("transform");
            if(transform_name == "log") {
                transform = [](double d){ return std::log(d); };
            }
        }
        int max_tag = -1;
        if(source.contains("max_tag")) {
            max_tag = source.at("max_tag").get<int>();
        }

        return std::make_unique<TaggedStat>(source.at("tag").get<std::string>(), max_tag,
                                            std::move(transform_name), std::move(transform));
    } else if (type == "LinHist") {
        return make_linear_histogram(source.at("bins").get<int>(),
                                     source.at("min").get<real_t>(), source.at("max").get<real_t>());
    } else if (type == "LogHist") {
        return make_logarithmic_histogram(source.at("bins").get<int>(),
                                          source.at("min").get<real_t>(), source.at("max").get<real_t>());
    } else if (type == "TagLinHist") {
        return make_linear_histogram(source.at("tag").get<std::string>(),
                                     source.at("max_tag").get<int>(), source.at("bins").get<int>(),
                                     source.at("min").get<real_t>(),
                                     source.at("max").get<real_t>());
    } else if (type == "TagLogHist") {
        return make_logarithmic_histogram(
                source.at("tag").get<std::string>(),
                source.at("max_tag").get<int>(), source.at("bins").get<int>(),
                source.at("min").get<real_t>(),
                source.at("max").get<real_t>());
    } else if (type == "Multi") {
        std::unordered_map<std::string, std::unique_ptr<Statistics>> sub_stats;
        for(auto& sub : source.at("stats").items()) {
            sub_stats[sub.key()] = make_stat_from_json(sub.value());
        }
        return std::make_unique<MultiStat>(std::move(sub_stats));
    } else if (type == "Full") {
        return std::make_unique<FullRecordStat>();
    } else if (type == "VectorReduction") {
        return std::make_unique<VectorReductionStat>(make_stat_from_json(source.at("stat")), source.at("reduction"));
    }
    else {
        throw std::runtime_error("Unknown statistics type");
    }
}
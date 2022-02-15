// Copyright (c) 2021, Aalto University, developed by Erik Schultheis
// All rights reserved.
//
// SPDX-License-Identifier: MIT

#include "histogram.h"
#include "collection.h"
#include <nlohmann/json.hpp>

using namespace boost::histogram;
using namespace stats;


namespace axis = boost::histogram::axis;

using log_axis_t = axis::regular<real_t, axis::transform::log>;
using lin_axis_t = axis::regular<real_t>;

namespace {
    template<class T>
    void combine_histograms(T& target, const T& source) {
        auto index_s = indexed(source, coverage::all);
        auto index_t = indexed(target, coverage::all);

        auto s_it = begin(index_s);
        auto t_it = begin(index_t);
        auto end_it = end(index_s);
        while(s_it != end_it) {
            **t_it += **s_it;
            ++t_it;
            ++s_it;
        }
    }

    template<class T, class U>
    void combine_histograms(T& target, const U& source) {
        throw std::logic_error("Trying to combine histograms of different type");
    }

    template<class... Args>
    nlohmann::json to_json(const histogram<Args...>& hist) {
        nlohmann::json result;
        nlohmann::json lower_bounds;
        nlohmann::json upper_bounds;
        nlohmann::json counts;

        for(auto&& b : indexed(hist, coverage::all)) {
            lower_bounds.push_back(b.bin(0).lower());
            upper_bounds.push_back(b.bin(0).upper());
            counts.push_back(long(*b));
        }

        result["Lower"] = std::move(lower_bounds);
        result["Upper"] = std::move(upper_bounds);
        result["Count"] = std::move(counts);
        return result;
    }

    template<class Axis>
    std::string get_type();

    template<>
    std::string get_type<lin_axis_t>() {
        return "Hist1dLin";
    }
    template<>
    std::string get_type<log_axis_t>() {
        return "Hist1dLog";
    }

    template<class Axis>
    real_t transform(real_t value) {
        return value;
    }

    template<>
    real_t transform<log_axis_t>(real_t value) {
        return std::abs(value);
    }
}


template<class Axis>
HistogramStat<Axis>::HistogramStat(int bins, real_t min, real_t max) :
    m_Histogram(make_histogram(Axis(bins, min, max))),
    m_Bins(bins), m_Min(min), m_Max(max) {

}

template<class Axis>
void HistogramStat<Axis>::record(long value) {
    record(real_t(value));
}

template<class Axis>
void HistogramStat<Axis>::record(real_t value) {
    m_Histogram(value);
}

template<class Axis>
void HistogramStat<Axis>::record(const DenseRealVector& vector) {
    for(int i = 0; i < vector.size(); ++i) {
        m_Histogram(transform<Axis>(vector.coeff(i)));
    }
}

template<class Axis>
std::unique_ptr<Statistics> HistogramStat<Axis>::clone() const {
    return std::make_unique<HistogramStat<Axis>>(m_Bins, m_Min, m_Max);
}
template<class Axis>
void HistogramStat<Axis>::merge(const HistogramStat<Axis>& other) {
    combine_histograms(m_Histogram,  other.m_Histogram);
}

template<class Axis>
nlohmann::json HistogramStat<Axis>::to_json() const {
    auto temp = ::to_json(m_Histogram);
    temp["Type"] = get_type<Axis>();
    return temp;
}

std::unique_ptr<Statistics> stats::make_linear_histogram(int bins, real_t min, real_t max) {
    return std::make_unique<HistogramStat<lin_axis_t>>(bins, min, max);
}

std::unique_ptr<Statistics> stats::make_logarithmic_histogram(int bins, real_t min, real_t max) {
    return std::make_unique<HistogramStat<log_axis_t>>(bins, min, max);
}

template<class Axis>
TaggedHistogramStat<Axis>::TaggedHistogramStat(std::string tag, int max_tag, int bins, real_t min, real_t max) :
    m_Bins(bins), m_MaxTag(max_tag), m_Min(min), m_Max(max), m_Tag(TagContainer::create_empty_container(std::move(tag))) {

}

template<class Axis>
void TaggedHistogramStat<Axis>::record(long value) {
    record(real_t(value));
}

template<class Axis>
auto TaggedHistogramStat<Axis>::get_active_hist() -> histogram_t& {
    int tag = m_Tag.get_value();
    if(tag < 0)
        throw std::logic_error("Missing tag!");
    if(tag > m_MaxTag)
        tag = m_MaxTag;

    while(tag >= m_Histograms.size()) {
        m_Histograms.push_back(make_histogram(Axis(m_Bins, m_Min, m_Max)));
    }
    return m_Histograms[tag];
}

template<class Axis>
void TaggedHistogramStat<Axis>::record(real_t value) {
    get_active_hist()(value);
}

template<class Axis>
void TaggedHistogramStat<Axis>::record(const DenseRealVector& vector) {
    auto& hist = get_active_hist();
    for(int i = 0; i < vector.size(); ++i) {
        hist(transform<Axis>(vector.coeff(i)));
    }
}

template<class Axis>
std::unique_ptr<Statistics> TaggedHistogramStat<Axis>::clone() const {
    return std::make_unique<TaggedHistogramStat<Axis>>(m_Tag.get_name(), m_MaxTag, m_Bins, m_Min, m_Max);
}

template<class Axis>
void TaggedHistogramStat<Axis>::merge(const TaggedHistogramStat<Axis>& other) {
    m_Histograms.reserve(other.m_Histograms.size());
    while(other.m_Histograms.size() > m_Histograms.size()) {
        m_Histograms.emplace_back(make_histogram(Axis(m_Bins, m_Min, m_Max)));
    }

    for(int i = 0; i < other.m_Histograms.size(); ++i) {
        combine_histograms(m_Histograms[i], other.m_Histograms[i]);
    }
}

template<class Axis>
nlohmann::json TaggedHistogramStat<Axis>::to_json() const {
    nlohmann::json result;
    for(int i = 0; i < m_Histograms.size(); ++i) {
        auto temp = ::to_json(m_Histograms[i]);
        result["Counts"].push_back(temp["Count"]);
        if ( i == 0 ) {
            result["Lower"] = std::move(temp["Lower"]);
            result["Upper"] = std::move(temp["Upper"]);
        }
    }
    result["Type"] = "Tagged" + get_type<Axis>();
    return result;
}

template<class Axis>
void TaggedHistogramStat<Axis>::setup(const StatisticsCollection& source) {
    m_Tag = source.get_tag_by_name(m_Tag.get_name());
}

std::unique_ptr<Statistics> stats::make_linear_histogram(std::string tag, int max_tag, int bins, real_t min, real_t max) {
    return std::make_unique<TaggedHistogramStat<lin_axis_t>>(std::move(tag), max_tag, bins, min, max);
}

std::unique_ptr<Statistics> stats::make_logarithmic_histogram(std::string tag, int max_tag, int bins, real_t min, real_t max) {
    return std::make_unique<TaggedHistogramStat<log_axis_t>>(std::move(tag), max_tag, bins, min, max);
}
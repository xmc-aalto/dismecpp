// Copyright (c) 2021, Aalto University, developed by Erik Schultheis
// All rights reserved.
//
// SPDX-License-Identifier: MIT

#ifndef DISMEC_HISTOGRAM_H
#define DISMEC_HISTOGRAM_H

#include "stats_base.h"
#if defined __has_include
#if __has_include(<boost/histogram.hpp>)
#include <boost/histogram.hpp>
#define DISMEC_STATS_SUPPORT_HISTOGRAM 1
#endif
#endif

namespace dismec::stats
{
#if DISMEC_STATS_SUPPORT_HISTOGRAM
    template<class Axis>
    class HistogramStat final : public StatImplBase<HistogramStat<Axis>> {
    public:
        HistogramStat(int bins, real_t min, real_t max);
        ~HistogramStat() override = default;

        void record(long value) override;
        void record(real_t value) override;
        void record(const DenseRealVector& vector) override;

        [[nodiscard]] std::unique_ptr<Statistics> clone() const override;
        void merge(const HistogramStat<Axis>& other);

        [[nodiscard]] nlohmann::json to_json() const override;
    private:
        boost::histogram::histogram<std::tuple<Axis>> m_Histogram;

        // we keep a copy of the constructor parameters for convenience, so we can easily clone!
        int m_Bins;
        real_t m_Min;
        real_t m_Max;
    };

    template<class Axis>
    class TaggedHistogramStat final : public StatImplBase<TaggedHistogramStat<Axis>> {
    public:
        TaggedHistogramStat(std::string tag, int max_tag, int bins, real_t min, real_t max);
        ~TaggedHistogramStat() override = default;

        void record(long value) override;
        void record(real_t value) override;
        void record(const DenseRealVector& vector) override;

        [[nodiscard]] std::unique_ptr<Statistics> clone() const override;
        void merge(const TaggedHistogramStat<Axis>& other);

        [[nodiscard]] nlohmann::json to_json() const override;
        void setup(const StatisticsCollection& source) override;
    private:
        using histogram_t = boost::histogram::histogram<std::tuple<Axis>>;
        std::vector<histogram_t> m_Histograms;

        histogram_t& get_active_hist();

        // we keep a copy of the constructor parameters for convenience, so we can easily clone!
        int m_Bins;
        int m_MaxTag;
        real_t m_Min;
        real_t m_Max;

        TagContainer m_Tag;
    };
#else
#warning "Compiling without histogram support"
#endif

    std::unique_ptr<Statistics> make_linear_histogram(int bins, real_t min, real_t max);
    std::unique_ptr<Statistics> make_logarithmic_histogram(int bins, real_t min, real_t max);
    std::unique_ptr<Statistics> make_linear_histogram(std::string tag, int max_tag, int bins, real_t min, real_t max);
    std::unique_ptr<Statistics> make_logarithmic_histogram(std::string tag, int max_tag, int bins, real_t min, real_t max);

}

#endif //DISMEC_HISTOGRAM_H

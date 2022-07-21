// Copyright (c) 2021, Aalto University, developed by Erik Schultheis
// All rights reserved.
//
// SPDX-License-Identifier: MIT

#ifndef DISMEC_STATS_TIMER_H
#define DISMEC_STATS_TIMER_H

#include <chrono>
#include "stat_id.h"

namespace dismec::stats {
    class StatisticsCollection;

    class ScopeTimer {
        using clock_t = std::chrono::high_resolution_clock;
        stat_id_t m_Target;
        bool m_Enabled = false;
        clock_t::time_point m_Start;
        StatisticsCollection* m_Accu;

        friend ScopeTimer record_scope_time(StatisticsCollection& accumulator, stat_id_t id);

        ScopeTimer(StatisticsCollection* accu, stat_id_t id) :
                m_Target(id), m_Enabled(is_enabled(accu, id)), m_Accu(accu) {
            if(m_Enabled) {
                m_Start = clock_t::now();
            }
        }

        // We provide these two functions as non-inline definitions so we don't have to include `collection.h`
        // here.
        void record_duration();
        [[nodiscard]] static bool is_enabled(const StatisticsCollection* accu, stat_id_t stat);
    public:
        ~ScopeTimer() {
            if(m_Enabled) {
                record_duration();
            }
        }
        ScopeTimer(const ScopeTimer&) = delete;
        ScopeTimer operator=(const ScopeTimer&) = delete;

        /*!
         * \brief Move constructor. Needs to disable recording for the moved-from timer.
         */
        ScopeTimer(ScopeTimer&& other) noexcept :
                m_Target(other.m_Target), m_Enabled(other.m_Enabled), m_Start(other.m_Start), m_Accu(other.m_Accu) {
            other.m_Enabled = false;        // disable recording for other.
        }
    };

    inline ScopeTimer record_scope_time(StatisticsCollection& accumulator, stat_id_t id) {
        return {&accumulator, id};
    }
}

#endif //DISMEC_STATS_TIMER_H

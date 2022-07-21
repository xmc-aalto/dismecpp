// Copyright (c) 2021, Aalto University, developed by Erik Schultheis
// All rights reserved.
//
// SPDX-License-Identifier: MIT

#include "timer.h"
#include "collection.h"

using dismec::stats::ScopeTimer;

void ScopeTimer::record_duration() {
    auto dt = clock_t::now() - m_Start;
    m_Accu->record(m_Target, std::chrono::duration_cast<std::chrono::microseconds>(dt).count());
}

bool ScopeTimer::is_enabled(const StatisticsCollection* accu, stat_id_t stat) {
    return accu->is_enabled(stat);
}

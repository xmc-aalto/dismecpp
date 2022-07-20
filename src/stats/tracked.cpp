// Copyright (c) 2021, Aalto University, developed by Erik Schultheis
// All rights reserved.
//
// SPDX-License-Identifier: MIT

#include "tracked.h"
#include "collection.h"

using namespace dismec::stats;

Tracked::~Tracked() = default;

Tracked::Tracked() : m_Collection(std::make_shared<StatisticsCollection>()) {
}

void Tracked::declare_stat(stats::stat_id_t index, StatisticMetaData name) {
    m_Collection->declare_stat(index, std::move(name));
}

void Tracked::register_stat(const std::string& name, std::unique_ptr<Statistics> stat) {
    m_Collection->register_stat(name, std::move(stat));
}

void Tracked::declare_tag(tag_id_t index, std::string name) {
    m_Collection->declare_tag(index, std::move(name));
}
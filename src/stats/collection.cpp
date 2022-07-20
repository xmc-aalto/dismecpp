// Copyright (c) 2021, Aalto University, developed by Erik Schultheis
// All rights reserved.
//
// SPDX-License-Identifier: MIT

#include "collection.h"
#include "spdlog/spdlog.h"

using namespace dismec::stats;

void StatisticsCollection::enable(stat_id_t stat) {
    if(!m_Statistics.at(stat.to_index()))
        throw std::logic_error("Cannot enable tracking of id, because no `Statistics` object has been assigned.");
    m_Enabled.at(stat.to_index()) = true;
}

void StatisticsCollection::disable(stats::stat_id_t stat) {
    m_Enabled.at(stat.to_index()) = false;
}

namespace {
    stat_id_t str_to_id(const std::string& str, const std::vector<StatisticMetaData>& names) {
        auto dist = std::distance(begin(names), std::find_if(begin(names), end(names), [&](auto&& v){ return v.Name == str; }));
        if(dist < 0 || dist >= names.size()) {
            throw std::invalid_argument("No statistics of the given name has been declared.");
        }
        return stat_id_t{dist};
    }
}

void StatisticsCollection::enable(const std::string& stat) {
    enable(str_to_id(stat, m_MetaData));
}

void StatisticsCollection::disable(const std::string& stat) {
    disable(str_to_id(stat, m_MetaData));
}

void StatisticsCollection::declare_stat(stat_id_t index, StatisticMetaData meta) {
    if(index.to_index() < m_Enabled.size()) {
        throw std::invalid_argument("A stat with the given id already exists");
    }
    if(index.to_index() != m_Enabled.size()) {
        throw std::invalid_argument("Currently, stats must be declared consecutively!");
    }

    for(const auto& old : m_MetaData) {
        if(meta.Name == old.Name) {
            throw std::invalid_argument("A stat with the given name already exists");
        }
    }

    m_Enabled.push_back(false);
    m_MetaData.emplace_back(std::move(meta));
    m_Statistics.emplace_back();
}

void StatisticsCollection::register_stat(const std::string& name, std::unique_ptr<Statistics> stat) {
    auto id = str_to_id(name, m_MetaData);
    if(m_Statistics.at(id.to_index()) && stat) {
        throw std::invalid_argument("Cannot register stat. Already registered!");
    }
    if(stat) {
        stat->setup(*this);
    }
    m_Statistics.at(id.to_index()) = std::move(stat);
    if(m_Statistics[id.to_index()]) {
        enable(id);
    } else {
        disable(id);
    }
}

const Statistics& StatisticsCollection::get_stat(const std::string& name) const {
    auto id = str_to_id(name, m_MetaData);
    auto ptr = m_Statistics.at(id.to_index()).get();
    if(!ptr)
        throw std::invalid_argument("No Statistics registered for the given name");
    return *ptr;
}

bool StatisticsCollection::is_enabled_by_name(const std::string& name) const {
    return is_enabled(str_to_id(name, m_MetaData));
}

void StatisticsCollection::declare_tag(tag_id_t index, std::string name) {
    if(index.to_index() < m_TagValues.size()) {
        throw std::invalid_argument("A tag with the given id already exists");
    }
    if(index.to_index() != m_TagValues.size()) {
        throw std::invalid_argument("Currently, tags must be declared consecutively!");
    }

    for(const auto& old : m_TagValues) {
        if(old.get_name() == name) {
            throw std::invalid_argument("A tag with the given name already exists");
        }
    }

    m_TagValues.emplace_back(TagContainer::create_full_container(name));
    m_TagLookup.emplace(m_TagValues.back().get_name(), m_TagValues.back());
}

TagContainer StatisticsCollection::get_tag_by_name(const std::string& name) const {
    return m_TagLookup.at(name);
}

void StatisticsCollection::provide_tags(const StatisticsCollection& other) {
    for(auto& other_tag : other.m_TagLookup) {
        auto result = m_TagLookup.insert(other_tag);
        // we cannot check this because of mutual registration.
        // TODO maybe we can verify that they point to the same memory
        /*if(!result.second) {
            spdlog::error("Cannot combine statistics collections because tag {} is a duplicate",
                          other_tag.second.get_name());
            throw std::runtime_error("Duplicate tag names are forbidden");
        }*/
    }
}

bool StatisticsCollection::has_stat(const std::string& name) const {
    return std::any_of(begin(m_MetaData), end(m_MetaData), [&](auto&& v){ return v.Name == name; });
}

#include "doctest.h"
#include "nlohmann/json.hpp"

namespace {
    struct MockStat : public Statistics {
        [[nodiscard]] std::unique_ptr<Statistics> clone() const override {
            assert(0);
            __builtin_unreachable();
        }
        void merge(const Statistics& other) override {
            assert(0);
            __builtin_unreachable();
        };

        [[nodiscard]] nlohmann::json to_json() const override {
            assert(0);
            __builtin_unreachable();
        };

        void record(long integer) override { LastValue = integer; }

        long LastValue;
    };
}

TEST_CASE("check errors for stats") {
    StatisticsCollection collection;
    collection.declare_stat(stat_id_t{0}, {"stat"});
    REQUIRE(collection.get_statistics_meta().size() == 1);

    // duplicate id
    DOCTEST_REQUIRE_THROWS(collection.declare_stat(stat_id_t{0}, {"other"}));
    REQUIRE(collection.get_statistics_meta().size() == 1);

    // duplicate name
    DOCTEST_REQUIRE_THROWS(collection.declare_stat(stat_id_t{1}, {"stat"}));
    REQUIRE(collection.get_statistics_meta().size() == 1);

    // enable or access without Statistics object
    DOCTEST_REQUIRE_THROWS(collection.enable(stat_id_t{0}));
    DOCTEST_REQUIRE_THROWS(collection.get_stat("stat"));

    // access undeclared name
    DOCTEST_REQUIRE_THROWS(collection.enable("unknown"));
    DOCTEST_REQUIRE_THROWS(collection.disable("unknown"));
    DOCTEST_REQUIRE_THROWS(collection.is_enabled_by_name("unknown"));
    DOCTEST_REQUIRE_THROWS(collection.register_stat("unknown", nullptr));
    DOCTEST_REQUIRE_THROWS(collection.get_stat("unknown"));
}

TEST_CASE("check errors for tags") {
    StatisticsCollection collection;
    collection.declare_tag(tag_id_t {0}, "tag");
    REQUIRE(collection.get_all_tags().size() == 1);

    // duplicate id
    DOCTEST_REQUIRE_THROWS(collection.declare_tag(tag_id_t{0}, "other"));
    REQUIRE(collection.get_all_tags().size() == 1);

    // duplicate name
    DOCTEST_REQUIRE_THROWS(collection.declare_tag(tag_id_t{1}, "tag"));
    REQUIRE(collection.get_all_tags().size() == 1);

    // access undeclared name
    DOCTEST_REQUIRE_THROWS(collection.get_tag_by_name("unknown"));
}

TEST_CASE("register stat") {
    StatisticsCollection collection;
    collection.declare_stat(stat_id_t{0}, {"stat"});

    CHECK(collection.has_stat("stat"));
    CHECK_FALSE(collection.has_stat("stat2"));

    auto stat = std::make_unique<MockStat>();
    collection.register_stat("stat", std::move(stat));
    CHECK(collection.is_enabled_by_name("stat"));

    DOCTEST_REQUIRE_THROWS(collection.register_stat("stat", std::make_unique<MockStat>()));
    collection.register_stat("stat", nullptr);
    CHECK_FALSE(collection.is_enabled_by_name("stat"));

    // now that we've reset it, we can safely set it again
    auto second = std::make_unique<MockStat>();
    auto* ptr = second.get();
    collection.register_stat("stat", std::move(second));
    CHECK(collection.is_enabled_by_name("stat"));

    CHECK(ptr == &collection.get_stat("stat"));
}

TEST_CASE("enable-disable") {
    StatisticsCollection collection;
    collection.declare_stat(stat_id_t{0}, {"stat"});
    collection.declare_stat(stat_id_t{1}, {"stat2"});

    collection.register_stat("stat", std::make_unique<MockStat>());
    REQUIRE(collection.is_enabled_by_name("stat"));

    // enable and disable using both named and indexed calls
    CHECK(collection.is_enabled(stat_id_t{0}));
    CHECK_FALSE(collection.is_enabled(stat_id_t{1}));

    collection.disable("stat");
    CHECK_FALSE(collection.is_enabled(stat_id_t{0}));
    CHECK_FALSE(collection.is_enabled(stat_id_t{1}));

    collection.enable("stat");
    CHECK(collection.is_enabled(stat_id_t{0}));
    CHECK_FALSE(collection.is_enabled(stat_id_t{1}));

    collection.disable(stat_id_t{0});
    CHECK_FALSE(collection.is_enabled(stat_id_t{0}));
    CHECK_FALSE(collection.is_enabled(stat_id_t{1}));

    collection.enable(stat_id_t{0});
    CHECK(collection.is_enabled(stat_id_t{0}));
    CHECK_FALSE(collection.is_enabled(stat_id_t{1}));
}

TEST_CASE("recording") {
    StatisticsCollection collection;
    collection.declare_stat(stat_id_t{0}, {"stat"});
    collection.register_stat("stat", std::make_unique<MockStat>());
    auto& stat = dynamic_cast<const MockStat&>(collection.get_stat("stat"));

    // direct recording
    collection.record(stat_id_t{0}, 5);
    CHECK(stat.LastValue == 5);

    // callable recording
    collection.record(stat_id_t{0}, [](){ return 8; });
    CHECK(stat.LastValue == 8);

    // don't call the callable for a disabled stat
    collection.disable(stat_id_t{0});
    collection.record(stat_id_t{0}, [](){ DOCTEST_FAIL("callable was called for disabled stat"); return 0; });
}

TEST_CASE("tag handling") {
    StatisticsCollection collection;
    collection.declare_tag(tag_id_t {0}, "tag");

    collection.set_tag(tag_id_t{0}, 25);

    CHECK(collection.get_tag_by_name("tag").get_value() == 25);

    // check that the container's value is really tied to the internal value
    TagContainer value = collection.get_tag_by_name("tag");
    CHECK(value.get_name() == "tag");

    collection.set_tag(tag_id_t{0}, 35);
    CHECK(value.get_value() == 35);
}

TEST_CASE("tag sharing") {
    StatisticsCollection collection;
    collection.declare_tag(tag_id_t {0}, "tag");

    StatisticsCollection other;
    other.declare_tag(tag_id_t {0}, "tag2");

    collection.provide_tags(other);

    // make sure that the system does not confuse the two
    other.set_tag(tag_id_t{0}, 25);
    collection.set_tag(tag_id_t{0}, 15);

    /// TODO we really should have scoped names here!
    auto foreign_tag = collection.get_tag_by_name("tag2");

    CHECK(foreign_tag.get_value() == 25);

    // check dupliate errors
    /*
    other.declare_tag(tag_id_t{1}, "tag");
    DOCTEST_CHECK_THROWS(collection.provide_tags(other));
     */
}

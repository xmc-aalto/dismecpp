// Copyright (c) 2021, Aalto University, developed by Erik Schultheis
// All rights reserved.
//
// SPDX-License-Identifier: MIT

#include "hyperparams.h"

using namespace dismec;

[[nodiscard]] auto HyperParameterBase::get_hyper_parameter(const std::string& name) const -> hyper_param_t {
    const hyper_param_ptr_t& hp = m_HyperParameters.at(name);
    hyper_param_t result;
    std::visit([&](auto&& current) -> void {
        result = current.Getter(this);
    }, hp);
    return result;
}

void HyperParameterBase::set_hyper_parameter(const std::string& name, long value) {
    hyper_param_ptr_t& hp = m_HyperParameters.at(name);
    std::get<HyperParamData<long>>(hp).Setter(this, value);
}

void HyperParameterBase::set_hyper_parameter(const std::string& name, double value) {
    hyper_param_ptr_t& hp = m_HyperParameters.at(name);
    std::get<HyperParamData<double>>(hp).Setter(this, value);
}

std::vector<std::string> HyperParameterBase::get_hyper_parameter_names() const {
    std::vector<std::string> result{};
    result.reserve(m_HyperParameters.size());
    for(auto& hp : m_HyperParameters) {
        result.push_back(hp.first);
    }
    return result;
}

void HyperParameters::set(const std::string &name, long value) {
    m_Values[name] = value;
}

void HyperParameters::set(const std::string &name, double value) {
    m_Values[name] = value;
}

auto HyperParameters::get(const std::string& name) const -> hyper_param_t {
    return m_Values.at(name);
}

void HyperParameters::apply(HyperParameterBase& target) const {
    for(auto& hp : m_Values) {
        std::visit([&](auto&& value) {
            target.set_hyper_parameter(hp.first, value);
        }, hp.second);
    }
}


#include "doctest.h"

namespace {
    /// A subclass of HyperParameterBase that declares on constrained HP `b` and one unconstrained
    /// HP `b`, to be used in the test cases.
    struct TestObject : public HyperParameterBase {
        double direct_hp = 0;
        long indirect_hp = 0;
        void set_b(long v) { indirect_hp = v; }
        long get_b() const { return indirect_hp; }

        TestObject() {
            declare_hyper_parameter("a", &TestObject::direct_hp);
            declare_hyper_parameter("b", &TestObject::get_b, &TestObject::set_b);
        }
    };

    /// A subclass of HyperParameterBase that declares subobject `so` of type \ref TestObject, for use in the
    /// test cases.
    struct NestedTestObject : public HyperParameterBase {
        NestedTestObject() {
            declare_sub_object("so", &NestedTestObject::sub);
        }

        TestObject sub;
    };
}

/*!
 * \test This test case verifies that direct hyperparamters work as expected for classes derived from
 * HyperParameterBase. We check that setting and getting correctly read/write the parameters, and that
 * type mismatches and name mismatches throw errors.
 */
TEST_CASE("HyperParameterBase") {
    TestObject object;
    SUBCASE("get and set") {
        object.set_hyper_parameter("a", 1.0);
        CHECK(object.direct_hp == 1.0);
        CHECK(std::get<double>(object.get_hyper_parameter("a")) == 1.0);

        object.set_hyper_parameter("b", 5l);
        CHECK(object.indirect_hp == 5);
        CHECK(std::get<long>(object.get_hyper_parameter("b")) == 5);
    }

    SUBCASE("type mismatch") {
        CHECK_THROWS(object.set_hyper_parameter("a", 3l));
        CHECK_THROWS(object.set_hyper_parameter("b", 3.5));
    }

    SUBCASE("name mismatch") {
        CHECK_THROWS(object.set_hyper_parameter("wrong", 5l));
        CHECK_THROWS(object.set_hyper_parameter("wrong", 2.0));
        CHECK_THROWS(object.get_hyper_parameter("wrong"));
    }

    SUBCASE("list hps") {
        auto hp_names = object.get_hyper_parameter_names();
        REQUIRE(hp_names.size() == 2);
        CHECK(((hp_names[0] == "a" && hp_names[1] == "b") || (hp_names[0] == "b" && hp_names[1] == "a")));
    }
}

/*!
 * \test This test verifies that registering a sub-object results in correct getters and setters for
 * the sub-object values.
 */
TEST_CASE("nested hyper parameter object") {
    NestedTestObject object;
    CHECK_THROWS(object.set_hyper_parameter("a", 1.0));

    object.set_hyper_parameter("so.a", 1.0);
    CHECK_THROWS(object.set_hyper_parameter("so.a", 5l));
    CHECK(object.sub.direct_hp == 1.0);
    CHECK(std::get<double>(object.get_hyper_parameter("so.a")) == 1.0);

    object.set_hyper_parameter("so.b", 5l);
    CHECK_THROWS(object.set_hyper_parameter("so.b", 1.0));
    CHECK(object.sub.indirect_hp == 5);
    CHECK(std::get<long>(object.get_hyper_parameter("so.b")) == 5);
}

/*!
 * \test This verifies that supplying a ptr-to-member to a different class than this
 * results in an exception.
 */
TEST_CASE("wrong subtype causes error") {
    struct InvalidRegistration : public HyperParameterBase {
        InvalidRegistration() {
            declare_hyper_parameter("a", &TestObject::direct_hp);
            //                               ^-- this is wrong, so an error should be thrown.
        }
        double direct_hp;
    };
    CHECK_THROWS(InvalidRegistration{});
}

/*!
 * \test Here we test the \ref HyperParameters objects. We check
 *   * throw on getting unknown name
 *   * read/write of values
 *   * applying to HyperParameterBase object with corresponding type and name checks.
 */
TEST_CASE("HyperParameters") {
    HyperParameters hps;
    // getting unknown parameter throws
    CHECK_THROWS(hps.get("test"));

    // setting and getting round-trip
    hps.set("b", 10l);
    CHECK(std::get<long>(hps.get("b")) == 10);

    // applying valid hyper-parameters to target. Works even for partial HPs
    TestObject target;
    hps.apply(target);
    CHECK(target.indirect_hp == 10);

    // applying invalid type hps
    hps.set("a", 10l);
    CHECK_THROWS(hps.apply(target));

    // updating type in HP collection works
    hps.set("a", 0.5);
    hps.apply(target);
    CHECK(target.direct_hp == 0.5);

    // applying breaks with additional hps
    hps.set("c", 0.5);
    CHECK_THROWS(hps.apply(target));
}

/*!
 * \test This test ensures that the \ref HyperParameters::apply function
 * can cope with nested objects.
 */
TEST_CASE("HyperParameters nested") {
    HyperParameters hps;
    hps.set("so.a", 1.0);
    hps.set("so.b", 5l);
    NestedTestObject object;
    hps.apply(object);
    CHECK(object.sub.direct_hp == 1.0);
    CHECK(object.sub.indirect_hp == 5);
}

// Copyright (c) 2021, Aalto University, developed by Erik Schultheis
// All rights reserved.
//
// SPDX-License-Identifier: MIT

#include "weighting.h"
#include <stdexcept>
#include <cmath>
#include "spdlog/spdlog.h"
#include "data/data.h"

PropensityModel::PropensityModel(const DatasetBase* data, double a, double b) :
    m_A(a), m_B(b), m_Data(data)
{
    if(!m_Data) {
        throw std::invalid_argument("data must not be nullptr");
    }
    m_C = (std::log(data->num_examples()) - 1) * std::pow(m_B + 1, m_A);
}

double PropensityModel::get_propensity(label_id_t label_id) const {

    double d = m_C * std::exp(-m_A * std::log(m_Data->num_positives(label_id) + m_B));
    return 1.0 / (1.0 + d);
}

double ConstantWeighting::get_positive_weight(label_id_t label_id) const {
    return m_PositiveCost;
}

double ConstantWeighting::get_negative_weight(label_id_t label_id) const {
    return m_NegativeCost;
}

ConstantWeighting::ConstantWeighting(double positive_cost, double negative_cost) :
    m_PositiveCost(positive_cost), m_NegativeCost(negative_cost) {
    if(positive_cost < 0 || negative_cost < 0) {
        throw std::invalid_argument("Negative cost");
    }
}

PropensityWeighting::PropensityWeighting(PropensityModel model) : m_Propensity(model) {

}

double PropensityWeighting::get_positive_weight(label_id_t label_id) const {
    return 2.0 / m_Propensity.get_propensity(label_id) - 1.0;
}

double PropensityWeighting::get_negative_weight(label_id_t label_id) const {
    return 1.0;
}

PropensityDownWeighting::PropensityDownWeighting(PropensityModel model) : m_Propensity(model) {

}

double PropensityDownWeighting::get_positive_weight(label_id_t label_id) const {
    return 1.0;
}

double PropensityDownWeighting::get_negative_weight(label_id_t label_id) const {
    // p / (2-p)
    double p = m_Propensity.get_propensity(label_id);
    return p / (2.0 - p);
}

CustomWeighting::CustomWeighting(DenseRealVector positive_weights, DenseRealVector negative_weights) :
    m_PositiveWeights(std::move(positive_weights)), m_NegativeWeights(std::move(negative_weights)) {
    // we do not know how many labels there are, but in any case the number should be the same for pos and neg
    if(m_PositiveWeights.size() != m_NegativeWeights.size()) {
        throw std::logic_error(fmt::format("Mismatched number of entries: {} in positive and {} in negative weights",
                                           m_PositiveWeights.size(), m_NegativeWeights.size()
                                           ));
    }
}

double CustomWeighting::get_positive_weight(label_id_t label_id) const {
    auto index = label_id.to_index();
    if(index < 0 || index >= m_PositiveWeights.size()) {
        throw std::logic_error(fmt::format("Trying to get positive weight for label {}, but only {} weights are known.",
                                           index, m_PositiveWeights.size()
        ));
    }
    return m_PositiveWeights.coeff(index);
}
double CustomWeighting::get_negative_weight(label_id_t label_id) const {
    auto index = label_id.to_index();
    if(index < 0 || index >= m_NegativeWeights.size()) {
        throw std::logic_error(fmt::format("Trying to get positive weight for label {}, but only {} weights are known.",
                                           index, m_NegativeWeights.size()
        ));
    }
    return m_NegativeWeights.coeff(index);
}

#include "doctest.h"

TEST_CASE("propensity calculation") {
    auto features = SparseFeatures(50, 50);
    auto labels = std::make_shared<BinaryLabelVector>(BinaryLabelVector::Zero(50));
    labels->coeffRef(0) = 1;
    BinaryData fake_data(features, labels);

    // 1 of 50
    PropensityModel pm{&fake_data, 0.55, 1.5};
    auto prop = pm.get_propensity(label_id_t{0});
    CHECK(prop == doctest::Approx(0.25562221863533147));

    for(int i = 0; i < 25; ++i) {
        labels->coeffRef(i) = 1;
    }

    // 25 / 50
    prop = pm.get_propensity(label_id_t{0});
    CHECK(prop == doctest::Approx(0.5571545100089221));
}

TEST_CASE("constant weighting") {
    ConstantWeighting cw(2.0, 5.0);
    CHECK(cw.get_positive_weight(label_id_t{0}) == 2.0);
    CHECK(cw.get_negative_weight(label_id_t{0}) == 5.0);
    CHECK(cw.get_positive_weight(label_id_t{10}) == 2.0);
    CHECK(cw.get_negative_weight(label_id_t{10}) == 5.0);

    CHECK_THROWS(ConstantWeighting(-1.0, 2.0));
    CHECK_THROWS(ConstantWeighting(1.0, -2.0));
}

TEST_CASE("prop weighting") {
    auto features = SparseFeatures(50, 50);
    auto labels = std::make_shared<BinaryLabelVector>(BinaryLabelVector::Zero(50));
    labels->coeffRef(0) = 1;
    BinaryData fake_data(features, labels);
    PropensityWeighting pw{PropensityModel(&fake_data, 0.55, 1.5)};
    CHECK(pw.get_positive_weight(label_id_t{0}) == doctest::Approx(2.0 / 0.25562221863533147 - 1.0));
    CHECK(pw.get_negative_weight(label_id_t{0}) == doctest::Approx(1.0));
}


// Copyright (c) 2021, Aalto University, developed by Erik Schultheis
// All rights reserved.
//
// SPDX-License-Identifier: MIT

#ifndef DISMEC_WEIGHTING_H
#define DISMEC_WEIGHTING_H

#include "matrix_types.h"
#include "data/types.h"

class PropensityModel {
public:
    explicit PropensityModel(const DatasetBase* data, double a=0.55, double b=1.5);
    [[nodiscard]] double get_propensity(label_id_t label_id) const;
private:
    const DatasetBase* m_Data;
    double m_A;
    double m_B;
    double m_C;
};

/*!
 * \brief Base class for label-based weighting schemes.
 * \details A label-based weighting scheme assigns each training example a weight
 * depending on whether the labels is present or absent. These weights can depend
 * on the label id, and are requested using the `get_positive_weight()` and
 * `get_negative_weight()` functions respectively.
 */
class WeightingScheme {
public:
    virtual ~WeightingScheme() = default;
    /// Gets the weight to use for all examples where the label `label_id` is present.
    [[nodiscard]] virtual double get_positive_weight(label_id_t label_id) const = 0;
    /// Gets the weight to use for all examples where the label `label_id` is absent.
    [[nodiscard]] virtual double get_negative_weight(label_id_t label_id) const = 0;
};

/*!
 * \brief Simple weighting scheme that assigns the same weighting to all `label_id`s.
 * \details This realization of a \ref WeightingScheme only returns two different values,
 * one if the label is there and another if it is not.
 */
class ConstantWeighting : public WeightingScheme {
public:
    ConstantWeighting(double positive_cost, double negative_cost);
    [[nodiscard]] double get_positive_weight(label_id_t label_id) const override;
    [[nodiscard]] double get_negative_weight(label_id_t label_id) const override;
private:
    double m_PositiveCost;  //!< Cost to use if the label is present, independent of the `label_id`.
    double m_NegativeCost;  //!< Cost to use if the label is absent, independent of the `label_id`.
};

class PropensityWeighting : public WeightingScheme  {
public:
    explicit PropensityWeighting(PropensityModel model);
    [[nodiscard]] double get_positive_weight(label_id_t label_id) const override;
    [[nodiscard]] double get_negative_weight(label_id_t label_id) const override;
private:
    PropensityModel m_Propensity;
};

class PropensityDownWeighting : public WeightingScheme  {
public:
    explicit PropensityDownWeighting(PropensityModel model);
    [[nodiscard]] double get_positive_weight(label_id_t label_id) const override;
    [[nodiscard]] double get_negative_weight(label_id_t label_id) const override;
private:
    PropensityModel m_Propensity;
};

class CustomWeighting : public WeightingScheme {
public:
    explicit CustomWeighting(DenseRealVector positive_weights, DenseRealVector negative_weights);

    [[nodiscard]] double get_positive_weight(label_id_t label_id) const override;
    [[nodiscard]] double get_negative_weight(label_id_t label_id) const override;
private:
    DenseRealVector m_PositiveWeights;
    DenseRealVector m_NegativeWeights;
};


#endif //DISMEC_WEIGHTING_H

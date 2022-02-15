// Copyright (c) 2021, Aalto University, developed by Erik Schultheis
// All rights reserved.
//
// SPDX-License-Identifier: MIT

#include "model/sparse.h"
#include "spdlog/spdlog.h"
#include "utils/eigen_generic.h"

using namespace model;

namespace {
    //! \todo we have this code twice now.
    long check_positive(long v, const char* error_msg) {
        if(v > 0) {
            return v;
        }
        throw std::invalid_argument(error_msg);
    }
}

SparseModel::SparseModel(long num_features, long num_labels) :
        SparseModel(num_features, PartialModelSpec{label_id_t{0}, num_labels, num_labels}){

}

SparseModel::SparseModel(long num_features, PartialModelSpec partial) :
    Model(partial),
    m_Weights(check_positive(partial.label_count, "Number of weight must be positive!")),
    m_NumFeatures(num_features) {
    for(auto& w : m_Weights) {
        w.resize(num_features);
    }
}


long SparseModel::num_features() const {
    return m_NumFeatures;
}

namespace {
    struct PredictVisitor {
        PredictVisitor(Eigen::Ref<PredictionMatrix> target, const std::vector<SparseRealVector>* weights) :
            Target(target), Weights(weights) {

        }
        void operator()(const GenericInMatrix::DenseColMajorRef& instances) {
            for(int i = 0; i < Weights->size(); ++i) {
                Target.col(i) = instances * (*Weights)[i];
            }
        }

        void operator()(const GenericInMatrix::DenseRowMajorRef& instances) {
            for(int i = 0; i < Weights->size(); ++i) {
                Target.col(i) = instances * (*Weights)[i];
            }
        }

        void operator()(const GenericInMatrix::SparseColMajorRef& instances) {
            for(int i = 0; i < Weights->size(); ++i) {
                Target.col(i) = instances * (*Weights)[i];
            }
        }

        void operator()(const GenericInMatrix::SparseRowMajorRef& instances) {
            types::SparseColMajor<real_t> copy = instances;
            for(int i = 0; i < Weights->size(); ++i) {
                Target.col(i) = copy * (*Weights)[i];
            }
        }

        Eigen::Ref<PredictionMatrix> Target;
        const std::vector<SparseRealVector>* Weights;
    };
}

void SparseModel::predict_scores_unchecked(const GenericInMatrix& instances, PredictionMatrixOut target) const {
    PredictVisitor visitor(target, &m_Weights);
    visit(visitor, instances);
}

void SparseModel::get_weights_for_label_unchecked(label_id_t label, Eigen::Ref<DenseRealVector> target) const {
    target = m_Weights.at(label.to_index());
}

namespace {
    struct SetWeightsVisitor {
        SetWeightsVisitor(label_id_t label, Eigen::SparseVector<real_t>* target) :
            Label(label), Target(target) {

        }

        void operator()(const GenericInVector::DenseRef& source) {
            Target->setZero();
            for(long i = 0; i < source.size(); ++i) {
                if(abs(source.coeff(i)) > 0) {
                    Target->insert(i) = source.coeff(i);
                }
            }
        }

        void operator()(const GenericInVector::SparseRef& source) {
            *Target = source;
        }

        label_id_t Label;
        Eigen::SparseVector<real_t>* Target;
    };
}

void SparseModel::set_weights_for_label_unchecked(label_id_t label, const GenericInVector& weights) {
    SetWeightsVisitor visitor(label, &m_Weights.at(label.to_index()));
    visit(visitor, weights);
}
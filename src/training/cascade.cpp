// Copyright (c) 2021, Aalto University, developed by Erik Schultheis
// All rights reserved.
//
// SPDX-License-Identifier: MIT

#include "cascade.h"
#include "objective/dense_and_sparse.h"
#include "solver/newton.h"
#include "data/data.h"
#include "data/transform.h"
#include "utils/conversion.h"
#include "postproc.h"
#include "initializer.h"
#include "model/sparse.h"

using namespace dismec;

namespace {
    class CombinedWeightInitializer : public init::WeightsInitializer {
    public:
        CombinedWeightInitializer( std::unique_ptr<init::WeightsInitializer> di,  std::unique_ptr<init::WeightsInitializer> si,
                                   long num_dense_features) :
            m_NumDenseFeatures(num_dense_features), m_DenseInit(std::move(di)), m_SparseInit(std::move(si)) {

        }
        void get_initial_weight(label_id_t label_id, Eigen::Ref<DenseRealVector> target,
                                objective::Objective& objective) override {
            m_DenseInit->get_initial_weight(label_id, target.head(m_NumDenseFeatures), objective);
            m_SparseInit->get_initial_weight(label_id, target.tail(target.size() - m_NumDenseFeatures), objective);
        }
    private:
        long m_NumDenseFeatures;
        std::unique_ptr<init::WeightsInitializer> m_DenseInit;
        std::unique_ptr<init::WeightsInitializer> m_SparseInit;
    };
}

std::shared_ptr<objective::Objective> CascadeTraining::make_objective() const {
    // we make a copy of the features, so they are in the local numa memory
    auto sp_ftr = m_SparseReplicator.get_local();
    auto ds_ftr = m_DenseReplicator.get_local();
    return objective::make_sp_dense_squared_hinge(ds_ftr, m_DenseReg,
                                                  sp_ftr, m_SparseReg);
}

std::unique_ptr<solvers::Minimizer> CascadeTraining::make_minimizer() const {
    auto minimizer = std::make_unique<solvers::NewtonWithLineSearch>(m_NumFeatures);
    m_NewtonSettings.apply(*minimizer);
    //minimizer->set_logger(get_logger());
    return minimizer;
}

void CascadeTraining::update_minimizer(solvers::Minimizer& base_minimizer, label_id_t label_id) const {
    auto* minimizer = dynamic_cast<solvers::NewtonWithLineSearch*>(&base_minimizer);
    if(!minimizer)
        throw std::logic_error("Could not cast minimizer to <NewtonWithLineSearch>");

    // adjust the epsilon parameter according to number of positives/number of negatives
    std::size_t num_pos = get_data().num_positives(label_id);
    double small_count = static_cast<double>(std::min(num_pos, get_data().num_examples() - num_pos));
    double epsilon_scale = std::max(small_count, 1.0) / static_cast<double>(get_data().num_examples());
    if(m_Shortlist) {
        std::size_t actual_num_pos = 0;
        std::size_t actual_num_neg = 0;
        const auto& shortlist = m_Shortlist->at(label_id.to_index());
        auto label_vec = get_data().get_labels(label_id);
        for(const auto& row : shortlist) {
            if(label_vec->coeff(row)) {
                ++actual_num_pos;
            } else {
                ++actual_num_neg;
            }
        }
        epsilon_scale = std::max( static_cast<double>(std::min(actual_num_neg, actual_num_pos)), 1.0 ) / static_cast<double>( actual_num_pos + actual_num_neg );
    }

    minimizer->set_epsilon(m_BaseEpsilon * epsilon_scale);
}

void CascadeTraining::update_objective(objective::Objective& base_objective, label_id_t label_id) const {
    auto* objective = dynamic_cast<objective::DenseAndSparseLinearBase*>(&base_objective);
    if(!objective)
        throw std::logic_error("Could not cast objective to <DenseAndSparseLinearBase>");

    if(m_Shortlist) {
        // TODO this causes several memory allocations
        const auto& shortlist = m_Shortlist->at(label_id.to_index());
        DenseFeatures shortlisted_dense = shortlist_features(m_DenseReplicator.get_local()->dense(),
                                                             shortlist);
        SparseFeatures shortlisted_sparse = shortlist_features(m_SparseReplicator.get_local()->sparse(),
                                                               shortlist);
        objective->update_features(shortlisted_dense, shortlisted_sparse);
        BinaryLabelVector& target_labels = objective->get_label_ref();
        target_labels.resize(ssize(shortlist));
        auto label_vec = get_data().get_labels(label_id);
        long target_id = 0;
        for(const auto& row : shortlist) {
            target_labels.coeffRef(target_id) = label_vec->coeff(row);
            ++target_id;
        }
        objective->update_costs(1.0, 1.0);
    } else {
        // we need to set the labels before we update the costs, since the label information is needed
        // to determine whether to apply the positive or the negative weighting
        get_data().get_labels(label_id, objective->get_label_ref());
    }
}

std::unique_ptr<init::WeightsInitializer> CascadeTraining::make_initializer() const {
    auto dense = m_DenseReplicator.get_local();
    auto sparse = m_SparseReplicator.get_local();

    auto dense_init = m_DenseInitStrategy->make_initializer(dense);
    auto sparse_init = m_SparseInitStrategy->make_initializer(sparse);
    return std::make_unique<CombinedWeightInitializer>(std::move(dense_init), std::move(sparse_init), dense->cols());

}

std::shared_ptr<model::Model> CascadeTraining::make_model(long num_features, model::PartialModelSpec spec) const {
    return std::make_shared<model::SparseModel>(num_features, spec);
}

std::unique_ptr<postproc::PostProcessor>
CascadeTraining::make_post_processor(const std::shared_ptr<objective::Objective>& objective) const {
    return m_PostProcessor->make_processor(objective);
}

TrainingStatsGatherer& CascadeTraining::get_statistics_gatherer() {
    return *m_StatsGather;
}

CascadeTraining::CascadeTraining(std::shared_ptr<const DatasetBase> tfidf_data,
                                 std::shared_ptr<const GenericFeatureMatrix> dense_data,
                                 HyperParameters hyper_params,
                                 std::shared_ptr<init::WeightInitializationStrategy> dense_init,
                                 real_t dense_reg,
                                 std::shared_ptr<init::WeightInitializationStrategy> sparse_init,
                                 real_t sparse_reg,
                                 std::shared_ptr<postproc::PostProcessFactory> post_proc,
                                 std::shared_ptr<TrainingStatsGatherer> gatherer,
                                 std::shared_ptr<const std::vector<std::vector<long>>> shortlist) :
    TrainingSpec(std::move(tfidf_data)),
    m_NewtonSettings( std::move(hyper_params) ),
    m_SparseReplicator(get_data().get_features() ),
    m_DenseReplicator(std::move(dense_data) ),
    m_Shortlist( std::move(shortlist) ),
    m_PostProcessor( std::move(post_proc) ),
    m_DenseInitStrategy( std::move(dense_init) ),
    m_SparseInitStrategy( std::move(sparse_init) ),
    m_StatsGather( std::move(gatherer) ),
    m_NumFeatures(m_SparseReplicator.get_local()->cols() + m_DenseReplicator.get_local()->cols()),
    m_DenseReg(dense_reg),
    m_SparseReg(sparse_reg)
    {

    // extract the base value of `epsilon` from the `hyper_params` object.
    m_BaseEpsilon = std::get<double>(m_NewtonSettings.get("epsilon"));
}


std::shared_ptr<TrainingSpec> dismec::create_cascade_training(
        std::shared_ptr<const DatasetBase> data,
        std::shared_ptr<const GenericFeatureMatrix> dense,
        std::shared_ptr<const std::vector<std::vector<long>>> shortlist,
        HyperParameters params,
        CascadeTrainingConfig config)
{
    if(!config.SparseInit)
        config.SparseInit = init::create_zero_initializer();
    if(!config.DenseInit)
        config.DenseInit = init::create_zero_initializer();
    if(!config.PostProcessing)
        config.PostProcessing = postproc::create_identity();
    return std::make_shared<CascadeTraining>(std::move(data),
                                             std::move(dense),
                                             std::move(params),
                                             std::move(config.DenseInit),
                                             config.DenseReg,
                                             std::move(config.SparseInit),
                                             config.SparseReg,
                                             std::move(config.PostProcessing),
                                             std::move(config.StatsGatherer),
                                             std::move(shortlist)
                                            );
}

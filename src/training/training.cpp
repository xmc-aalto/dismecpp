// Copyright (c) 2021, Aalto University, developed by Erik Schultheis
// All rights reserved.
//
// SPDX-License-Identifier: MIT

#include "training.h"

#include <utility>
#include "data/data.h"
#include "spdlog/fmt/chrono.h"
#include "model/submodel.h"
#include "parallel/runner.h"
#include "initializer.h"
#include "postproc.h"
#include "statistics.h"
#include "utils/eigen_generic.h"

TrainingTaskGenerator::TrainingTaskGenerator(std::shared_ptr<TrainingSpec> spec,
                                             label_id_t begin_label, label_id_t end_label) :
        m_TaskSpec(std::move(spec)),
        m_LabelRangeBegin(begin_label),
        m_LabelRangeEnd(end_label.to_index() > 0 ? end_label : label_id_t{m_TaskSpec->get_data().num_labels()})
{
    m_Results.resize(m_LabelRangeEnd - m_LabelRangeBegin);

    model::PartialModelSpec model_spec{m_LabelRangeBegin,
                                m_LabelRangeEnd - m_LabelRangeBegin,
                                static_cast<long>(m_TaskSpec->get_data().num_labels())};

    m_Model = m_TaskSpec->make_model(m_TaskSpec->get_data().num_features(), model_spec);
}

TrainingTaskGenerator::~TrainingTaskGenerator() = default;

void TrainingTaskGenerator::run_tasks(long begin, long end, thread_id_t thread_id) {
    for(long t = begin; t < end; ++t) {
        run_task(t, thread_id);
    }
}

void TrainingTaskGenerator::run_task(long task_id, thread_id_t thread_id) {
    label_id_t label_id = m_LabelRangeBegin + task_id;
    assert(0 <= label_id.to_index());
    assert(label_id.to_index() < m_TaskSpec->get_data().num_labels());
    m_Results.at(task_id) = train_label(label_id, thread_id);
}

solvers::MinimizationResult TrainingTaskGenerator::train_label(label_id_t label_id, thread_id_t thread_id) {
    m_ResultGatherers.at(thread_id.to_index())->start_label(label_id);

    // first, update the thread local objective and minimizer
    auto& objective = m_ThreadLocalObjective.at(thread_id.to_index());
    m_TaskSpec->update_objective(*objective, label_id);
    auto& minimizer = m_ThreadLocalMinimizer.at(thread_id.to_index());
    m_TaskSpec->update_minimizer(*minimizer, label_id);

    // get a reference to the thread-local weight buffer and initialize the weight.
    DenseRealVector& target = m_ThreadLocalWorkingVector.at(thread_id.to_index());
    m_ThreadLocalWeightInit.at(thread_id.to_index())->get_initial_weight(label_id, target, *objective);
    m_ResultGatherers.at(thread_id.to_index())->start_training(target);

    // run the minimizer and update the weights in the model
    auto result = minimizer->minimize(*objective, target);
    m_ResultGatherers.at(thread_id.to_index())->record_result(target, result);
    m_ThreadLocalPostProc.at(thread_id.to_index())->process(label_id, target, result);
    m_Model->set_weights_for_label(label_id, model::Model::WeightVectorIn{target});

    // some logging
    if(result.Outcome != solvers::MinimizerStatus::SUCCESS) {
        spdlog::warn("Minimization for label {:5} failed after {:4} iterations", label_id.to_index(), result.NumIters);
    }

    if(m_TaskSpec->get_logger()) {
        m_TaskSpec->get_logger()->info(
                "Thread {} finished minimization for label {:5} in {:4} iterations ({}) with loss {:6.3} -> {:6.3} and gradient {:6.3} -> {:6.3}.",
                thread_id.to_index(), label_id.to_index(), result.NumIters, result.Duration, result.InitialValue,
                result.FinalValue, result.InitialGrad, result.FinalGrad);
    }
    return result;
}

void TrainingTaskGenerator::prepare(long num_threads, long chunk_size) {
    m_ThreadLocalWorkingVector.resize(num_threads);
    m_ThreadLocalMinimizer.resize(num_threads);
    m_ThreadLocalObjective.resize(num_threads);
    m_ThreadLocalWeightInit.resize(num_threads);
    m_ThreadLocalPostProc.resize(num_threads);
    m_ResultGatherers.resize(num_threads);
}

void TrainingTaskGenerator::init_thread(thread_id_t thread_id)
{
    m_ThreadLocalWorkingVector.at(thread_id.to_index()) = DenseRealVector::Zero(m_TaskSpec->get_data().num_features());
    m_ThreadLocalMinimizer.at(thread_id.to_index()) = m_TaskSpec->make_minimizer();
    m_ThreadLocalObjective.at(thread_id.to_index()) = m_TaskSpec->make_objective();
    m_ThreadLocalWeightInit.at(thread_id.to_index()) = m_TaskSpec->make_initializer();
    m_ThreadLocalPostProc.at(thread_id.to_index()) = m_TaskSpec->make_post_processor(m_ThreadLocalObjective.at(thread_id.to_index()));
    m_ResultGatherers.at(thread_id.to_index()) = m_TaskSpec->get_statistics_gatherer().create_results_gatherer(thread_id, m_TaskSpec);

    m_TaskSpec->get_statistics_gatherer().setup_minimizer(thread_id, *m_ThreadLocalMinimizer.at(thread_id.to_index()));
    m_TaskSpec->get_statistics_gatherer().setup_initializer(thread_id, *m_ThreadLocalWeightInit.at(thread_id.to_index()));
    m_TaskSpec->get_statistics_gatherer().setup_objective(thread_id, *m_ThreadLocalObjective.at(thread_id.to_index()));
    m_TaskSpec->get_statistics_gatherer().setup_postproc(thread_id, *m_ThreadLocalPostProc.at(thread_id.to_index()));
}

void TrainingTaskGenerator::finalize() {
    m_ThreadLocalWorkingVector.clear();
    m_ThreadLocalMinimizer.clear();
    m_ThreadLocalObjective.clear();
    m_ThreadLocalWeightInit.clear();
    m_ThreadLocalPostProc.clear();

    m_TaskSpec->get_statistics_gatherer().finalize();
}

long TrainingTaskGenerator::num_tasks() const {
    return m_Results.size();
}

TrainingResult run_training(parallel::ParallelRunner& runner, std::shared_ptr<TrainingSpec> spec,
                            label_id_t begin_label, label_id_t end_label)
{
    auto task = TrainingTaskGenerator(std::move(spec), begin_label, end_label);
    auto result = runner.run(task);

    real_t total_loss = 0.0;
    real_t total_grad = 0.0;
    for(auto& r : task.get_results()) {
        total_loss += r.FinalValue;
        total_grad += r.FinalGrad;
    }

    auto model = task.get_model();
    // if training did time out, we need to adapt the resulting model to only declare the weight vectors
    // which have actually been calculated
    if(!result.IsFinished)
    {
        using SubWrapperType = model::SubModelWrapper<std::shared_ptr<model::Model>>;
        model = std::make_shared<SubWrapperType>(model, model->labels_begin(),
                                                 label_id_t{result.NextTask});
    }

    return {result.IsFinished, std::move(model), total_loss, total_grad};
}
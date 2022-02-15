// Copyright (c) 2021, Aalto University, developed by Erik Schultheis
// All rights reserved.
//
// SPDX-License-Identifier: MIT

#include "prediction/metrics.h"
#include "parallel/runner.h"
#include "prediction/prediction.h"
#include "model/model.h"
#include "data/data.h"
#include "data/transform.h"
#include "io/model-io.h"
#include "io/prediction.h"
#include "io/xmc.h"
#include "io/slice.h"
#include "CLI/CLI.hpp"
#include "spdlog/spdlog.h"

int main(int argc, const char** argv) {
    CLI::App app{"DiSMEC"};

    std::string problem_file;
    std::string model_file;
    std::string result_file;
    std::string labels_file;
    int threads = -1;
    bool one_based_index = false;
    bool request_normalize_instances = false;
    DatasetTransform request_transform = DatasetTransform::IDENTITY;
    int top_k = 5;
    double bias;

    app.add_option("problem-file", problem_file, "The file from which the data will be loaded.")->required()->check(CLI::ExistingFile);;
    app.add_option("model-file", model_file, "The file to which the model will be written.")->required()->check(CLI::ExistingFile);;
    app.add_option("result-file", result_file, "The file to which the predictions will be written.")->required();
    app.add_option("--label-file", labels_file, "For SLICE-type datasets, this specifies where the labels can be found")->check(CLI::ExistingFile);;
    app.add_option("--threads", threads, "Number of threads to use. -1 means auto-detect");
    app.add_flag("--xmc-one-based-index", one_based_index,
                 "If this flag is given, then we assume that the input dataset in xmc format"
                 " has one-based indexing, i.e. the first label and feature are at index 1  (as opposed to the usual 0)");
    app.add_option("--topk, --top-k", top_k, "Only the top k predictions will be saved. "
                                             "Set to -1 if you need all predictions. (Warning: This may result in very large files!)");
    auto augment_for_bias = app.add_flag("--augment-for-bias", bias,
                                         "If this flag is given, then all training examples will be augmented with an additional"
                                         "feature of value 1 or the specified value.")->default_val(1.0);
    app.add_flag("--normalize-instances", request_normalize_instances,
                 "If this flag is given, then the feature vectors of all instances are normalized to one.");

    app.add_option("--transform", request_transform, "Apply a transformation to the features of the dataset.")->default_str("identity")
        ->transform(CLI::Transformer(std::map<std::string, DatasetTransform>{
            {"identity",     DatasetTransform::IDENTITY},
            {"log-one-plus", DatasetTransform::LOG_ONE_PLUS},
            {"one-plus-log", DatasetTransform::ONE_PLUS_LOG}
        },CLI::ignore_case));

    int Verbose;
    app.add_flag("-v", Verbose);

    try {
        app.parse(argc, argv);
    } catch (const CLI::ParseError &e) {
        return app.exit(e);
    }

    spdlog::info("Reading data file from '{}'", problem_file);
    auto test_set = [&](){
        if(labels_file.empty()) {
            return read_xmc_dataset(problem_file,
                                    one_based_index ? io::IndexMode::ONE_BASED : io::IndexMode::ZERO_BASED);
        } else {
            return io::read_slice_dataset(problem_file, labels_file);
        }
    }();

    if(request_transform != DatasetTransform::IDENTITY) {
        if(Verbose >= 0)
            spdlog::info("Applying data transformation");
        transform_features(test_set, request_transform);
    }

    if(request_normalize_instances) {
        spdlog::info("Normalizing instances.");
        normalize_instances(test_set);
    }

    if(!augment_for_bias->empty()) {
        spdlog::info("Appending bias features with value {}", bias);
        augment_features_with_bias(test_set, bias);
    }

    parallel::ParallelRunner runner(threads);
    if(Verbose > 0)
        runner.set_logger(spdlog::default_logger());

    runner.set_chunk_size(1024);

    if(top_k > 0) {
        io::PartialModelLoader loader(model_file, io::PartialModelLoader::DEFAULT);

        spdlog::info("Calculating top-{} predictions", top_k);
        int wf_it  = 0;
        if(loader.num_weight_files() == 0) {
            spdlog::error("No weight files");
            return EXIT_FAILURE;
        }

        // generate a transpose of the label matrix
        std::vector<std::vector<long>> examples_to_labels(test_set.num_examples());
        for(label_id_t label{0}; label.to_index() < test_set.num_labels(); ++label) {
            for(auto example : test_set.get_label_instances(label)) {
                examples_to_labels[example].push_back(label.to_index());
            }
        }

        auto initial_model = loader.load_model(wf_it);
        spdlog::info("Using {} representation for model weights", initial_model->has_sparse_weights() ? "sparse" : "dense");

        TopKPredictionTaskGenerator task = TopKPredictionTaskGenerator(&test_set, initial_model, top_k);
        while(true) {
            ++wf_it;
            auto preload_weights = std::async(std::launch::async, [iter=wf_it, &loader]() {
                if(iter != loader.num_weight_files()) {
                    return loader.load_model(iter);
                } else {
                    return std::shared_ptr<Model>{};
                }
            });
            auto start_time = std::chrono::steady_clock::now();
            runner.run(task);
            spdlog::info("Finished prediction in {}ms", std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - start_time).count());
            if(wf_it == loader.num_weight_files()) {
                break;
            }
            task.update_model(preload_weights.get());
        }

        spdlog::info("Saving to '{}'", result_file);
        io::prediction::save_sparse_predictions(result_file,
                                                task.get_top_k_values(),
                                                task.get_top_k_indices());

        CalculateMetrics metrics{&examples_to_labels, &task.get_top_k_indices()};
        metrics.add_p_at_k(1);
        if(top_k >= 3) {
            metrics.add_p_at_k(3);
        }
        if(top_k >= 5) {
            metrics.add_p_at_k(5);
        }
        spdlog::info("Calculating metrics");
        runner.set_chunk_size(4096);
        runner.run(metrics);
        for(auto& result : metrics.get_metrics() ) {
            spdlog::info("{} = {:.3}", result.first, result.second);
        }

        auto& cm = task.get_confusion_matrix();
        std::int64_t tp = cm[TopKPredictionTaskGenerator::TRUE_POSITIVES];
        std::int64_t fp = cm[TopKPredictionTaskGenerator::FALSE_POSITIVES];
        std::int64_t tn = cm[TopKPredictionTaskGenerator::TRUE_NEGATIVES];
        std::int64_t fn = cm[TopKPredictionTaskGenerator::FALSE_NEGATIVES];
        std::int64_t total = tp + fp + tn + fn;

        spdlog::info("Confusion matrix is: \n"
                     "TP: {:15L}   FP: {:15L}\n"
                     "FN: {:15L}   TN: {:15L}", tp, fp, fn, tn);

        // calculates a percentage with decimals for extremely large integers.
        // we do the division still as integers, with two additional digits,
        // and only then convert to floating point.
        auto percentage = [](std::int64_t enumerator, std::int64_t denominator) {
            std::int64_t base_result = (std::int64_t{10'000} * enumerator) / denominator;
            return double(base_result) / 100.0;
        };

        spdlog::info("Accuracy:  {:.3}%", percentage(tp + tn,  total));
        spdlog::info("Precision: {:.3}%", percentage(tp, tp + fp));
        spdlog::info("Recall:    {:.3}%", percentage(tp, tp + fn));

    } else {
        spdlog::info("Reading model file from '{}'", model_file);
        auto model = io::load_model(model_file);

        spdlog::info("Calculating full predictions");
        PredictionTaskGenerator task(&test_set, model);
        runner.run(task);
        auto &predictions = task.get_predictions();
    }
}

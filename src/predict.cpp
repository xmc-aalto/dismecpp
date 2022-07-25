// Copyright (c) 2021, Aalto University, developed by Erik Schultheis
// All rights reserved.
//
// SPDX-License-Identifier: MIT

#include "prediction/metrics.h"
#include "parallel/runner.h"
#include "prediction/prediction.h"
#include "prediction/evaluate.h"
#include "model/model.h"
#include "data/data.h"
#include "data/transform.h"
#include "io/model-io.h"
#include "io/prediction.h"
#include "io/xmc.h"
#include "CLI/CLI.hpp"
#include "app.h"
#include "spdlog/spdlog.h"
#include "nlohmann/json.hpp"

using namespace dismec;


prediction::MacroMetricReporter* add_macro_metrics(prediction::EvaluateMetrics& metrics, int k) {
    auto* macro = metrics.add_macro_at_k(k);
    macro->add_coverage(0.0);
    macro->add_confusion_matrix();
    macro->add_precision(prediction::MacroMetricReporter::MACRO);
    macro->add_precision(prediction::MacroMetricReporter::MICRO);
    macro->add_recall(prediction::MacroMetricReporter::MACRO);
    macro->add_recall(prediction::MacroMetricReporter::MICRO);
    macro->add_f_measure(prediction::MacroMetricReporter::MACRO);
    macro->add_f_measure(prediction::MacroMetricReporter::MICRO);
    macro->add_accuracy(prediction::MacroMetricReporter::MICRO);
    macro->add_accuracy(prediction::MacroMetricReporter::MACRO);
    macro->add_balanced_accuracy(prediction::MacroMetricReporter::MICRO);
    macro->add_balanced_accuracy(prediction::MacroMetricReporter::MACRO);
    macro->add_specificity(prediction::MacroMetricReporter::MICRO);
    macro->add_specificity(prediction::MacroMetricReporter::MACRO);
    macro->add_informedness(prediction::MacroMetricReporter::MICRO);
    macro->add_informedness(prediction::MacroMetricReporter::MACRO);
    macro->add_markedness(prediction::MacroMetricReporter::MICRO);
    macro->add_markedness(prediction::MacroMetricReporter::MACRO);
    macro->add_fowlkes_mallows(prediction::MacroMetricReporter::MICRO);
    macro->add_fowlkes_mallows(prediction::MacroMetricReporter::MACRO);
    macro->add_negative_predictive_value(prediction::MacroMetricReporter::MICRO);
    macro->add_negative_predictive_value(prediction::MacroMetricReporter::MACRO);
    macro->add_matthews(prediction::MacroMetricReporter::MICRO);
    macro->add_matthews(prediction::MacroMetricReporter::MACRO);
    macro->add_positive_likelihood_ratio(prediction::MacroMetricReporter::MICRO);
    macro->add_positive_likelihood_ratio(prediction::MacroMetricReporter::MACRO);
    macro->add_negative_likelihood_ratio(prediction::MacroMetricReporter::MICRO);
    macro->add_negative_likelihood_ratio(prediction::MacroMetricReporter::MACRO);
    macro->add_diagnostic_odds_ratio(prediction::MacroMetricReporter::MICRO);
    macro->add_diagnostic_odds_ratio(prediction::MacroMetricReporter::MACRO);
    return macro;
};

void setup_metrics(prediction::EvaluateMetrics& metrics, int top_k) {
    metrics.add_precision_at_k(1);
    metrics.add_abandonment_at_k(1);
    metrics.add_dcg_at_k(1, false);
    metrics.add_dcg_at_k(1, true);

    add_macro_metrics(metrics, 1);

    if(top_k >= 3) {
        metrics.add_precision_at_k(3);
        metrics.add_abandonment_at_k(3);
        metrics.add_dcg_at_k(3, false);
        metrics.add_dcg_at_k(3, true);
        add_macro_metrics(metrics, 3);
    }
    if(top_k >= 5) {
        metrics.add_precision_at_k(5);
        metrics.add_abandonment_at_k(5);
        metrics.add_dcg_at_k(5, false);
        metrics.add_dcg_at_k(5, true);
        add_macro_metrics(metrics, 5);
    }
}


int main(int argc, const char** argv) {
    CLI::App app{"DiSMEC"};

    std::string problem_file;
    std::string model_file;
    std::string result_file;
    std::string labels_file;
    std::filesystem::path save_metrics;
    int threads = -1;
    int top_k = 5;

    DataProcessing DataProc;
    DataProc.setup_data_args(app);

    app.add_option("model-file", model_file, "The file from which the model will be read.")->required()->check(CLI::ExistingFile);;
    app.add_option("result-file", result_file, "The file to which the predictions will be written.")->required();
    app.add_option("--threads", threads, "Number of threads to use. -1 means auto-detect");
    app.add_option("--save-metrics", save_metrics, "Target file in which the metric values are saved");
    app.add_option("--topk, --top-k", top_k, "Only the top k predictions will be saved. "
                                             "Set to -1 if you need all predictions. (Warning: This may result in very large files!)");
    int Verbose = 0;
    app.add_flag("-v", Verbose);

    try {
        app.parse(argc, argv);
    } catch (const CLI::ParseError &e) {
        return app.exit(e);
    }

    auto test_set = DataProc.load(Verbose);

    parallel::ParallelRunner runner(threads);
    if(Verbose > 0)
        runner.set_logger(spdlog::default_logger());

    runner.set_chunk_size(PREDICTION_RUN_CHUNK_SIZE);

    if(top_k > 0) {
        io::PartialModelLoader loader(model_file, io::PartialModelLoader::DEFAULT);
        if(!loader.validate()) {
            return EXIT_FAILURE;
        }

        int wf_it  = 0;
        if(loader.num_weight_files() == 0) {
            spdlog::error("No weight files");
            return EXIT_FAILURE;
        }

        spdlog::info("Calculating top-{} predictions", top_k);

        // generate a transpose of the label matrix
        std::vector<std::vector<label_id_t>> examples_to_labels(test_set->num_examples());
        for(label_id_t label{0}; label.to_index() < test_set->num_labels(); ++label) {
            for(auto example : test_set->get_label_instances(label)) {
                examples_to_labels[example].push_back(label);
            }
        }

        auto initial_model = loader.load_model(wf_it);
        spdlog::info("Using {} representation for model weights", initial_model->has_sparse_weights() ? "sparse" : "dense");

        prediction::TopKPredictionTaskGenerator task(test_set.get(), initial_model, top_k);
        while(true) {
            ++wf_it;
            auto preload_weights = std::async(std::launch::async, [iter=wf_it, &loader]() {
                if(iter != loader.num_weight_files()) {
                    return loader.load_model(iter);
                } else {
                    return std::shared_ptr<dismec::model::Model>{};
                }
            });
            auto result = runner.run(task);
            if(!result.IsFinished) {
                spdlog::error("Something went wrong, prediction computation was not finished!");
                std::exit(1);
            }
            spdlog::info("Finished prediction in {}s", result.Duration.count());
            if(wf_it == loader.num_weight_files()) {
                break;
            }
            task.update_model(preload_weights.get());
        }

        spdlog::info("Saving to '{}'", result_file);
        io::prediction::save_sparse_predictions(result_file,
                                                task.get_top_k_values(),
                                                task.get_top_k_indices());

        prediction::EvaluateMetrics metrics{&examples_to_labels, &task.get_top_k_indices(), test_set->num_labels()};
        setup_metrics(metrics, top_k);

        spdlog::info("Calculating metrics");
        runner.set_chunk_size(PREDICTION_METRICS_CHUNK_SIZE);
        auto result_info = runner.run(metrics);
        spdlog::info("Calculated metrics in {}ms", std::chrono::duration_cast<std::chrono::milliseconds>(result_info.Duration).count());

        // sort thew results and present them
        std::vector<std::pair<std::string, double>> results =metrics.get_metrics();
        std::sort(results.begin(), results.end());

        for(const auto& [name, value] : results ) {
            std::cout << fmt::format("{:15} = {:.4}", name, value) << "\n";
        }

        if(!save_metrics.empty()) {
            nlohmann::json data;
            for(const auto& [name, value] : results ) {
                data[name] = value;
            }
            std::ofstream file(save_metrics);
            file << std::setw(4) << data;
        }

        const auto& cm = task.get_confusion_matrix();
        std::int64_t tp = cm[prediction::TopKPredictionTaskGenerator::TRUE_POSITIVES];
        std::int64_t fp = cm[prediction::TopKPredictionTaskGenerator::FALSE_POSITIVES];
        std::int64_t tn = cm[prediction::TopKPredictionTaskGenerator::TRUE_NEGATIVES];
        std::int64_t fn = cm[prediction::TopKPredictionTaskGenerator::FALSE_NEGATIVES];
        std::int64_t total = tp + fp + tn + fn;

        std::cout << fmt::format("Confusion matrix is: \n"
                     "TP: {:15L}   FP: {:15L}\n"
                     "FN: {:15L}   TN: {:15L}\n", tp, fp, fn, tn);

        // calculates a percentage with decimals for extremely large integers.
        // we do the division still as integers, with two additional digits,
        // and only then convert to floating point.
        auto percentage = [](std::int64_t enumerator, std::int64_t denominator) {
            std::int64_t base_result = (std::int64_t{10'000} * enumerator) / denominator;
            return double(base_result) / 100.0;
        };

        std::cout << fmt::format("Accuracy:     {:.3}%\n", percentage(tp + tn,  total));
        std::cout << fmt::format("Precision:    {:.3}%\n", percentage(tp, tp + fp));
        std::cout << fmt::format("Recall:       {:.3}%\n", percentage(tp, tp + fn));
        std::cout << fmt::format("F1:           {:.3}%\n", percentage(tp, tp + (fp + fn) / 2));

    } else {
        spdlog::error("Full predictions are currently not supported");
        exit(1);
        /*
        spdlog::info("Reading model file from '{}'", model_file);
        auto model = io::load_model(model_file);

        spdlog::info("Calculating full predictions");
        prediction::FullPredictionTaskGenerator task(test_set.get(), model);
        auto result = runner.run(task);
        if(!result.IsFinished) {
            spdlog::error("Something went wrong, prediction computation was not finished!");
            std::exit(1);
        }
        const auto& predictions = task.get_predictions();
        */

        // TODO fix handling of full predictions
    }
}

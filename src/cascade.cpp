// Copyright (c) 2021, Aalto University, developed by Erik Schultheis
// All rights reserved.
//
// SPDX-License-Identifier: MIT

#include "parallel/runner.h"
#include "io/model-io.h"
#include "io/xmc.h"
#include "io/slice.h"
#include "data/data.h"
#include "data/transform.h"
#include "training/training.h"
#include "training/weighting.h"
#include "training/postproc.h"
#include "training/initializer.h"
#include "training/statistics.h"
#include "CLI/CLI.hpp"
#include "spdlog/spdlog.h"
#include "io/numpy.h"
#include "io/common.h"
#include "spdlog/stopwatch.h"
#include <future>

using namespace dismec;

class TrainingProgram {
public:
    TrainingProgram();
    int run(int argc, const char** argv);
private:
    CLI::App app{"DiSMEC-Cascade"};

    // command line parameters
    //  source data
    void setup_source_cmdline();
    std::string TfIdfFile;
    std::string DenseFile;
    std::string ShortlistFile;

    // target model
    void setup_save_cmdline();
    std::filesystem::path ModelFile;
    io::model::SaveOption SaveOptions;

    // run range
    void setup_label_range();
    int FirstLabel = 0;
    int NumLabels = -1;
    bool ContinueRun = false;

    void parse_label_range();
    label_id_t LabelsBegin{0};
    label_id_t LabelsEnd{-1};

    CLI::Option* FirstLabelOpt;
    CLI::Option* NumLabelsOpt;

    // hyper params
    void setup_hyper_params();
    HyperParameters hps;

    // statistics
    std::string StatsOutFile = "stats.json";
    std::string StatsLevelFile = {};

    // normalization
    bool NormalizeSparse = false;
    bool NormalizeDense = false;
    DatasetTransform TransformSparse = DatasetTransform::IDENTITY;

    // initialization
    std::filesystem::path DenseWeightsFile;
    std::filesystem::path DenseBiasesFile;
    bool InitSparseMSI = false;
    bool InitDenseMSI = false;

    // regularization
    real_t RegScaleSparse = 1.0;
    real_t RegScaleDense = 1.0;

    // bias
    bool AugmentDenseWithBias = false;
    bool AugmentSparseWithBias = false;


    // others
    long NumThreads = -1;
    long Timeout = -1;
    long BatchSize = -1;
    std::filesystem::path ExportProcessedData;

    int Verbose = 0;

    // config setup helpers
    CascadeTrainingConfig make_config(const std::shared_ptr<MultiLabelData>& data, std::shared_ptr<const GenericFeatureMatrix> dense);
};

int main(int argc, const char** argv) {
    //openblas_set_num_threads(1);
    TrainingProgram program;
    program.run(argc, argv);
}

void TrainingProgram::setup_save_cmdline()
{
    SaveOptions.Format = io::model::WeightFormat::SPARSE_TXT;
    SaveOptions.Culling = 0.01;

    app.add_option("output,--model-file", ModelFile,
                   "The file to which the model will be written. Note that models are saved in multiple different files, so this"
                                  "just specifies the base name of the metadata file.")->required();

    app.add_option("--weight-culling", SaveOptions.Culling,
                   "When saving in a sparse format, any weight lower than this will be omitted.")->check(CLI::NonNegativeNumber);
                   ;
    app.add_option("--save-precision", SaveOptions.Precision,
                   "The number of digits to write for real numbers in text file format.")->check(CLI::NonNegativeNumber);

}

void TrainingProgram::setup_source_cmdline() {
    app.add_option("tfidf-file", TfIdfFile,
                   "The file from which the tfidf data will be loaded.")->required()->check(CLI::ExistingFile);
    app.add_option("dense-file", DenseFile,
                   "The file from which the dense data will be loaded.")->required()->check(CLI::ExistingFile);
    app.add_option("--shortlist", ShortlistFile,
                   "A file containing the shortlist of hard-negative instances for each label.")->check(CLI::ExistingFile);

}

void TrainingProgram::setup_label_range() {
    FirstLabelOpt = app.add_option("--first-label", FirstLabel,
                   "If you want to train only a subset of labels, this is the id of the first label to be trained."
                   "The subset of labels trained is `[first_label, first_label + num-labels)`")->check(CLI::NonNegativeNumber);
    NumLabelsOpt = app.add_option("--num-labels", NumLabels,
                   "If you want to train only a subset of labels, this is the total number of labels to be trained.")->check(CLI::NonNegativeNumber);
    app.add_flag("--continue", ContinueRun,
                 "If this flag is given, the new weights will be appended to the model "
                              "file, instead of overwriting it. You can use the --first-label option to explicitly specify "
                              "at which label to start. If omitted, training starts at the first label for which no "
                              "weight vector is known.");
}

void TrainingProgram::setup_hyper_params()
{
    // this needs to be set in all cases, because we need to adapt it dynamically and thus cannot rely on
    // default values
    hps.set("epsilon", 0.01);

    auto add_hyper_param_option = [&](const char* option, const char* name, const char* desc) {
        return app.add_option_function<double>(
                option,
                [this, name](double value) { hps.set(name, value); },
                desc)->group("hyper-parameters");
    };

    add_hyper_param_option("--epsilon", "epsilon",
                           "Tolerance for the minimizer. Will be adjusted by the number of positive/negative instances")
                           ->check(CLI::NonNegativeNumber);

    add_hyper_param_option("--alpha-pcg", "alpha-pcg",
                           "Interpolation parameter for preconditioning of CG optimization.")->check(CLI::Range(0.0, 1.0));

    add_hyper_param_option("--line-search-step-size", "search.step-size",
                           "Step size for the line search.")->check(CLI::NonNegativeNumber);

    add_hyper_param_option("--line-search-alpha", "search.alpha",
                           "Shrink factor for updating the line search step")->check(CLI::Range(0.0, 1.0));

    add_hyper_param_option("--line-search-eta", "search.eta",
                           "Acceptance criterion for the line search")->check(CLI::Range(0.0, 1.0));
    add_hyper_param_option("--cg-epsilon", "cg.epsilon",
                           "Stopping criterion for the CG solver")->check(CLI::PositiveNumber);

    app.add_option_function<long>(
            "--max-steps",
            [this](long value) { hps.set("max-steps", value); },
            "Maximum number of newton steps.")->check(CLI::PositiveNumber)->group("hyper-parameters");

    app.add_option_function<long>(
            "--line-search-max-steps",
            [this](long value) { hps.set("search.max-steps", value); },
            "Maximum number of line search steps.")->check(CLI::PositiveNumber)->group("hyper-parameters");
}

void TrainingProgram::parse_label_range()
{
    // continue with automatic first label selection
    if(ContinueRun)
    {
        io::PartialModelSaver saver(ModelFile, SaveOptions, true);
        if(FirstLabelOpt->count() == 0)
        {
            auto missing = saver.get_missing_weights();
            spdlog::info("Model is missing weight vectors {} to {}.", missing.first.to_index(), missing.second.to_index() - 1);
            LabelsBegin = missing.first;
            LabelsEnd = missing.second;
            if (NumLabelsOpt->count() > 0) {
                if (LabelsEnd - LabelsBegin >= NumLabels) {
                    LabelsEnd = LabelsBegin + NumLabels;
                } else {
                    spdlog::warn("Number of labels to train was specified as {}, but only {} labels will be trained",
                                 NumLabels, LabelsEnd - LabelsBegin);
                }
            }
            return;
        } else {
            // user has given us a label from which to start.
            LabelsBegin = label_id_t{FirstLabel};
            if (NumLabelsOpt->count() > 0) {
                LabelsBegin = LabelsBegin + NumLabels;
                // and a label count. Then we need to check is this is valid
                if(saver.any_weight_vector_for_interval(LabelsBegin, LabelsEnd)) {
                    spdlog::error("Specified continuation of training weight vectors for labels {}-{}, "
                                  "which overlaps with existing weight vectors.", LabelsBegin.to_index(), LabelsEnd.to_index()-1);
                    exit(EXIT_FAILURE);
                }
                return;
            }
            LabelsEnd = label_id_t{saver.num_labels()};
            return;
        }
    }

    // OK, we are not continuing a run.

    if(FirstLabelOpt->count()) {
        LabelsBegin = label_id_t{FirstLabel};
    } else {
        LabelsBegin = label_id_t{0};
    }

    if (NumLabelsOpt->count() > 0) {
        LabelsEnd = LabelsBegin + NumLabels;
    } else {
        LabelsEnd = label_id_t{-1};
    }
}

TrainingProgram::TrainingProgram() {
    setup_source_cmdline();
    setup_save_cmdline();
    setup_label_range();
    setup_hyper_params();

    app.add_option("--threads", NumThreads, "Number of threads to use. -1 means auto-detect");
    app.add_option("--batch-size", BatchSize, "If this is given, training is split into batches "
                                              "and results are written to disk after each batch.");
    app.add_option("--timeout", Timeout, "No new training tasks will be started after this time. "
                                         "This can be used e.g. on a cluster system to ensure that the training finishes properly "
                                         "even if not all work could be done in the allotted time.")
    ->transform(CLI::AsNumberWithUnit(std::map<std::string, float>{{"ms", 1},
                                                               {"s", 1'000}, {"sec", 1'000},
                                                               {"m", 60'000}, {"min", 60'000},
                                                               {"h", 60*60'000}},
                                      CLI::AsNumberWithUnit::UNIT_REQUIRED, "TIME"));

    app.add_option("--record-stats", StatsLevelFile,
                   "Record some statistics and save to file. The argument is a json file which describes which statistics are gathered.")
                   ->check(CLI::ExistingFile);
    app.add_option("--stats-file", StatsOutFile, "Target file for recorded statistics");
    app.add_option("--init-dense-weights", DenseWeightsFile, "File from which the initial weights for the dense part will be loaded.")->check(CLI::ExistingFile);
    app.add_option("--init-dense-biases", DenseBiasesFile, "File from which the initial biases for the dense part will be loaded.")->check(CLI::ExistingFile);
    app.add_flag("--init-sparse-msi", InitSparseMSI, "If this flag is given, then the sparse part will use mean-separating initialization.");
    app.add_flag("--init-dense-msi", InitDenseMSI, "If this flag is given, then the dense part will use mean-separating initialization.");

    app.add_option("--sparse-reg-scale", RegScaleSparse, "Scaling factor for the sparse-part regularizer")->check(CLI::NonNegativeNumber);
    app.add_option("--dense-reg-scale", RegScaleDense, "Scaling factor for the dense-part regularizer")->check(CLI::NonNegativeNumber);
    app.add_flag("--normalize-dense", NormalizeDense, "Normalize the dense part of the feature matrix");
    app.add_flag("--normalize-sparse", NormalizeSparse, "Normalize the sparse part of the feature matrix");
    app.add_option("--transform-sparse", TransformSparse, "Apply a transformation to the sparse features.")->default_str("identity")
        ->transform(CLI::Transformer(std::map<std::string, DatasetTransform>{
            {"identity",     DatasetTransform::IDENTITY},
            {"log-one-plus", DatasetTransform::LOG_ONE_PLUS},
            {"one-plus-log", DatasetTransform::ONE_PLUS_LOG},
            {"sqrt",         DatasetTransform::SQRT}
        },CLI::ignore_case));

    app.add_flag("--augment-dense-bias", AugmentDenseWithBias, "Add an additional feature column to the dense matrix with values one.");
    app.add_flag("--augment-sparse-bias", AugmentSparseWithBias, "Add an additional feature column to the sparse matrix with values one.");

    app.add_option("--export-dataset", ExportProcessedData,
                   "Exports the preprocessed dataset to the given file.");
    app.add_flag("-v,-q{-1}", Verbose);
}

MultiLabelData join_data(const std::shared_ptr<MultiLabelData>& data,
                      std::shared_ptr<const GenericFeatureMatrix> dense_data) {

    const SparseFeatures& sparse = data->get_features()->sparse();
    const DenseFeatures& dense = dense_data->dense();

    SparseFeatures new_sparse(data->num_examples(), data->num_features() + dense.cols());
    new_sparse.reserve(sparse.nonZeros() + dense.size());
    for (int k=0; k < data->num_examples(); ++k) {
        new_sparse.startVec(k);
        for (DenseFeatures::InnerIterator it(dense, k); it; ++it) {
            new_sparse.insertBack(it.row(), it.col()) = it.value();
        }
        for (SparseFeatures::InnerIterator it(sparse, k); it; ++it) {
            new_sparse.insertBack(it.row(), it.col() + dense.cols()) = it.value();
        }
    }
    new_sparse.finalize();
    return {new_sparse, data->all_labels()};
}

CascadeTrainingConfig TrainingProgram::make_config(const std::shared_ptr<MultiLabelData>& data,
                                                   std::shared_ptr<const GenericFeatureMatrix> dense) {
    CascadeTrainingConfig config;

    if(InitSparseMSI)
        config.SparseInit = init::create_feature_mean_initializer(data, 1.0, -2.0);

    if(!DenseWeightsFile.empty()) {
        if(InitDenseMSI) {
            spdlog::error("Cannot use MSI and pretrained weights at the same time!");
            exit(EXIT_FAILURE);
        }
        if(DenseBiasesFile.empty()) {
            config.DenseInit = init::create_numpy_initializer(DenseWeightsFile, {});
        } else {
            config.DenseInit = init::create_numpy_initializer(DenseWeightsFile, DenseBiasesFile);
        }
    } else if(InitDenseMSI) {
        auto dense_ds = std::make_shared<MultiLabelData>(dense->dense(), data->all_labels());
        config.DenseInit = init::create_feature_mean_initializer(dense_ds, 1.0, -2.0);
    }

    config.StatsGatherer = std::make_shared<TrainingStatsGatherer>(StatsLevelFile, StatsOutFile);

    return config;
}

int TrainingProgram::run(int argc, const char** argv)
{
    try {
        app.parse(argc, argv);
    } catch (const CLI::ParseError &e) {
        return app.exit(e);
    }

    // check validity of save location
    auto parent = std::filesystem::absolute(ModelFile).parent_path();
    if(!std::filesystem::exists(parent)) {
        spdlog::warn("Save directory '{}' does not exist. Trying to create it.", parent.c_str());
        std::filesystem::create_directories(parent);
        if(!std::filesystem::exists(parent)) {
            spdlog::error("Could not create directory -- exiting.");
            return EXIT_FAILURE;
        }
    }

    // TODO At this point, we know that the target directory exists, but not whether it is writeable.
    // still, it's a start.


    auto start_time = std::chrono::steady_clock::now();
    auto timeout_time = start_time + std::chrono::milliseconds(Timeout);

    spdlog::info("Loading training data from file '{}'", TfIdfFile);
    auto data = std::make_shared<MultiLabelData>([&]() {
        return read_xmc_dataset(TfIdfFile, io::IndexMode::ZERO_BASED);
    } ());

    if(TransformSparse != DatasetTransform::IDENTITY) {
        spdlog::info("Applying data transformation");
        transform_features(*data, TransformSparse);
    }

    if(NormalizeSparse) {
        spdlog::stopwatch timer;
        normalize_instances(*data);
        spdlog::info("Normalized sparse features in {:.3} seconds.", timer);
    }
    if(AugmentSparseWithBias) {
        spdlog::stopwatch timer;
        augment_features_with_bias(*data);
        spdlog::info("Added bias column to sparse features in {:.3} seconds.", timer);
    }
    //auto permute = sort_features_by_frequency(*data);

    auto dense_data = std::make_shared<GenericFeatureMatrix>(io::load_matrix_from_npy(DenseFile));
    if(NormalizeDense) {
        spdlog::stopwatch timer;
        normalize_instances(dense_data->dense());
        spdlog::info("Normalized dense features in {:.3} seconds.", timer);
    }

    if(AugmentDenseWithBias) {
        spdlog::stopwatch timer;
        augment_features_with_bias(dense_data->dense());
        spdlog::info("Added bias column to dense features in {:.3} seconds.", timer);
    }
    // dense_data->dense().setZero();

    if(!ExportProcessedData.empty()) {
        spdlog::stopwatch timer;
        auto exported = join_data(data, dense_data);
        io::save_xmc_dataset(ExportProcessedData, exported, 6);
        spdlog::info("Saved preprocessed data to {} in {:.3} seconds", ExportProcessedData.string(), timer);
        exit(0);
    }

    std::shared_ptr<const std::vector<std::vector<long>>> shortlist;
    if(!ShortlistFile.empty()) {
        auto stream = std::fstream(ShortlistFile, std::fstream::in);
        auto result = io::read_binary_matrix_as_lil(stream);
        if(result.NumCols != data->num_labels()) {
            spdlog::error("Mismatch between number of labels in shortlist {} and in dataset {}",
                          result.NumCols, data->num_labels());
            exit(1);
        }
        if(result.NumRows != data->num_examples()) {
            spdlog::error("Mismatch between number of examples in shortlist {} and in dataset {}",
                          result.NumRows, data->num_examples());
            exit(1);
        }

        shortlist = std::make_shared<std::vector<std::vector<long>>>(std::move(result.NonZeros));
    }

    parse_label_range();

    auto runner = parallel::ParallelRunner(NumThreads);
    if(Verbose > 0)
        runner.set_logger(spdlog::default_logger());

    auto config = make_config(data, dense_data);

    std::shared_ptr<postproc::PostProcessFactory> post_proc = postproc::create_culling(SaveOptions.Culling);
    SaveOptions.Culling = 1e-10;

    if(BatchSize <= 0) {
        BatchSize = data->num_labels();
    }

    if(Verbose >= 0) {
        spdlog::info("handled preprocessing in {} seconds",
                     std::chrono::duration_cast<std::chrono::seconds>(std::chrono::steady_clock::now() - start_time).count() );
    }

    // batched training
    spdlog::info("Start training");
    io::PartialModelSaver saver(ModelFile, SaveOptions, ContinueRun);
    label_id_t first_label = LabelsBegin;
    if(LabelsEnd == label_id_t{-1}) {
        LabelsEnd = label_id_t{data->num_labels()};
    }
    label_id_t next_label = std::min(LabelsEnd, first_label + BatchSize);
    std::future<io::model::WeightFileEntry> saving;

    config.PostProcessing = post_proc;
    config.DenseReg = RegScaleDense;
    config.SparseReg = RegScaleSparse;

    while(true) {
        spdlog::info("Starting batch {} - {}", first_label.to_index(), next_label.to_index());

        // update time limit to respect remaining time
        runner.set_time_limit(std::chrono::duration_cast<std::chrono::milliseconds>(timeout_time - std::chrono::steady_clock::now()));

        std::shared_ptr<TrainingSpec> train_spec = create_cascade_training(data, dense_data, shortlist, hps, config);
        if(Verbose >= 0) {
            train_spec->set_logger(spdlog::default_logger());
        }
        auto result = run_training(runner, train_spec,
                                   first_label, next_label);

        /* do async saving. This has some advantages and some drawbacks:
            + all the i/o latency will be interleaved with actual new computation and we don't waste much time
              in this essentially non-parallel code
            - we may overcommit the processor. If run_training uses all cores, then we will spawn an additional thread
              here
            - increased memory consumption. Instead of 1 model, we need to keep 2 in memory at the same time: The one
              that is currently worked on and the one that is still being saved.
         */
        // make sure we don't interleave saving, as we don't do any locking in `saver`. Also, throw any exception
        // that happened during the saving
        if(saving.valid()) {
            saving.get();
            // saving weights has finished, we can update the meta data
            saver.update_meta_file();
        }

        saving = saver.add_model(result.Model);

        first_label = next_label;
        if(first_label == LabelsEnd) {
            // wait for the last saving process to finish
            saving.get();
            saver.update_meta_file();
            break;
        }
        next_label = std::min(LabelsEnd, first_label + BatchSize);
        // special case -- if the remaining labels are less than half a batch, we add them to this
        // batch
        if(next_label + BatchSize/2 > LabelsEnd) {
            next_label = LabelsEnd;
        }
    }

    spdlog::info("program finished after {} seconds", std::chrono::duration_cast<std::chrono::seconds>(std::chrono::steady_clock::now() - start_time).count() );

    return EXIT_SUCCESS;
}

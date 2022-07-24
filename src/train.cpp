// Copyright (c) 2021, Aalto University, developed by Erik Schultheis
// All rights reserved.
//
// SPDX-License-Identifier: MIT

#include "parallel/runner.h"
#include "io/model-io.h"
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
#include "app.h"
#include <future>

using namespace dismec;

// extern "C" void openblas_set_num_threads(int num_threads);


class TrainingProgram {
public:
    TrainingProgram();
    int run(int argc, const char** argv);
private:
    CLI::App app{"DiSMEC"};

    // command line parameters
    //  source data
    void setup_source_cmdline();
    bool ReorderFeatures = false;

    DataProcessing DataProc;

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
    std::string WeightingMode;
    double PropA = 0.55;
    double PropB = 1.5;
    std::string WeightingPosFile;
    std::string WeightingNegFile;

    // regularization
    void setup_regularization();

    // Pre-trained model
    std::filesystem::path SourceModel;
    CLI::Option* PreTrainedOpt;

    std::string InitMode;
    std::optional<real_t> BiasInitValue;
    real_t MSI_PFac = 1;
    real_t MSI_NFac = -2;
    int InitMaxPos = 1;

    RegularizerType Regularizer = RegularizerType::REG_L2;
    real_t RegScale = 1.0;
    bool RegBias = false;

    real_t Sparsify = -1;

    LossType Loss = LossType::SQUARED_HINGE;

    // statistics
    std::string StatsOutFile = "stats.json";
    std::string StatsLevelFile = {};


    // others
    long NumThreads = -1;
    long Timeout = -1;
    long BatchSize = -1;

    int Verbose = 0;

    // config setup helpers
    DismecTrainingConfig make_config(const std::shared_ptr<MultiLabelData>& data);
};

int main(int argc, const char** argv) {
    //openblas_set_num_threads(1);
    TrainingProgram program;
    program.run(argc, argv);
}

void TrainingProgram::setup_save_cmdline()
{
    SaveOptions.Format = io::model::WeightFormat::DENSE_TXT;

    app.add_option("output,--model-file", ModelFile,
                   "The file to which the model will be written. Note that models are saved in multiple different files, so this"
                                  "just specifies the base name of the metadata file.")->required();

    // save format flags
    auto* dense_txt_flag = app.add_flag("--save-dense-txt", [&](std::size_t){ SaveOptions.Format = io::WeightFormat::DENSE_TXT; },
                 "Save dense weights in a human-readable text format")->take_last();
    auto* dense_npy_flag = app.add_flag("--save-dense-npy", [&](std::size_t){ SaveOptions.Format = io::WeightFormat::DENSE_NPY; },
                 "Save dense weights in a npy file")->take_last();
    auto* sparse_flag = app.add_flag("--save-sparse-txt", [&](std::size_t){ SaveOptions.Format = io::WeightFormat::SPARSE_TXT; },
                 "Save sparse weights in a human-readable text format. Sparsity can be adjusted using the --weight-culling option")->take_last();

    dense_npy_flag->excludes(dense_txt_flag, sparse_flag);
    dense_txt_flag->excludes(dense_npy_flag, sparse_flag);
    sparse_flag->excludes(dense_txt_flag, dense_npy_flag);

    app.add_option("--weight-culling", SaveOptions.Culling,
                   "When saving in a sparse format, any weight lower than this will be omitted.")->needs(sparse_flag)->check(CLI::NonNegativeNumber);

    app.add_option("--save-precision", SaveOptions.Precision,
                   "The number of digits to write for real numbers in text file format.")->check(CLI::NonNegativeNumber)->excludes(dense_npy_flag);

}

void TrainingProgram::setup_source_cmdline() {
    DataProc.setup_data_args(app);
    app.add_flag("--reorder-features", ReorderFeatures,
                 "If this flag is given, then the feature columns are sorted by the frequency before training. "
                 "This can lead to fast computations in case the number of features is very large and their frequencies imbalanced, "
                 "because it may improve data locality.");
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


void TrainingProgram::setup_regularization() {
    app.add_option("--regularizer", Regularizer, "The weight regularizer")->default_str("l2")
        ->transform(CLI::Transformer(std::map<std::string, RegularizerType>{{"l2", RegularizerType::REG_L2},
                                                                            {"l1", RegularizerType::REG_L1},
                                                                            {"l1-relaxed", RegularizerType::REG_L1_RELAXED},
                                                                            {"huber", RegularizerType::REG_HUBER},
                                                                            {"elastic-50-50", RegularizerType::REG_ELASTIC_50_50},
                                                                            {"elastic-90-10", RegularizerType::REG_ELASTIC_90_10}
        },CLI::ignore_case));
    app.add_option("--reg-scale", RegScale, "Scaling factor for the regularizer")->check(CLI::NonNegativeNumber);
    app.add_flag("--reg-bias", RegBias, "Include bias in regularization")->default_val(false);
}


TrainingProgram::TrainingProgram() {
    setup_source_cmdline();
    setup_save_cmdline();
    setup_label_range();
    setup_hyper_params();
    setup_regularization();

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

    auto WMOpt = app.add_option("--weighting-mode", WeightingMode,
                   "Determines the re-weighting algorithm used to address missing labels.");
    app.add_option("--propensity-a", PropA,
                   "Parameter a for the propensity model when using propensity based weighting")->needs(WMOpt);
    app.add_option("--propensity-b", PropB,
                   "Parameter b for the propensity model when using propensity based weighting")->needs(WMOpt);
    app.add_option("--weighting-pos-file", WeightingPosFile,
                   "File (npz or txt) that contains the weights for the positive instances for each label.")->needs(WMOpt);
    app.add_option("--weighting-neg-file", WeightingNegFile,
                   "File (npz or txt) that contains the weights for the negative instances for each label.")->needs(WMOpt);



    PreTrainedOpt = app.add_option("--pretrained", SourceModel, "The model file which will be "
                                                                "used to initialize the weights.");
    PreTrainedOpt->check(CLI::ExistingFile);

    app.add_option("--loss", Loss, "The loss function")->default_str("squared-hinge")
        ->transform(CLI::Transformer(std::map<std::string, LossType>{{"squared-hinge", LossType::SQUARED_HINGE},
                                                                         {"logistic", LossType::LOGISTIC},
                                                                         {"huber-hinge", LossType::HUBER_HINGE},
                                                                         {"hinge", LossType::HINGE},
                                                                         },CLI::ignore_case));

    app.add_option("--sparsify", Sparsify, "Feedback-driven sparsification. Specify the maximum amount (in %) up to which the binary loss "
                                           "is allowed to increase.");

    app.add_option("--init-mode", InitMode, "How to initialize the weight vectors")
        ->check(CLI::IsMember({"zero", "mean", "bias", "msi", "multi-pos", "ova-primal"}));
    app.add_option("--bias-init-value", BiasInitValue, "The value that is assigned to the bias weight for bias-init.");
    app.add_option("--msi-pos", MSI_PFac, "Positive target for msi init");
    app.add_option("--msi-neg", MSI_NFac, "Negative target for msi init");
    app.add_option("--max-num-pos", InitMaxPos, "Number of positives to consider for `multi-pos` initialization")->check(CLI::NonNegativeNumber);

    app.add_option("--record-stats", StatsLevelFile,
                   "Record some statistics and save to file. The argument is a json file which describes which statistics are gathered.")
                   ->check(CLI::ExistingFile);
    app.add_option("--stats-file", StatsOutFile, "Target file for recorded statistics");

    app.add_flag("-v,-q{-1}", Verbose);
}

DismecTrainingConfig TrainingProgram::make_config(const std::shared_ptr<MultiLabelData>& data) {
    DismecTrainingConfig config;

    // Positive / Negative weighting
    if(WeightingMode == "2pm1") {
        config.Weighting = std::make_shared<PropensityWeighting>(PropensityModel(data.get(), PropA, PropB));
    } else if(WeightingMode == "p2mp") {
        config.Weighting = std::make_shared<PropensityDownWeighting>(PropensityModel(data.get(), PropA, PropB));
    }else if(WeightingMode == "from-file") {
        auto load_vec = [&](std::string source){
            DenseRealVector wgt = DenseRealVector::Ones(data->num_labels());
            if(!source.empty()) {
                std::fstream file(source, std::fstream::in);
                if(!file.is_open()) {
                    THROW_ERROR("Could not open file {}", source);
                }
                if(io::is_npy(file)) {
                    auto header = io::parse_npy_header(*file.rdbuf());
                    if(header.DataType != io::data_type_string<real_t>()) {
                        THROW_ERROR("Unsupported data type {}", header.DataType);
                    }
                    if(header.Rows != 1 && header.Cols != 1) {
                        THROW_ERROR("Expected a vector for weighting data");
                    }
                    io::binary_load(*file.rdbuf(), wgt.data(), wgt.data() + header.Rows * header.Cols);
                } else {
                    io::read_vector_from_text(file, wgt);
                }
            }
            return wgt;
        };
        config.Weighting = std::make_shared<CustomWeighting>(load_vec(WeightingPosFile),
                                                             load_vec(WeightingNegFile));
    } else if (!WeightingMode.empty()) {
        spdlog::error("Unknown weighting mode {}. Aborting.", WeightingMode);
        exit(EXIT_FAILURE);
    } else {
        config.Weighting = std::make_shared<ConstantWeighting>(1.0, 1.0);
    }

    // Regularizer
    switch(Regularizer) {
        case RegularizerType::REG_L2:
            config.Regularizer = objective::SquaredNormConfig{RegScale, !RegBias};
            break;
        case RegularizerType::REG_L1:
            config.Regularizer = objective::HuberConfig{RegScale, 1e-2, !RegBias};
            break;
        case RegularizerType::REG_L1_RELAXED:
            config.Regularizer = objective::HuberConfig{RegScale, 1e-1, !RegBias};
            break;
        case RegularizerType::REG_HUBER:
            config.Regularizer = objective::HuberConfig{RegScale, 1.0, !RegBias};
            break;
        case RegularizerType::REG_ELASTIC_50_50:
            config.Regularizer = objective::ElasticConfig{RegScale, 1e-1, 0.5, !RegBias};
            break;
        case RegularizerType::REG_ELASTIC_90_10:
            config.Regularizer = objective::ElasticConfig{RegScale, 1e-1, 0.9, !RegBias};
            break;
        default:
            spdlog::error("Unknown regularization mode {}. Aborting.", Regularizer);
            exit(EXIT_FAILURE);
    }


    std::shared_ptr<init::WeightInitializationStrategy> init_strategy;
    if(InitMode == "mean") {
        config.Init = init::create_constant_initializer(-get_mean_feature(*data->get_features()));
    } else if(InitMode == "msi") {
        config.Init = init::create_feature_mean_initializer(data, MSI_PFac, MSI_NFac);
    } else if(InitMode == "multi-pos") {
        config.Init = init::create_multi_pos_mean_strategy(data, InitMaxPos, MSI_PFac, MSI_NFac);
    } else if(InitMode == "ova-primal") {
        config.Init = init::create_ova_primal_initializer(data, config.Regularizer, Loss);
    } else if(InitMode == "bias" || (InitMode.empty() && BiasInitValue.has_value())) {
        if(DataProc.augment_for_bias()) {
            DenseRealVector init_vec(data->num_features());
            init_vec.setZero();
            init_vec.coeffRef(init_vec.size()-1) = BiasInitValue.value_or(-1.0);
            config.Init = init::create_constant_initializer(std::move(init_vec));
        } else {
            spdlog::error("--init-mode=bias requires --augment-for-bias");
            exit(EXIT_FAILURE);
        }
    }

    config.StatsGatherer = std::make_shared<TrainingStatsGatherer>(StatsLevelFile, StatsOutFile);
    config.Loss = Loss;

    return config;
}

int TrainingProgram::run(int argc, const char** argv)
{
    try {
        app.parse(argc, argv);
    } catch (const CLI::ParseError &e) {
        std::exit(app.exit(e));
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

    auto data = DataProc.load(Verbose);

    std::shared_ptr<postproc::PostProcessFactory> permute_post_proc;
    if(ReorderFeatures) {
        auto permute = sort_features_by_frequency(*data);
        permute_post_proc = postproc::create_reordering(permute);
    }

    parse_label_range();

    auto runner = parallel::ParallelRunner(NumThreads);
    if(Verbose > 0)
        runner.set_logger(spdlog::default_logger());

    auto config = make_config(data);

    std::shared_ptr<postproc::PostProcessFactory> post_proc;
    bool use_sparse_model = false;
    switch (SaveOptions.Format) {
        case io::WeightFormat::SPARSE_TXT:
            post_proc = postproc::create_culling(SaveOptions.Culling);
            use_sparse_model = true;
            break;
        case io::WeightFormat::DENSE_TXT:
        default:
            break;
    }

    // if we explicitly enable sparsification, we override the culling post-proc that
    // may implicitly be generated due to the WeightFormat::SPARSE_TXT
    if(Sparsify > 0) {
        post_proc = postproc::create_sparsify(Sparsify / real_t{100});
        use_sparse_model = true;
        SaveOptions.Culling = 1e-10;
    }

    // make sure to combine the post-processing operations
    if(permute_post_proc) {
        if (post_proc) {
            post_proc = postproc::create_combined({post_proc, permute_post_proc});
        } else {
            post_proc = permute_post_proc;
        }
    }


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
    std::optional<io::PartialModelLoader> loader;
    if(*PreTrainedOpt) {
        loader.emplace(SourceModel);
    }
    label_id_t first_label = LabelsBegin;
    if(LabelsEnd == label_id_t{-1}) {
        LabelsEnd = label_id_t{data->num_labels()};
    }
    label_id_t next_label = std::min(LabelsEnd, first_label + BatchSize);
    std::future<io::model::WeightFileEntry> saving;

    config.PostProcessing = post_proc;
    config.Sparse = use_sparse_model;

    while(true) {
        spdlog::info("Starting batch {} - {}", first_label.to_index(), next_label.to_index());

        if(loader.has_value()) {
            auto initial_weights = loader->load_model(first_label, next_label);
            config.Init = init::create_pretrained_initializer(initial_weights);
        }

        // update time limit to respect remaining time
        runner.set_time_limit(std::chrono::duration_cast<std::chrono::milliseconds>(timeout_time - std::chrono::steady_clock::now()));

        std::shared_ptr<TrainingSpec> train_spec = create_dismec_training(data, hps, config);
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

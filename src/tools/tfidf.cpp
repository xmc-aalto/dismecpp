//
// Created by erik on 28.1.2022.
//

#include "io/xmc.h"
#include "data/data.h"
#include "data/transform.h"
#include "CLI/CLI.hpp"
#include "spdlog/spdlog.h"
#include "spdlog/stopwatch.h"

using namespace dismec;

void apply_tfidf(SparseFeatures& features, const DenseRealVector& idf) {
    for(int row = 0; row < features.rows(); ++row) {
        for(auto it = SparseFeatures::InnerIterator(features, row); it; ++it) {
            it.valueRef() = (1 + std::log(it.valueRef())) * idf.coeff(it.col());
        }
    }

    normalize_instances(features);
}

int main(int argc, const char** argv) {
    std::string TrainSetFile;
    std::string TestSetFile;
    std::string OutputTrain;
    std::string OutputTest;
    bool OneBasedIndex = false;
//    bool Reorder = false;
    CLI::App app{"tfidf"};
    app.add_option("train-set", TrainSetFile,
                   "The training dataset will be loaded from here.")->required()->check(CLI::ExistingFile);
    auto* test_set_opt = app.add_option("--test-set", TestSetFile,
                   "The test dataset will be loaded from here. "
                   "If given, it will use the idf as calculated on the training set")->check(CLI::ExistingFile);
    app.add_option("out", OutputTrain,
                   "The file to which the result (for the train set) will be saved.")->required();
    app.add_option("--test-out", OutputTest,
                   "The file to which the result for the test set will be saved.")->needs(test_set_opt);

    app.add_flag("--one-based-index", OneBasedIndex,
                 "If this flag is given, then we assume that the input dataset in xmc format and"
                 " has one-based indexing, i.e. the first label and feature are at index 1  (as opposed to the usual 0)");
/*
 * TODO
 */
/*
    app.add_flag("--reorder", Reorder,
                 "If given, the features will be reordered based on their frequency. For large feature matrices, this may result in better"
                 "performance for sparse matrix multiplications.");
*/
    try {
        app.parse(argc, argv);
    } catch (const CLI::ParseError &e) {
        return app.exit(e);
    }

    auto train_data = read_xmc_dataset(TrainSetFile, OneBasedIndex ? io::IndexMode::ONE_BASED : io::IndexMode::ZERO_BASED);
    spdlog::info("Read dataset from {} with {} instances and {} features.", TrainSetFile, train_data.num_examples(), train_data.num_features());
    auto& train_features = train_data.edit_features()->sparse();

    spdlog::stopwatch timer;
    auto ftr_count = count_features(train_features);

    // then rescale by idf
    DenseRealVector scale = DenseRealVector::NullaryExpr(ftr_count.size(), 1,
                                                         [&](Eigen::Index i){ return std::log(train_features.rows() / std::max(1l, ftr_count[i])); });

    apply_tfidf(train_features, scale);
    spdlog::info("Applied tfidf transform in {:.3}s.", timer);

    timer.reset();
    io::save_xmc_dataset(OutputTrain, train_data);
    spdlog::info("Saved dataset to {} in {:.3}s.", OutputTrain, timer);

    if(!TestSetFile.empty()) {
        spdlog::info("Processing test dataset");
        auto test_data = read_xmc_dataset(TestSetFile, OneBasedIndex ? io::IndexMode::ONE_BASED : io::IndexMode::ZERO_BASED);
        auto& test_features = test_data.edit_features()->sparse();
        timer.reset();
        apply_tfidf(test_features, scale);
        spdlog::info("Applied tfidf transform to test data in {:.3}s.", timer);

        timer.reset();
        io::save_xmc_dataset(OutputTest, test_data);
        spdlog::info("Saved test data to {} in {:.3}s.", OutputTest, timer);
    }
}
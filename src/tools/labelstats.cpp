//
// Created by erik on 28.1.2022.
//

#include "io/xmc.h"
#include "data/data.h"
#include "nlohmann/json.hpp"
#include "CLI/CLI.hpp"
#include <numeric>
#include <random>

using json = nlohmann::json;

double obesity(const std::vector<long>& values, int num_samples);

int main(int argc, const char** argv) {
    std::string DataSetFile;
    std::string OutputFile;
    bool OneBasedIndex = false;
    CLI::App app{"labelstats"};
    app.add_option("dataset", DataSetFile,
                   "The file from which the data will be loaded.")->required()->check(CLI::ExistingFile);
    app.add_option("target", OutputFile,
                   "The file to which the result will be saved.")->required();

    app.add_flag("--one-based-index", OneBasedIndex,
                 "If this flag is given, then we assume that the input dataset in xmc format and"
                 " has one-based indexing, i.e. the first label and feature are at index 1  (as opposed to the usual 0)");

    try {
        app.parse(argc, argv);
    } catch (const CLI::ParseError &e) {
        return app.exit(e);
    }

    auto data = read_xmc_dataset(DataSetFile, OneBasedIndex ? io::IndexMode::ONE_BASED : io::IndexMode::ZERO_BASED);
    std::vector<long> label_counts;
    for(long id = 0; id < data.num_labels(); ++id) {
        label_counts.push_back(static_cast<long>(data.num_positives(label_id_t{id})));
    }

    std::sort(begin(label_counts), end(label_counts));

    json result;
    result["num-labels"] = data.num_labels();
    result["num-instances"] = data.num_examples();
    result["most-frequent"] = label_counts.back();
    result["least-frequent"] =  label_counts.front();
    result["intra-IR-min"] = double(data.num_examples()) / double(std::max(1l, label_counts.back()));
    result["intra-IR-max"] = double(data.num_examples()) / double(std::max(1l, label_counts.front()));
    result["inter-IR"] = double(label_counts.back()) / double(std::max(1l, label_counts.front()));

    // check where the 80-20 (and similar) rule would bring us
    std::vector<long> cumulative;
    std::partial_sum(label_counts.rbegin(), label_counts.rend(), std::back_inserter(cumulative));
    int target = 10;
    std::cout << cumulative[0] << " " << cumulative[1] << " " << cumulative[cumulative.size() - 1] << "\n";
    for(int i = 0; i < cumulative.size(); ++i) {
        if(cumulative[i] / target >= cumulative.back() / 100) {
            result["cumulative-" + std::to_string(target)] = i;
            result["cumulative-rel-" + std::to_string(target)] = 100.0 * double(i) / double(data.num_labels());
            target += 10;
        }
    }

    result["obesity"] = obesity(label_counts, 10000);

    std::fstream result_file(OutputFile, std::fstream::out);
    result_file << std::setw(4) << result << "\n";


}


double obesity(const std::vector<long>& values, int num_samples) {
    std::ranlux48 rng;
    std::uniform_int_distribution<long> dist(0, values.size() - 1);
    std::array<long, 4> sample;
    int larger = 0;
    for(int i = 0; i < num_samples; ++i) {
        for(auto& s : sample)  s = dist(rng);
        std::sort(begin(sample), end(sample));
        if(values[sample[0]] + values[sample[3]] > values[sample[1]] + values[sample[2]]) {
            ++larger;
        }
    }

    return double(larger) / double(num_samples / 100);
}
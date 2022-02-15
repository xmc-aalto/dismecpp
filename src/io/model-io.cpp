// Copyright (c) 2021, Aalto University, developed by Erik Schultheis
// All rights reserved.
//
// SPDX-License-Identifier: MIT

#include "io/model-io.h"
#include "io/common.h"
#include "io/weights.h"
#include "model/dense.h"
#include "model/sparse.h"
#include "model/submodel.h"
#include <fstream>
#include <array>
#include <iomanip>
#include <utility>
#include "spdlog/spdlog.h"
#include "spdlog/fmt/fmt.h"
#include "spdlog/fmt/chrono.h"
#include "nlohmann/json.hpp"
#include "parallel/numa.h"

using json = nlohmann::json;
using namespace io::model;

namespace {
    /// Translation from \ref io::model::WeightFormat to `std::string`.
    const std::array<const char*, 4> weight_format_names = {
            "DenseTXT", "SparseTXT", "DenseNPY", "<NULL>"
    };

    /// Lookup which mode has sparse weights
    const std::array<bool, 4> weight_format_sparsity = {
            false, true, false, true
    };

    /*!
     * \brief This function calls on of the save functions, depending on the format specified on `options`
     * \param target Stream where to save the weights.
     * \param model The model whose weights will be saved.
     * \param options Saving options. The format of the save file depends on this.
     */
    void save_weights_dispatch(std::ostream& target, const Model& model, SaveOption& options) {
        target.precision(options.Precision);
        switch (options.Format) {
            case WeightFormat::DENSE_TXT:
                save_dense_weights_txt(target, model);
                break;
            case WeightFormat::SPARSE_TXT:
                save_as_sparse_weights_txt(target, model, options.Culling);
                break;
            case WeightFormat::DENSE_NPY:
                save_dense_weights_npy(*target.rdbuf(), model);
                break;
            case WeightFormat::NULL_FORMAT:
                return;
        }
    }

    /*!
     * \brief This function calls the reads function that corresponds to the given weight format.
     * \param source Stream from which the data is read.
     * \param format Data format used to interpret `source`.
     * \param model Model in which the weights will be stored. Must already have allocated
     * enough space to store the weights.
     */
    void read_weights_dispatch(std::istream& source, WeightFormat format, Model& model) {
        switch (format) {
            case WeightFormat::DENSE_TXT:
                load_dense_weights_txt(source, model);
                break;
            case WeightFormat::SPARSE_TXT:
                load_sparse_weights_txt(source, model);
                break;
            case WeightFormat::DENSE_NPY:
                load_dense_weights_npy(*source.rdbuf(), model);
                break;
            default:
                throw std::runtime_error("Invalid format");
        }
    }
}

WeightFormat io::model::parse_weights_format(std::string_view weight_format) {
    auto format_index = std::find(begin(weight_format_names), end(weight_format_names), weight_format);
    return static_cast<WeightFormat>(std::distance(begin(weight_format_names), format_index));
}

const char* io::model::to_string(WeightFormat format) {
    return weight_format_names.at(static_cast<unsigned long>(format));
}

void PartialModelIO::read_metadata_file(const path& meta_file) {
    std::fstream source(meta_file, std::fstream::in);
    if(!source.is_open()) {
        throw std::runtime_error(fmt::format("Could not open model metadata file '{}'", meta_file.c_str()));
    }

    // read and parse the metadata
    std::string s;
    getline (source, s,  '\0');
    json meta = json::parse(s);

    m_NumFeatures = meta["num-features"];
    m_TotalLabels = meta["num-labels"];

    for(auto& weight_file : meta["files"]) {
        label_id_t first = label_id_t{weight_file["first"]};
        long count = weight_file["count"];
        std::string weight_format = weight_file["weight-format"];
        auto format = parse_weights_format(weight_format);
        m_SubFiles.push_back(WeightFileEntry{first, count, weight_file["file"], format});
    }
}

auto io::model::PartialModelIO::label_lower_bound(label_id_t pos) const -> std::vector<WeightFileEntry>::const_iterator {
    return std::lower_bound(begin(m_SubFiles), end(m_SubFiles), pos,
                            [](const WeightFileEntry& s, label_id_t val) {
                                return s.First < val;
                            });
}

void io::model::PartialModelIO::insert_sub_file(const WeightFileEntry& sub) {
    auto last_label = [](const WeightFileEntry& sf) {
        return sf.First + (sf.Count - 1);
    };

    auto insert_pos = label_lower_bound(sub.First);
    // now, insert pos points to the element that will end up following the newly inserted element.

    // check that there is no overlap between partial models
    // if there is a successor element
    if(insert_pos != m_SubFiles.end()) {
        if(last_label(sub) >= insert_pos->First) {
            throw std::logic_error(fmt::format("Overlap detected! Partial model in file {} stores weights {}-{}, "
                                               "partial model in file {} stores {}-{}", insert_pos->FileName,
                                               insert_pos->First.to_index(), last_label(*insert_pos).to_index(), sub.FileName,
                                               sub.First.to_index(), last_label(sub).to_index()));
        }

    }

    // if there is a previous element, also check that
    if(insert_pos != m_SubFiles.begin()) {
        const auto& prev_el = *std::prev(insert_pos);
        if(last_label(prev_el) >= sub.First) {
            throw std::logic_error(fmt::format("Overlap detected! Partial model in file {} stores weights {}-{}, "
                                               "partial model in file {} stores {}-{}", prev_el.FileName, prev_el.First.to_index(),
                                               last_label(prev_el).to_index(), sub.FileName, sub.First.to_index(), last_label(sub).to_index()));
        }
    }

    m_SubFiles.insert(insert_pos, sub);
}

// -------------------------------------------------------------------------------
//                      Partial Model Saver
// -------------------------------------------------------------------------------

PartialModelSaver::PartialModelSaver(path target_file, SaveOption options, bool load_partial) :
    m_MetaFileName(std::move(target_file)), m_Options(options) {
    if(load_partial) {
        read_metadata_file(m_MetaFileName);
    }
    // check validity of save location
    if(!m_MetaFileName.parent_path().empty() && !std::filesystem::exists(m_MetaFileName.parent_path())) {
        throw std::runtime_error(fmt::format("Cannot save to '{}' because directory does not exist.",
                                             m_MetaFileName.c_str()));
    }
}

std::future<WeightFileEntry> PartialModelSaver::add_model(const std::shared_ptr<const Model>& model, std::optional<std::string> file_path)
{
    // check compatibility of the partial model
    // if this is the first partial model, accept the values
    if(m_TotalLabels == -1) {
        m_TotalLabels = model->num_labels();
        m_NumFeatures = model->num_features();
    } else {
        // we know what to expect, verify
        if(m_TotalLabels != model->num_labels()) {
            throw std::logic_error(fmt::format("Received partial model for {} labels, but expected {} labels",
                                   model->num_labels(), m_TotalLabels));
        }
        if(m_NumFeatures != model->num_features()) {
            throw std::logic_error(fmt::format("Received partial model for {} features, but expected {} features",
                                               model->num_features(), m_NumFeatures));
        }
    }

    path target_file = file_path.value_or(m_MetaFileName);
    if(!file_path.has_value()) {
        target_file.replace_filename(fmt::format("{}.weights-{}-{}", m_MetaFileName.filename().c_str(),
                                                 model->labels_begin().to_index(), model->labels_end().to_index() - 1));
    }

    // find out where to insert this partial model into the ordered list
    WeightFileEntry new_weights_file{model->labels_begin(), model->num_weights(), target_file.filename(), m_Options.Format};

    // skip if we don't want to save weights
    if(m_Options.Format == WeightFormat::NULL_FORMAT) {
        // if we don't actually want to write anything (e.g. for test cases, we just register the sub file and be done)
        insert_sub_file(new_weights_file);
        return std::async(std::launch::deferred, [new_weights_file](){
            return new_weights_file;
        });
    }

    using namespace std::chrono;
    std::fstream weights_file(target_file, std::fstream::out | std::fstream::binary);
    if(!weights_file.is_open()) {
        throw std::runtime_error(fmt::format("Could not create weights file {}", target_file.c_str()));
    }

    // add the weights file to the list of weights
    insert_sub_file(new_weights_file);

    // OK, now we are sure everything is fine, do the actual work on saving stuff
    // we do this asynchronous, as we expect this to mostly IO bound
    return std::async(std::launch::async, [this, target=std::move(weights_file), model, new_weights_file]() mutable
    {
        // let the saving thread only run on cores of the numa node which stores the model.
        // this assumes that the internal buffers of model are on the same node as model.
        parallel::pin_to_data(model.get());
        auto now = steady_clock::now();
        save_weights_dispatch(target, *model, m_Options);
        spdlog::info("Saving partial model took {} ms", duration_cast<milliseconds>(steady_clock::now() - now).count());
        return new_weights_file;
    });
}

void PartialModelSaver::update_meta_file() {
    json meta;
    meta["num-features"] = m_NumFeatures;
    meta["num-labels"] = m_TotalLabels;
    std::time_t tt = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
    std::tm tm = *std::gmtime(&tt);
    char date_buffer[128];
    strftime(date_buffer, sizeof(date_buffer), "%F - %T", &tm);
    meta["date"] = date_buffer;

    for(auto& sub : m_SubFiles)
    {
        json file_data = {{"first", sub.First.to_index()}, {"count", sub.Count},
                          {"file", sub.FileName}, {"weight-format", to_string(sub.Format)}};
        meta["files"].push_back(file_data);
    }

    std::fstream meta_file(m_MetaFileName, std::fstream::out);
    meta_file << std::setw(4) <<  meta << "\n";
}

bool io::model::PartialModelSaver::any_weight_vector_for_interval(label_id_t begin, label_id_t end) const
{
    auto iter = label_lower_bound(begin);
    // check that there is no overlap between partial models
    // if there is a successor element
    if(iter != m_SubFiles.end()) {
        if(end > iter->First) {
            return true;
        }
    }

    // if there is a previous element, also check that
    if(iter != m_SubFiles.begin()) {
        const auto& prev_el = *std::prev(iter);
        if(prev_el.First + prev_el.Count > begin) {
            return true;
        }
    }

    return false;
}

void PartialModelSaver::finalize() {
    label_id_t last_end{0};
    for(auto& sub : m_SubFiles) {
        if(last_end != sub.First) {
            throw std::logic_error(fmt::format("Some labels are missing. Gap from {} to {}", last_end.to_index(), sub.First.to_index() - 1));
        }
        last_end = sub.First + sub.Count;
    }
    if(last_end.to_index() != m_TotalLabels) {
        throw std::logic_error(fmt::format("Some labels are missing. Gap from {} to {}", last_end.to_index(), m_TotalLabels - 1));
    }

    update_meta_file();
}

std::pair<label_id_t, label_id_t> PartialModelSaver::get_missing_weights() const {
    label_id_t last_end{0};
    label_id_t label_end{m_TotalLabels};
    for(auto& sub : m_SubFiles) {
        if(last_end != sub.First) {
            return {last_end, sub.First};
        }
        last_end = sub.First + sub.Count;
    }
    if(last_end != label_end) {
        return {last_end, label_end};
    }

    return {label_end, label_end};
}

void io::model::save_model(const path& target_file, const std::shared_ptr<const Model>& model, SaveOption options)
{
    if(model->is_partial_model()) {
        throw std::logic_error("save_model can only save complete models");
    }

    PartialModelSaver saver{target_file, options};

    // TODO more info, i.e. bias, which labels, etc
    if(model->num_labels() < options.SplitFiles) {
        saver.add_model(model, fmt::format("{}.weights", target_file.filename().c_str()));
    } else {
        int num_files = std::round( model->num_weights() / static_cast<double>(options.SplitFiles) );
        for(int sub = 0; sub < num_files; ++sub) {
            std::string weights_file_name = fmt::format("{}.weights-{}-of-{}", target_file.filename().c_str(), 1+sub, num_files);
            label_id_t first{sub * options.SplitFiles};
            long end_label = (sub == num_files - 1) ? model->num_weights() : (sub+1) * options.SplitFiles;
            end_label = std::min(end_label, model->num_labels());
            auto submodel = std::make_shared<::model::ConstSubModelView>(model.get(), first, label_id_t{end_label});
            saver.add_model(submodel, weights_file_name);
        }
    }

    saver.finalize();
}

std::shared_ptr<Model> io::model::load_model(path source)
{
    PartialModelLoader loader{std::move(source)};
    return loader.load_model(label_id_t{0}, label_id_t{loader.num_labels()});
}


PartialModelLoader::PartialModelLoader(path meta_file, ESparseMode mode) :
    m_MetaFileName(std::move(meta_file)), m_SparseMode(mode) {
    read_metadata_file(m_MetaFileName);
}

auto PartialModelLoader::get_loading_range(label_id_t label_begin, label_id_t label_end) const ->
    SubModelRangeSpec
{
    auto sub_files = std::equal_range(begin(m_SubFiles), end(m_SubFiles),
                                      WeightFileEntry{label_begin, label_end - label_begin},
                                      [](const WeightFileEntry& left, const WeightFileEntry& right) {
                                          auto a1 = left.First;
                                          auto a2 = right.First;
                                          auto b1 = a1 + left.Count;
                                          return b1 <= a2;
                                      });
    if(sub_files.first == end(m_SubFiles)) {
        throw std::runtime_error(fmt::format("Could not find weights for interval [{}, {})",
                                             label_begin.to_index(), label_end.to_index()));
    }

    auto calc_label_count = [&](){
        if(sub_files.second == end(m_SubFiles)) {
            return label_id_t{m_TotalLabels};
        } else {
            return sub_files.second->First;
        }
    };

    return SubModelRangeSpec{sub_files.first, sub_files.second, sub_files.first->First, calc_label_count()};
}

std::shared_ptr<Model> PartialModelLoader::load_model(label_id_t label_begin, label_id_t label_end) const {
    auto sub_range = get_loading_range(label_begin, label_end);


    ::model::PartialModelSpec spec {sub_range.LabelsBegin, sub_range.LabelsEnd - sub_range.LabelsBegin, m_TotalLabels};

    auto model = std::make_shared<::model::DenseModel>(m_NumFeatures, spec);

    for(auto file = sub_range.FilesBegin; file < sub_range.FilesEnd; ++file) {
        ::model::SubModelView submodel{model.get(), file->First, file->First + file->Count};
        path weights_file = m_MetaFileName;
        std::fstream source(weights_file.replace_filename(file->FileName), std::fstream::in);
        if(!source.is_open()) {
            THROW_ERROR("Could not open weights file ", weights_file.replace_filename(file->FileName).string());
        }
        read_weights_dispatch(source, file->Format, submodel);
        spdlog::info("read weight file {}", weights_file.replace_filename(file->FileName).c_str());
    }

    return model;
}

long PartialModelLoader::num_weight_files() const {
    return m_SubFiles.size();
}

namespace {
    std::shared_ptr<Model> make_model(long num_features, ::model::PartialModelSpec spec, bool sparse) {
        if(sparse) {
            return std::make_shared<::model::SparseModel>(num_features, spec);
        } else {
            return std::make_shared<::model::DenseModel>(num_features, spec);
        }
    }

    bool use_sparse_weights(PartialModelLoader::ESparseMode mode, WeightFormat format) {
        switch (mode) {
            case PartialModelLoader::FORCE_SPARSE:
                return true;
            case PartialModelLoader::FORCE_DENSE:
                return false;
            case PartialModelLoader::DEFAULT:
                return weight_format_sparsity[static_cast<int>(format)];
            default:
                assert(0);
                __builtin_unreachable();
        }
    }
}

std::shared_ptr<Model> PartialModelLoader::load_model(int index) const {
    auto start = std::chrono::steady_clock::now();
    const WeightFileEntry& entry = m_SubFiles.at(index);
    ::model::PartialModelSpec spec {entry.First, entry.Count, num_labels()};

    auto model = make_model(num_features(), spec, use_sparse_weights(m_SparseMode, entry.Format));
    path weights_file = meta_file_path();
    std::fstream source(weights_file.replace_filename(entry.FileName), std::fstream::in);
    if(!source.is_open()) {
        THROW_ERROR("Could not open weights file ", weights_file.replace_filename(entry.FileName).string());
    }
    read_weights_dispatch(source, entry.Format, *model);
    auto duration = std::chrono::steady_clock::now() - start;
    spdlog::info("read weight file '{}' in {}ms", weights_file.replace_filename(entry.FileName).c_str(),
                 std::chrono::duration_cast<std::chrono::milliseconds>(duration).count());
    return model;
}



#include "doctest.h"

using ::model::DenseModel;
using ::model::PartialModelSpec;

/*!
 * \test This test case checks that `PartialModelSaver` verifies that the partial models it receives are compatible.
 * This includes checking that the number of total labels/features match.
 * To check that the error condition we want to check for is really the problem, we first try the `add_model` call
 * with an invalid partial model, and secondly call it with a corrected version. If the corrected version also fails,
 * we know that our check did not work, or that the first call did throw but still added the partial model to its
 * model list. We assume that label overlap tests are correctly performed; this is tested in the unit test for
 * `insert_sub_file`.
 */
TEST_CASE("partial model writer verifier") {
    SaveOption options;
    options.Format = WeightFormat::NULL_FORMAT;
    PartialModelSaver pms("test_data/pms-test", options);

    auto first_part = std::make_shared<DenseModel>(4, PartialModelSpec{label_id_t{1}, 4, 20});
    // this just causes the setup, we can't create any contradictions yet
    REQUIRE_NOTHROW(pms.add_model(first_part));

    SUBCASE("mismatched features") {
        PartialModelSpec spec{label_id_t{5}, 2, 20};
        auto mismatched_features = std::make_shared<DenseModel>(7, spec);
        auto valid = std::make_shared<DenseModel>(4, spec);
        REQUIRE_THROWS(pms.add_model(mismatched_features));
        CHECK_NOTHROW(pms.add_model(valid));
    }

    SUBCASE("mismatched total labels") {
        auto mismatched_labels = std::make_shared<DenseModel>(4, PartialModelSpec{label_id_t{5}, 2, 50});
        auto valid = std::make_shared<DenseModel>(4, PartialModelSpec{label_id_t{5}, 2, 20});
        REQUIRE_THROWS(pms.add_model(mismatched_labels));
        CHECK_NOTHROW(pms.add_model(valid));
    }

    SUBCASE("incomplete model") {
        CHECK_THROWS(pms.finalize());
    }
}

TEST_CASE("label lower bound") {
    struct TestModel : public PartialModelIO {
        using PartialModelIO::label_lower_bound;
        using PartialModelIO::m_SubFiles;
        TestModel() {
            m_SubFiles.push_back(WeightFileEntry{label_id_t{20}, 30, "", WeightFormat::DENSE_TXT});
            m_SubFiles.push_back(WeightFileEntry{label_id_t{100}, 50, "", WeightFormat::DENSE_TXT});
        }
    };

    TestModel tm{};
    CHECK(tm.label_lower_bound(label_id_t{0}) == tm.m_SubFiles.begin());
    CHECK(tm.label_lower_bound(label_id_t{20}) == tm.m_SubFiles.begin());
    CHECK(tm.label_lower_bound(label_id_t{40}) == tm.m_SubFiles.begin() + 1);
    CHECK(tm.label_lower_bound(label_id_t{50}) == tm.m_SubFiles.begin() + 1);
    CHECK(tm.label_lower_bound(label_id_t{80}) == tm.m_SubFiles.begin() + 1);
    CHECK(tm.label_lower_bound(label_id_t{100}) == tm.m_SubFiles.begin() + 1);
    CHECK(tm.label_lower_bound(label_id_t{120}) == tm.m_SubFiles.end());
    CHECK(tm.label_lower_bound(label_id_t{200}) == tm.m_SubFiles.end());
}

TEST_CASE("insert_sub_file") {
    struct TestModel : public PartialModelIO {
        TestModel() {}
        void insert(long first, long count) {
            insert_sub_file(WeightFileEntry{label_id_t{first}, count, "", WeightFormat::NULL_FORMAT});
        }
    };
    TestModel pms{};

    // this just causes the setup, we can't create any contradictions yet
    REQUIRE_NOTHROW(pms.insert(1, 4));


    SUBCASE("overlap predecessor") {
        REQUIRE_THROWS(pms.insert(4, 2));
        CHECK_NOTHROW(pms.insert(5, 2));
    }

    SUBCASE("overlap successor") {
        REQUIRE_THROWS(pms.insert(0, 2));
        CHECK_NOTHROW(pms.insert(0, 1));
    }
}

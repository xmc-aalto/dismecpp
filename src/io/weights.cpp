// Copyright (c) 2021, Aalto University, developed by Erik Schultheis
// All rights reserved.
//
// SPDX-License-Identifier: MIT

#include "io/weights.h"
#include "io/common.h"
#include "model/model.h"
#include "spdlog/spdlog.h"
#include "io/numpy.h"
#include "utils/eigen_generic.h"

using namespace dismec;
using namespace dismec::io::model;

namespace {
    /*!
     * \brief Basic scaffold for saving weights.
     * \details This function handles the iteration over all the model weights, and extracts
     * the weight vectors. The actual saving is delegated to `weight_callback`.
     */
    template<class F>
    void save_weights(const Model& model, F&& weight_callback) {
        DenseRealVector buffer(model.num_features());
        for (label_id_t label = model.labels_begin(); label < model.labels_end(); ++label) {
            model.get_weights_for_label(label, buffer);
            weight_callback(buffer);
        }
    }

    /*!
     * \brief Basic scaffold for loading weights.
     * \details This function handles the iteration over all the model weights, and inserts
     * the weight vectors. The actual reading is delegated to `read_callback`.
     */
    template<class F>
    void load_weights(Model& target, F&& read_callback) {
        DenseRealVector buffer(target.num_features());
        for (label_id_t label = target.labels_begin(); label < target.labels_end(); ++label) {
            read_callback(buffer);
            target.set_weights_for_label(label, Model::WeightVectorIn{buffer});
        }
    }
}

// -------------------------------------------------------------------------------
//                      dense weights in txt file
// -------------------------------------------------------------------------------


void io::model::save_dense_weights_txt(std::ostream& target, const Model& model)
{
    save_weights(model, [&](const auto& data) {
        io::write_vector_as_text(target, data) << '\n';
        if(target.bad()) {
            throw std::runtime_error("Error while writing weights");
        }
    });
}

void io::model::load_dense_weights_txt(std::istream& source, Model& target)
{
    load_weights(target, [&](auto& data) {
        read_vector_from_text(source, data);
    });
}

// -------------------------------------------------------------------------------
//                      sparse weights in npy file
// -------------------------------------------------------------------------------


void io::model::save_dense_weights_npy(std::streambuf& target, const Model& model) {
    bool col_major = false;
    std::string description = io::make_npy_description(io::data_type_string<real_t>(), col_major, model.contained_labels(), model.num_features());
    io::write_npy_header(target, description);
    save_weights(model, [&](const DenseRealVector & data) {
        binary_dump(target, data.data(), data.data() + data.size());
    });
}

void io::model::load_dense_weights_npy(std::streambuf& source, Model& target)
{
    auto info = parse_npy_header(source);

    // verify that info is consistent
    if(info.DataType != data_type_string<real_t>()) {
        THROW_ERROR("Mismatch in data type, got {} but expected {}", info.DataType, data_type_string<real_t>());
    }

    if(info.Cols != target.num_features()) {
        THROW_ERROR("Weight data has {} columns, but model expects {} features", info.Cols, target.num_features());
    }
    if(info.Rows != target.contained_labels()) {
        THROW_ERROR("Weight data has {} rows, but model expects {} labels", info.Rows, target.contained_labels());
    }

    if(info.ColumnMajor) {
        THROW_ERROR("Weight data is required to be in row-major format");
    }

    load_weights(target, [&](DenseRealVector& data) {
        binary_load(source, data.data(), data.data() + data.size());
    });
}

// -------------------------------------------------------------------------------
//                      sparse weights in txt file
// -------------------------------------------------------------------------------

void io::model::save_as_sparse_weights_txt(std::ostream& target, const Model& model, double threshold)
{
    if(threshold < 0) {
        throw std::invalid_argument("Threshold cannot be negative");
    }

    // TODO should we save the nnz on each line? could make reading as sparse more efficient
    long nnz = 0;
    save_weights(model, [&](const DenseRealVector& data) {
        for(int j = 0; j < data.size(); ++j)
        {
            if(std::abs(data.coeff(j)) > threshold) {
                target << j << ':' << data.coeff(j) << ' ';
                ++nnz;
            }
        }
        target << '\n';
    });

    long entries = model.contained_labels() * model.num_features();
    if(nnz > 0.25 * entries) {
        spdlog::warn("Saved model in sparse mode, but sparsity is only {}%. "
                     "Consider increasing the threshold or saving as dense data.",
                     100 - (100 * nnz) / entries);
    } else {
        spdlog::info("Saved model in sparse mode. Only {:2.2}% of weights exceeded threshold.", double(100 * nnz) / entries);
    }
}

void io::model::load_sparse_weights_txt(std::istream& source, Model& target) {
    Eigen::SparseVector<real_t> sparse_vec;
    sparse_vec.resize(target.num_features());
    std::string line_buffer;
    long num_features = target.num_features();
    for (label_id_t label = target.labels_begin(); label < target.labels_end(); ++label)
    {
        if(!std::getline(source, line_buffer)) {
            THROW_ERROR("Input operation failed when trying to read weights for label {} out of {}",
                        label.to_index(), target.num_labels());
        }
        sparse_vec.setZero();
        try {
            io::parse_sparse_vector_from_text(line_buffer.data(), [&](long index, double value) {
                if (index >= num_features || index < 0) {
                    THROW_ERROR("Encountered index {:5} with value {} for weights of label {:6}. Number of features "
                                "was specified as {}.", index, value, label.to_index(), num_features);
                };
                sparse_vec.insertBack(index) = value;
            });
        } catch (const std::exception& error) {
            THROW_ERROR("Error while parsing weights for label {:6}: {}", label.to_index(), error.what());
        }

        target.set_weights_for_label(label, Model::WeightVectorIn{sparse_vec});
    }
}


#include "doctest.h"
#include "model/dense.h"

using ::model::DenseModel;
using ::model::PartialModelSpec;

/*! \test In this test we check that weight matrices are saved correctly in dense and sparse txt formats.
 * This also verifies that the partial model settings are used correctly, i.e. that we do not try to save
 * weight vectors that are not actually present in the partial model.
 * We then check that loading from the saved strings reproduces the original weights.
 */
TEST_CASE("save/load weights as plain text") {
    // generate some fake data
    DenseModel::WeightMatrix weights(2, 4);
    weights << 1, 0, 0, 2,
            0, 3, 0, -1;

    DenseModel model(std::make_shared<DenseModel::WeightMatrix>(weights), PartialModelSpec{label_id_t{1}, 4, 6});
    DenseModel reconstruct(2, PartialModelSpec{label_id_t{1}, 4, 6});
    std::stringstream target;

    std::string expected_dense = "1 0\n"
                                 "0 3\n"
                                 "0 0\n"
                                 "2 -1\n";

    // note that we currently produce trailing whitespace here
    std::string expected_sparse = "0:1 \n"
                                  "1:3 \n"
                                  "\n"
                                  "0:2 1:-1 \n";

    SUBCASE("save dense txt") {
        save_dense_weights_txt(target, model);
        std::string result = target.str();
        CHECK(result == expected_dense);
    }

    SUBCASE("save sparse txt") {
        save_as_sparse_weights_txt(target, model, 0.0);
        std::string result = target.str();
        CHECK(result == expected_sparse);
    }

    SUBCASE("load dense txt") {
        target.str(expected_dense);
        load_dense_weights_txt(target, reconstruct);
        CHECK(model.get_raw_weights() == reconstruct.get_raw_weights());
    }

    SUBCASE("load sparse txt") {
        target.str(expected_sparse);
        load_sparse_weights_txt(target, reconstruct);
        CHECK(model.get_raw_weights() == reconstruct.get_raw_weights());
    }
}

//! \todo test cases for mismatch

TEST_CASE("save dense npy") {
    DenseModel::WeightMatrix weights(2, 4);
    weights << 1, 0, 0, 2,
            0, 3, 0, -1;

    DenseModel model(std::make_shared<DenseModel::WeightMatrix>(weights), PartialModelSpec{label_id_t{1}, 4, 6});
    DenseModel reconstruct(2, PartialModelSpec{label_id_t{1}, 4, 6});

    std::stringbuf target;
    save_dense_weights_npy(target, model);
    target.pubseekpos(0);
    INFO(target.str());
    load_dense_weights_npy(target, reconstruct);

    CHECK(model.get_raw_weights() == reconstruct.get_raw_weights());
}
// Copyright (c) 2021, Aalto University, developed by Erik Schultheis
// All rights reserved.
//
// SPDX-License-Identifier: MIT

#include "binding.h"

#include "data/data.h"
#include "io/xmc.h"
#include "io/slice.h"

using namespace dismec;
using PyDataSet = std::shared_ptr<DatasetBase>;

#define PY_PROPERTY(function) \
def_property(#function, [](const DatasetBase& pds){ return pds.function(); } , nullptr)

namespace {
    auto num_positives(const DatasetBase& ds, long label) {
        return ds.num_positives(label_id_t{label});
    }
    auto num_negatives(const DatasetBase& ds, long label) {
        return ds.num_negatives(label_id_t{label});
    }
    auto get_labels(const DatasetBase& ds, long id) {
        return *ds.get_labels(label_id_t{id});
    }
    auto get_features(const DatasetBase& ds) {
        return ds.get_features()->unpack_variant();
    }
    auto set_features_sparse(DatasetBase& ds, SparseFeatures features) {
        (*ds.edit_features()) = GenericFeatureMatrix(std::move(features));
    }
    auto set_features_dense(DatasetBase& ds, DenseFeatures features) {
        (*ds.edit_features()) = GenericFeatureMatrix(std::move(features));
    }

    PyDataSet load_xmc(const std::filesystem::path& source_file, bool one_based_indexing) {
        if(one_based_indexing) {
            return wrap_shared(read_xmc_dataset(source_file, io::IndexMode::ONE_BASED));
        } else {
            return wrap_shared(read_xmc_dataset(source_file, io::IndexMode::ZERO_BASED));
        }
    }

    void save_xmc(const std::filesystem::path& target_file, const DatasetBase& ds, int precision) {
        io::save_xmc_dataset(target_file, dynamic_cast<const MultiLabelData&>(ds), precision);
    }

    PyDataSet load_slice(const std::filesystem::path& features_file, const std::filesystem::path& labels_file) {
        return wrap_shared(io::read_slice_dataset(features_file, labels_file));
    }
}

void register_dataset(pybind11::module_& m) {
    // data set
    py::class_<DatasetBase, PyDataSet>(m, "DataSet")
        // we need to distinguish these two overloads by kwarg name I think, because otherwise we get implicit conversions between dense and sparse matrices
        // by having these two signatures, we seem to prevent automatic conversions e.g. from double ndarray to float, and instead get an error message.
        // same later on for set_features
        .def(py::init([](SparseFeatures features, std::vector<std::vector<long>> positives) -> PyDataSet
                      { return std::make_shared<MultiLabelData>(std::move(features), std::move(positives)); }),
             py::kw_only(), py::arg("sparse_features"), py::arg("positives")
        )
        .def(py::init([](DenseFeatures features, std::vector<std::vector<long>> positives) -> PyDataSet
                      { return std::make_shared<MultiLabelData>(std::move(features), std::move(positives)); }),
             py::kw_only(), py::arg("dense_features"), py::arg("positives")
        )
        .PY_PROPERTY(num_features)
        .PY_PROPERTY(num_examples)
        .PY_PROPERTY(num_labels)
        .def("num_positives", num_positives, py::arg("label_id"))
        .def("num_negatives", num_negatives, py::arg("label_id"))
        .def("get_labels",    get_labels,    py::arg("label_id"))
        .def("get_features",  get_features)
        .def("set_features",  set_features_sparse,
             py::kw_only(),
             py::arg("sparse_features"))
        .def("set_features", set_features_dense,
             py::kw_only(),
             py::arg("dense_features"));

    // dataset io functions
    m.def("load_xmc", load_xmc,
          py::arg("source_file"), py::kw_only(),
          py::arg("one_based_index") = false,
          py::call_guard<py::gil_scoped_release>());

    m.def("save_xmc", save_xmc,
          py::arg("file_name"), py::arg("dataset"),
          py::kw_only(), py::arg("precision") = 4,
          py::call_guard<py::gil_scoped_release>()
    );

    m.def("load_slice", load_slice,
          py::kw_only(),py::arg("features"), py::arg("labels"),
          py::call_guard<py::gil_scoped_release>()
    );
}
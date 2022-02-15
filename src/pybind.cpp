// Copyright (c) 2021, Aalto University, developed by Erik Schultheis
// All rights reserved.
//
// SPDX-License-Identifier: MIT

#include <utility>

#include "python/binding.h"
#include "pybind11/pybind11.h"
#include "pybind11/eigen.h"
#include "pybind11/stl.h"
#include "data/data.h"
#include "model/model.h"

#include "io/xmc.h"
#include "io/model-io.h"
#include "io/prediction.h"

#include "training/weighting.h"
#include "training/training.h"
#include "training/initializer.h"

#include "parallel/runner.h"

namespace py = pybind11;

using PyTrainSpec = PyWrapper<TrainingSpec>;
using PySaver = PyWrapper<io::PartialModelSaver>;

#define PY_PROPERTY(type, function) \
def_property(#function, [](const type& pds){ return pds->function(); } , nullptr)

void register_dataset(pybind11::module_& m);
void register_training(pybind11::module_& m);

PYBIND11_MODULE(pydismec, m)
{
    register_dataset(m);
    register_training(m);


    // predictions
    m.def("load_predictions", [](const std::string& file_name) {
        return io::prediction::read_sparse_prediction(file_name);
    });

    // model
    py::class_<PyModel>(m, "Model")
            .PY_PROPERTY(PyModel, num_labels)
            .PY_PROPERTY(PyModel, num_features)
            .PY_PROPERTY(PyModel, num_weights)
            .PY_PROPERTY(PyModel, has_sparse_weights)
            .PY_PROPERTY(PyModel, is_partial_model)
            .def_property("labels_begin", [](const PyModel& pds) { return pds.access().labels_begin().to_index(); } , nullptr)
            .def_property("labels_end", [](const PyModel& pds){ return pds.access().labels_end().to_index(); } , nullptr)
            .def("get_weights_for_label", [](const PyModel& model, long label){
                DenseRealVector target(model.access().num_features());
                model.access().get_weights_for_label(label_id_t{label}, target);
                return target;
            })
            .def("set_weights_for_label", [](PyModel& model, long label, const DenseRealVector& dense_weights){
                model.access().set_weights_for_label(label_id_t{label}, model::Model::WeightVectorIn{dense_weights});
            })
            .def("predict_scores", [](const PyModel& model, const Eigen::Ref<const types::DenseColMajor<real_t>>& instances) {
                PredictionMatrix target(instances.rows(), model.access().num_weights());
                model.access().predict_scores(model::Model::FeatureMatrixIn{instances}, target);
                return target;
            })
            .def("predict_scores", [](const PyModel& model, const SparseFeatures& instances) {
                PredictionMatrix target(instances.rows(), model.access().num_weights());
                model.access().predict_scores(model::Model::FeatureMatrixIn(instances), target);
                return target;
            })
            ;

    m.def("load_model", [](const std::string& file_name) -> PyModel {
        return io::load_model(file_name);
    }, py::arg("file_name"), py::call_guard<py::gil_scoped_release>());

    py::class_<PySaver>(m, "ModelSaver")
            .def(py::init([](std::string_view path, std::string_view format, int precision, double culling, bool load_partial) {
                io::SaveOption options;
                options.Precision = precision;
                options.Culling = culling;
                options.Format = io::model::parse_weights_format(format);
                return io::PartialModelSaver(path, options, load_partial);
                }), py::arg("path"), py::arg("format"), py::arg("precision") = 6, py::arg("culling")=0.0,
                 py::arg("load_partial")=false)
            .PY_PROPERTY(PySaver, num_labels)
            .def("add_model", [](PySaver& saver, const PyModel& model, std::optional<std::string> target_file) {
                auto saved = saver.access().add_model(model.ptr(), std::move(target_file));
                py::dict result_dict;
                io::model::WeightFileEntry entry = saved.get();
                result_dict["first"] = entry.First.to_index();
                result_dict["count"] = entry.Count;
                result_dict["file"]  = entry.FileName;
                result_dict["format"] = to_string(entry.Format);
                saver.access().update_meta_file();
                return result_dict;
            }, py::arg("model"), py::arg("target_path") = std::nullopt, py::call_guard<py::gil_scoped_release>())
            .def("add_meta", [](PySaver& saver, py::dict data) {
                io::model::WeightFileEntry entry{
                    label_id_t(data["first"].cast<long>()),
                    data["count"].cast<long>(),
                    data["file"].cast<std::string>(),
                    io::model::parse_weights_format(data["format"].cast<std::string>())};
                saver.access().insert_sub_file(entry);
                saver.access().update_meta_file();
            })
            .def("get_missing_weights", [](PySaver& saver) {
                auto interval = saver.access().get_missing_weights();
                return std::make_pair(interval.first.to_index(), interval.second.to_index());
            })
            .def("any_weight_vector_for_interval", [](PySaver& saver, int begin, int end) {
                return saver.access().any_weight_vector_for_interval(label_id_t{begin}, label_id_t{end});
            })
            ;
}
// Copyright (c) 2021, Aalto University, developed by Erik Schultheis
// All rights reserved.
//
// SPDX-License-Identifier: MIT

#include "binding.h"

#include "data/data.h"

#include "training/weighting.h"
#include "training/training.h"
#include "training/initializer.h"
#include "training/postproc.h"

#include "parallel/runner.h"
#include "objective/regularizers.h"

#include "spdlog/fmt/fmt.h"


using PyWeighting = std::shared_ptr<WeightingScheme>;

void register_regularizers(pybind11::module_& root) {
    auto m = root.def_submodule("reg", "Regularizer configuration types");
    py::class_<objective::SquaredNormConfig>(m, "SquaredNormConfig")
        .def(py::init<real_t, bool>(),
             py::kw_only(), py::arg("strength"), py::arg("ignore_bias") = true)
        .def_readwrite("strength", &objective::SquaredNormConfig::Strength)
        .def_readwrite("ignore_bias", &objective::SquaredNormConfig::IgnoreBias)
        .def("__repr__",
             [](const objective::SquaredNormConfig &a) {
                 return fmt::format("SquaredNormConfig(strength={}, ignore_bias={})", a.Strength, a.IgnoreBias ? "True" : "False");
             }
        );

    py::class_<objective::HuberConfig>(m, "HuberConfig")
        .def(py::init<real_t, real_t, bool>(),
             py::kw_only(), py::arg("strength"), py::arg("epsilon"), py::arg("ignore_bias") = true)
        .def_readwrite("strength", &objective::HuberConfig::Strength)
        .def_readwrite("epsilon", &objective::HuberConfig::Epsilon)
        .def_readwrite("ignore_bias", &objective::HuberConfig::IgnoreBias)
        .def("__repr__",
             [](const objective::HuberConfig &a) {
                 return fmt::format("HuberConfig(strength={}, epsilon={}, ignore_bias={})", a.Strength, a.Epsilon, a.IgnoreBias ? "True" : "False");
             }
        );

    py::class_<objective::ElasticConfig>(m, "ElasticConfig")
        .def(py::init<real_t, real_t, real_t, bool>(),
             py::kw_only(), py::arg("strength"), py::arg("epsilon"), py::arg("interpolation"), py::arg("ignore_bias") = true)
        .def_readwrite("strength", &objective::ElasticConfig::Strength)
        .def_readwrite("epsilon", &objective::ElasticConfig::Epsilon)
        .def_readwrite("interpolation", &objective::ElasticConfig::Interpolation)
        .def_readwrite("ignore_bias", &objective::ElasticConfig::IgnoreBias)
        .def("__repr__",
             [](const objective::ElasticConfig &a) {
                 return fmt::format("ElasticConfig(strength={}, epsilon={}, interpolation={}, ignore_bias={})", a.Strength, a.Epsilon, a.Interpolation, a.IgnoreBias ? "True" : "False");
             }
        );
}


namespace {
    auto get_positive_weight(const WeightingScheme& pds, long label) {
        return pds.get_positive_weight(label_id_t{label});
    }
    auto get_negative_weight(const WeightingScheme& pds, long label) {
        return pds.get_negative_weight(label_id_t{label});
    }

    PyWeighting make_constant(double pos, double neg) {
        return std::make_shared<ConstantWeighting>(pos, neg);
    }
    PyWeighting make_propensity(const DatasetBase& data, double a, double b) {
        return std::make_shared<PropensityWeighting>(PropensityModel(&data, a, b));
    }
    PyWeighting make_custom(DenseRealVector pos, DenseRealVector neg) {
        return std::make_shared<CustomWeighting>(std::move(pos), std::move(neg));
    }
}

void register_weighting(pybind11::module_& m) {
    py::class_<WeightingScheme, std::shared_ptr<WeightingScheme>>(m, "WeightingScheme")
        .def("positive_weight", get_positive_weight, py::arg("label"))
        .def("negative_weight", get_negative_weight, py::arg("label"))
        .def_static("Constant", make_constant,
                    py::kw_only(), py::arg("positive") = 1.0, py::arg("negative") = 1.0)
        .def_static("Propensity", make_propensity,
                    py::arg("dataset"),
                    py::kw_only(), py::arg("a") = 0.55, py::arg("b") = 1.5)
        .def_static("Custom", make_custom,
                    py::kw_only(), py::arg("positive"), py::arg("negative"))
        ;
}

void register_init(pybind11::module_& root) {
    auto m = root.def_submodule("init", "Initialization configuration types");
    using namespace init;

    py::class_<WeightInitializationStrategy, std::shared_ptr<WeightInitializationStrategy>>(m, "Initializer");
    m.def("zero", [](){
        return create_zero_initializer();
    });

    m.def("constant", [](const DenseRealVector& vec){
        return create_constant_initializer(vec);
    }, py::arg("vector"));

    m.def("feature_mean", [](std::shared_ptr<DatasetBase> dataset, real_t pos, real_t neg){
        return create_feature_mean_initializer(dataset, pos, neg);
    }, py::kw_only(), py::arg("data"), py::arg("positive_margin")=1, py::arg("negative_margin")=-2);

    m.def("multi_feature_mean", [](std::shared_ptr<DatasetBase> dataset, int max_pos, real_t pos, real_t neg){
        return create_multi_pos_mean_strategy(dataset, max_pos, pos, neg);
    }, py::kw_only(), py::arg("data"), py::arg("max_pos"), py::arg("positive_margin")=1, py::arg("negative_margin")=-2);


    m.def("ova_primal", [](std::shared_ptr<DatasetBase> dataset, RegularizerSpec reg, LossType loss){
        return create_ova_primal_initializer(dataset, reg, loss);
    }, py::kw_only(), py::arg("data"), py::arg("reg"), py::arg("loss"));
}

void register_training(pybind11::module_& m) {
    register_weighting(m);
    register_regularizers(m);
    register_init(m);

    py::class_<DismecTrainingConfig>(m, "TrainingConfig")
        .def(py::init([](PyWeighting weighting, RegularizerSpec regularizer, std::shared_ptr<init::WeightInitializationStrategy> init, LossType loss, real_t culling) {
            std::shared_ptr<postproc::PostProcessFactory> pf{};
            bool sparse = false;
            if(culling > 0) {
                pf = postproc::create_culling(culling);
                sparse = true;
            }
            return DismecTrainingConfig{std::move(weighting), std::move(init), std::move(pf), nullptr, sparse, regularizer, loss};
        }), py::kw_only(), py::arg("weighting"), py::arg("regularizer"), py::arg("init"),
             py::arg("loss"), py::arg("culling"))
        .def_readwrite("regularizer", &DismecTrainingConfig::Regularizer)
        .def_readwrite("sparse_model", &DismecTrainingConfig::Sparse)
        .def_readwrite("weighting", &DismecTrainingConfig::Weighting)
        .def_readwrite("loss", &DismecTrainingConfig::Loss);

    py::enum_<LossType>(m, "LossType")
        .value("SquaredHinge", LossType::SQUARED_HINGE)
        .value("Hinge", LossType::HINGE)
        .value("Logistic", LossType::LOGISTIC)
        .value("HuberHinge", LossType::HUBER_HINGE);


    /*
    std::shared_ptr<postproc::PostProcessFactory> PostProcessing;
    std::shared_ptr<TrainingStatsGatherer> StatsGatherer;
     */



    m.def("parallel_train", [](const PyDataSet& data, const py::dict& hyper_params,
            const DismecTrainingConfig& config, long label_begin,
            long label_end, long threads) -> py::dict
    {
        HyperParameters hps;
        for (auto item : hyper_params)
        {
            if(pybind11::isinstance<pybind11::int_>(item.second)) {
                hps.set(item.first.cast<std::string>(), item.second.cast<long>());
            } else {
                hps.set(item.first.cast<std::string>(), item.second.cast<double>());
            }
        };

        auto spec = create_dismec_training(data, hps, config);

        parallel::ParallelRunner runner(threads);
        runner.set_logger(spdlog::default_logger());
        // TODO give more detailled result
        auto result = run_training(runner, spec, label_id_t{label_begin}, label_id_t{label_end});
        py::dict rdict;
        rdict["loss"] = result.TotalLoss;
        rdict["grad"] = result.TotalGrad;
        rdict["finished"] = result.IsFinished;
        rdict["model"] = PyModel(std::move(result.Model));
        return rdict;
    }, py::arg("data"), py::arg("hyperparameters"), py::arg("spec"), py::arg("label_begin") = 0, py::arg("label_end") = -1,
        py::arg("num_threads") = -1, py::call_guard<py::gil_scoped_release>());
    // TODO check constness and lifetime of returns
}
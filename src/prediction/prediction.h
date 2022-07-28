// Copyright (c) 2021, Aalto University, developed by Erik Schultheis
// All rights reserved.
//
// SPDX-License-Identifier: MIT

#ifndef DISMEC_PREDICTION_H
#define DISMEC_PREDICTION_H

#include "parallel/task.h"
#include "parallel/numa.h"
#include "matrix_types.h"
#include "fwd.h"
#include <memory>

namespace dismec::prediction {
    using model::Model;

    /*!
     * \brief Base class for handling predictions.
     * \details This class manages the dataset and model used for batch prediction. It ensures the
     * features are replicated across the NUMA nodes. Batch-prediction is a difficult process, because:
     *  -# The model may be too large to fit into RAM at once
     *  -# The predicted scores may be too many to fit into RAM at once.
     * There are several ways how this could be countered. If only the model is too large, then one may load the model
     * piece by piece, and predict sequentially for the different parts of the model. Parallelism can be achieved by having
     * different threads handle different examples. Such a process is implemented in `FullPredictionTaskGenerator`.
     *
     * However, if the predictions cannot be stored in memory, things become more complicated. For many applications, though,
     * one does not actually need the full prediction scores vector, because the prediction is interpreted as a ranking
     * task and only the top-k entries are of relevance. In that case, the top-k reduction can be performed as an
     * accumulation step over the different partial models, and the full score vector never need be stored. This type of
     * prediction is realized through `TopKPredictionTaskGenerator`.
     */
    class PredictionBase : public parallel::TaskGenerator {
    public:
        //! Constructor, checks that `data` and `model` are compatible.
        PredictionBase(const DatasetBase* data, std::shared_ptr<const Model> model);

    protected:
        const DatasetBase* m_Data;              //!< Data on which the prediction is run
        std::shared_ptr<const Model> m_Model;   //!< Model (possibly partial) for which prediction is run

        void make_thread_local_features(long num_threads);

        void init_thread(thread_id_t thread_id) final;

        /*!
         * \brief Predicts the scores for a subset of the instances given by the half-open interval `[begin, end)`.
         * \details This function is to be used by derived classes to generate the (partial) score predictions.
         * \param begin The index of the first instance for which prediction will be performed.
         * \param end The index past the last instance for which prediction will be performed.
         * \param thread_id Index of the thread which performs the prediction. This argument is needed so the function knows
         * which feature replication to use for optimal performance.
         * \param target Reference to the target array in which the predictions will be saved.
         * \throws std::logic_error if the shape of `target` is not compatible.
         */
        void do_prediction(long begin, long end, thread_id_t thread_id, Eigen::Ref<PredictionMatrix> target);

    private:
        //! The `NUMAReplicator` that generates NUMA-local copies for the feature matrices.
        parallel::NUMAReplicator<const GenericFeatureMatrix> m_FeatureReplicator;

        //! Vector that stores references (as `shared_ptr`) to the NUMA-local copies of the feature matrices,
        //! in such a way that the correct copy can be found by indexing with the thread id.
        std::vector<std::shared_ptr<const GenericFeatureMatrix>> m_ThreadLocalFeatures;
    };

    class FullPredictionTaskGenerator : public PredictionBase {
    public:

        FullPredictionTaskGenerator(const DatasetBase* data, std::shared_ptr<const Model> model);

        void prepare(long num_threads, long chunk_size) override;
        void run_tasks(long begin, long end, thread_id_t thread_id) override;
        [[nodiscard]] long num_tasks() const override;

        [[nodiscard]] const PredictionMatrix& get_predictions() const { return m_Predictions; }
    private:
        PredictionMatrix m_Predictions;
    };

    class TopKPredictionTaskGenerator : public PredictionBase {
    public:
        TopKPredictionTaskGenerator(const DatasetBase* data, std::shared_ptr<const Model> model, long K);

        void update_model(std::shared_ptr<const Model> model);

        void run_tasks(long begin, long end, thread_id_t thread_id) override;

        [[nodiscard]] long num_tasks() const override;
        void prepare(long num_threads, long chunk_size) override;
        void finalize() override;

        [[nodiscard]] const PredictionMatrix& get_top_k_values() const { return m_TopKValues; }
        [[nodiscard]] const IndexMatrix& get_top_k_indices() const { return m_TopKIndices; }

        [[nodiscard]] const std::array<std::int64_t, 4>& get_confusion_matrix() const { return m_ConfusionMatrix; }

        static constexpr const int TRUE_POSITIVES  = 0;
        static constexpr const int FALSE_POSITIVES = 1;
        static constexpr const int TRUE_NEGATIVES  = 2;
        static constexpr const int FALSE_NEGATIVES = 3;
    private:
        long m_K;

        PredictionMatrix m_TopKValues;
        IndexMatrix m_TopKIndices;

        std::vector<PredictionMatrix> m_ThreadLocalPredictionCache;
        std::vector<PredictionMatrix> m_ThreadLocalTopKValues;
        std::vector<IndexMatrix> m_ThreadLocalTopKIndices;
        std::vector<std::array<std::int64_t, 4>> m_ThreadLocalConfusionMatrix;

        std::vector<std::vector<long>> m_GroundTruth;
        std::array<std::int64_t, 4> m_ConfusionMatrix;
    };
}
#endif //DISMEC_PREDICTION_H

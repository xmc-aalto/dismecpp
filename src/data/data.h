// Copyright (c) 2021, Aalto University, developed by Erik Schultheis
// All rights reserved.
//
// SPDX-License-Identifier: MIT

#ifndef DISMEC_DATA_H
#define DISMEC_DATA_H

#include <memory>
#include "matrix_types.h"
#include "data/types.h"
#include "utils/eigen_generic.h"

namespace dismec {
    class DatasetBase {
    public:
        virtual ~DatasetBase() = default;

        /// get a shared pointer to the (immutable) feature data
        [[nodiscard]] std::shared_ptr<const GenericFeatureMatrix> get_features() const;

        /// get a shared pointer to mutable feature data. Use with care.
        [[nodiscard]] std::shared_ptr<GenericFeatureMatrix> edit_features();

        /// Get the total number of features, i.e. the number of columns in the feature matrix
        [[nodiscard]] std::size_t num_features() const noexcept;

        /// Get the total number of instances, i.e. the number of rows in the feature matrix
        [[nodiscard]] std::size_t num_examples() const noexcept;

        /// Gets the total number of different labels in the dataset.
        /// TODO call this num_classes instead?
        [[nodiscard]] virtual long num_labels() const noexcept = 0;

        /// Gets the number of instances where label `id` is present (=+1)
        /// Throws std::out_of_bounds, if id is not in `[0, num_labels())`.
        [[nodiscard]] virtual std::size_t num_positives(label_id_t id) const;

        /// Gets the number of instances where label `id` is absent (=-1)
        /// Throws std::out_of_bounds, if id is not in `[0, num_labels())`.
        [[nodiscard]] virtual std::size_t num_negatives(label_id_t id) const;

        /// Gets the label vector (encoded as dense vector with elements from {-1, 1}) for the `id`'th class.
        /// Throws std::out_of_bounds, if id is not in `[0, num_labels())`.
        [[nodiscard]] std::shared_ptr<const BinaryLabelVector> get_labels(label_id_t id) const;

        /// Gets the label vector (encoded as dense vector with elements from {-1, 1}) for the `id`'th class.
        /// The weights will be put into the given `target` buffer.
        /// Throws std::out_of_bounds, if id is not in `[0, num_labels())`.
        virtual void get_labels(label_id_t id, Eigen::Ref<BinaryLabelVector> target) const = 0;
    protected:
        explicit DatasetBase(SparseFeatures x);
        explicit DatasetBase(DenseFeatures x);

        // features
        std::shared_ptr<GenericFeatureMatrix> m_Features;
    };

    /*! \class BinaryData
     *  \brief Collects the data related to a single optimization problem.
     *  \details This contains the entire training set as sparse features,
     *  so we require that this fit into memory. The lables are strored as
     *  a dense vector.
     */
    class BinaryData : public DatasetBase {
    public:
        BinaryData(SparseFeatures x, std::shared_ptr<BinaryLabelVector> y) :
                DatasetBase(std::move(x)), m_Labels(std::move(y))
        {

        }

        [[nodiscard]] long num_labels() const noexcept override;
        void get_labels(label_id_t i, Eigen::Ref<BinaryLabelVector> target) const override;
    private:

        // targets
        std::shared_ptr<BinaryLabelVector> m_Labels;
    };


    class MultiLabelData : public DatasetBase {
    public:
        MultiLabelData(SparseFeatures x, std::vector<std::vector<long>> y) :
                DatasetBase(x.markAsRValue()), m_Labels(std::move(y)) {

        }

        MultiLabelData(DenseFeatures x, std::vector<std::vector<long>> y) :
            DatasetBase(std::move(x)), m_Labels(std::move(y)) {
        }

        [[nodiscard]] long num_labels() const noexcept override;
        void get_labels(label_id_t label, Eigen::Ref<BinaryLabelVector> target) const override;

        // these are faster than the default implementation
        [[nodiscard]] std::size_t num_positives(label_id_t id) const override;
        [[nodiscard]] std::size_t num_negatives(label_id_t id) const override;

        [[nodiscard]] const std::vector<long>& get_label_instances(label_id_t label) const;

        void select_labels(label_id_t start, label_id_t end);

        [[nodiscard]] const std::vector<std::vector<long>>& all_labels() const { return m_Labels; }
    private:
        // targets: vector of vectors of example ids: if label i is present in example j, `j \in then m_Labels[i]`
        std::vector<std::vector<long>> m_Labels;
    };
}

#endif //DISMEC_DATA_H

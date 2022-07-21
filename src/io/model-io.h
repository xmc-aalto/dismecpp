// Copyright (c) 2021, Aalto University, developed by Erik Schultheis
// All rights reserved.
//
// SPDX-License-Identifier: MIT

#ifndef DISMEC_MODEL_IO_H
#define DISMEC_MODEL_IO_H

#include <Eigen/Dense>
#include <vector>
#include <memory>
#include <filesystem>
#include <optional>
#include <future>
#include "fwd.h"
#include "data/types.h"
#include <boost/iterator/iterator_adaptor.hpp>

/*! \page model-data Model data format
 *  Models are saved in multiple files. One file contains metadata, whereas the weights are stored in separate files.
 *  We support multiple formats for storing the weights, but the metadata file has always the same structure.
 *
 *  \section model-data-meta Metadata File
 *  The metadata is saved as json. It contains the following keys
 *   - `"num-features"`: Number of features, i.e. the size of a single weight vector
 *   - `"num-labels"`: Number of labels, i.e. the number of weight vectors.
 *   - `"date"`: Contains the data and time when the file was created.
 *   - `"weights"`: Contains info on where the weights are stored. This is an array of dicts, where each entry
 *   corresponds to one weights file. Each weights file stores a contiguous subset (as seen over labels) of the weights.
 *   Each entry into the vector has the keys `"first"`, which is the index of the first label in the file, `"count"`
 *   which is the number of weight vectors, `"file"` which is the file name relative to the metadata file and
 *   `"weight-format"`, which specifies the format in which the weights are saved.
 *
 *   There are several advantages to allowing the weights to be distributed over multiple files. For one, it allows
 *   partial saves, e.g. if one wants to do checkpointing. While this could be achieved by simply appending for the
 *   human-readable text formats, it may require rewriting the entire file in other settings, e.g. when using compressed
 *   files.
 *   Secondly, in a distributed setting with a shared, networked file system, we can reduce the amount of data transfer.
 *   In distributed training, each worker can save its own weight files, and the main program only needs to be notified
 *   that it should update the metadata file. In distributed prediction, each worker only needs to load the weights
 *   for the labels it is responsible for, and does not have to parse the entire weight file only to discard most of
 *   the weights.
 *
 *   I am planning to add the following additional metadata:
 *    - version: To specify which version of the library was used when creating the file
 *    - training: A dict which contains information about the training process
 *    - custom: Guaranteed to never be written by the library, so save to use for others.
 *
 *  \section model-data-dense-txt Dense Text Format
 *  Writes all the weights as space-separated numbers. Each line in the file corresponds to a single weight vector.
 *  Note that this means that the *rows* in the text file correspond to the *columns* in the weight matrix. This is
 *  to make it easier to read only a subset of weight vectors from the text file.
 *
 *  Writing is implemented in \ref io::model::save_dense_weights_txt(), and reading in
 *  \ref io::model::load_dense_weights_txt()
 *
 *  The advantage of this format is that it is human readable and very portable. However, it is not efficient
 *  both in terms of storage and in terms of read/write performance. This can be influenced to some degree
 *  by setting adjusting the precision, i.e. the number of digits written.
 *
 *  \section model-data-sparse-txt Sparse Text Format
 *  Writes all the weights exceeding a given threshold in a sparse format. Each row corresponds to one weight vector,
 *  and consists of `index:value` pairs separated by whitespace. Here, index is the 0-based position in the weight
 *  vector and value its corresponding value.
 *
 *  This format is human readable and portable, and may be much more space efficient than the dense text format. Storage
 *  requirements can be adjusted by setting the precision with which the nonzero weights are written, and by setting
 *  the threshold below which weights are culled.
 *
 *  Writing is implemented in \ref io::model::save_as_sparse_weights_txt(), which can also save dense models by culling
 *  weights below a specified threshold. Reading is implemented through \ref io::model::load_sparse_weights_txt().
 *
 *  \section model-data-dense-npy Dense Numpy Format
 *  Writes the weights as a matrix to a `.npy` file. The data is written in row-major
 *  format to allow loading a subset of the labels by reading contiguous parts of the file.
 *  Since the output is binary, we operate directly on a stream-buffer here.
 *
 *  This format is more space efficient than the text format, and also has much lower computational overhead, since it
 *  does not require any number parsing or formatting. As a rough estimate, for eurlex (~20M weights) saving as (dense)
 *  text takes about 8.8 seconds; as npy it takes only about 200 ms. The file size decreases from 230 MB to 80 MB.
 *  Similar (though slightly less) speedups appear for loading the model.
 *
 *  Writing is implemented in \ref io::model::save_dense_weights_npy() and reading is done using
 *  \ref io::model::load_dense_weights_npy.
 */

namespace dismec::io
{

    //! namespace for all model-related io functions.
    namespace model
    {
        using dismec::model::Model;
        using std::filesystem::path;

        /*!
         * \brief Describes the format in which the weight data has been saved.
         */
        enum class WeightFormat {
            DENSE_TXT  = 0,      //!< \ref model-data-dense-txt
            SPARSE_TXT = 1,      //!< \ref model-data-sparse-txt
            DENSE_NPY  = 2,      //!< \ref model-data-dense-npy
            NULL_FORMAT = 3      //!< This format exists for testing purposes only, and indicates that the weights
                                 //!< will not be saved.
        };

        /// Gets the eighs
        WeightFormat parse_weights_format(std::string_view name);
        const char* to_string(WeightFormat format);

        struct SaveOption {
            int Precision  = 6;     //!< Precision with which the labels will be saved
            double Culling = 0;     //!< If saving in sparse mode, threshold below which weights will be omitted.
            int SplitFiles = 4096;  //!< Maximum number of weight vectors per file.
            WeightFormat Format = WeightFormat::DENSE_TXT;      //!< Format in which the weights will be saved.
        };

        // \todo should we by default overwrite files, or refuse?
        /*!
         * \brief Saves a complete model to a file.
         * \details This function saves a complete model using the specified options, to `target_file`. This function
         * cannot be used to save a partial model!
         * \param target_file The path to the file where the metadata will be saved. Will also be used as prefix for the
         * names of the weight files. If the files already exist, they will be overwritten.
         * \param model The model to be saved.
         * \param options Additional options to influence how the save file is generated.
         */
        void save_model(const path& target_file, const std::shared_ptr<const Model>& model, SaveOption options);


        std::shared_ptr<Model> load_model(path source);

        /*!
         * \brief Collect the data about a weight file.
         * \details A (potentially partial) saved model contains a list of `WeightFileEntry`s that point to
         * the files where the weights are stored, and contain meta information so they can be correctly loaded
         * into the model.
         */
        struct WeightFileEntry {
            label_id_t First;
            long Count;
            std::string FileName;
            WeightFormat Format;
        };


        /*!
         * \brief This class is used as an implementation detail to capture the common code of \ref PartialModelSaver
         * and \ref PartialModelLoader.
         */
        class PartialModelIO {
        public:
            /*!
             * \brief Gets the total number of labels.
             * \details If this value is unknown (i.e. if there hasn't been one model submitted yet), -1 is returned.
             * For a model reader, this is always a positive value.
             */
            [[nodiscard]] long num_labels() const noexcept { return m_TotalLabels; }

            /*!
             * \brief Gets the total number of features.
             * \details If this value is unknown (i.e. if there hasn't been one model submitted yet), -1 is returned.
             * For a model reader, this is always a positive value.
             */
            [[nodiscard]] long num_features() const noexcept { return m_NumFeatures; }

        protected:
            PartialModelIO() = default;
            ~PartialModelIO() = default;

            void read_metadata_file(const path& meta_file);

            long m_TotalLabels = -1;
            long m_NumFeatures = -1;

            std::vector<WeightFileEntry> m_SubFiles;
            using weight_file_iter_t = std::vector<WeightFileEntry>::const_iterator;

            /*!
             * \brief Inserts a new sub-file entry into the metadata object.
             * \param data The specifications of the sub file to be added.
             * \throw std::logic_error if `data` has weights for labels that overlap with already existing weights.
             */
            void insert_sub_file(const WeightFileEntry& data);

            /*!
             * \brief Gets an iterator into the weight-file list that points to the first element whose starting
             * label is larger than  or equal to `pos`.
             * \details Assuming the partial model contains the weights for the label ranges `[20, 50)` and `[100, 150)`.
             * Then `label_lower_bound(label)` will return an iterator to the first element for all `label` values
             * in the range `[00, 20]`, the second for the range `(20, 100]` and an iterator to end for all others. This
             * corresponds to the position in front of which a new sub-model for label `pos` would have to be inserted.
             */
            [[nodiscard]] weight_file_iter_t label_lower_bound(label_id_t pos) const;
        };

        /*!
         * \brief Manage saving a model consisting of multiple partial models.
         * \details In large-scale situations, model training will not produce a single, complete model, but instead
         * we will train multiple partial models (either successively, or in a distributed parallel fashion). In that
         * case, the `save_model` function is not suitable, and this class has to be used.
         *
         * The work flow looks like this: First, a model saver instance is created. You can pass in the name
         * of the metadata file, and additional options as \ref SaveOption.
         * \code
         * SaveOption options;
         * PartialModelSaver saver(target_file, options)
         * \endcode
         * Then the partial models will be added. The first model will be used to determine the total number of
         * labels and features, and all further partial models added will be verified to be compatible. Note that
         * \ref PartialModelSaver::add_model() does write the weight files, but it does not yet update the
         * metadata file.
         * \code
         * for(auto& partial_model : generator_of_partial_models) {
         *     saver.add_model(partial_model);
         * }
         * \endcode
         * After all partial models have been added, you should call \ref PartialModelSaver::finalize(). This method
         * first verifies that you have in fact submitted all partial models, i.e. that all together the weight vector
         * for each label has been saved. If that is the case, it updates the metadata file.
         * If you want to checkpoint the models you have saved up to a certain point, but have not yet saved all
         * weights, you can instead use the \ref PartialModelSaver::update_meta_file() function. This will only update
         * the meta-data file, but not check that all weights are present. You can continue writing more partial models
         * to that PartialModelSaver. Another, typical use case is that you want to continue adding to the save file
         * in another run of the program. In that case, you can load the partial save file by passing true as the
         * `load_partial` argument to the constructor.
         * \code
         * saver.update_meta_file();        // model is still incomplete
         * // ...
         * PartialModelSaver continued_save(target_file, options, true);    // load the partial save file
         * continued_save.add_model(missing_parts);                         // add more partial models
         * continued_save.finalize();                                       // call finalize once it is complete.
         * //
         * \endcode
         */
        class PartialModelSaver : public PartialModelIO {
        public:
            /*!
             * \brief Create a new PartialModelSaver
             * \param target_file File name of the metadata file. Also used as the base name for automatically generated
             * weight file name.
             * \param options Options that will be used for saving the weights of all partial models that will be
             * submitted to `add_model`. The \ref SaveOption::SplitFiles parameter is ignored, though, since file splits
             * are done explicitly through the partial models.
             * \param load_partial If this is set to true, then it is expected that `target_file` already exists, and
             * the corresponding metadata will be loaded (but not the weights). This allows to add more weights to an
             * unfinished save file.
             * \note When continuing from an existing checkpoint, we assume that all existing partial weights are valid.
             * If that is not the case, the `PartialModelSaver` will still be able to add more weights (that don't
             * overlap with the existing ones), but actually loading the resulting model file will be impossible.
             */
            PartialModelSaver(path target_file, SaveOption options, bool load_partial=false);

            /*!
             * \brief Adds the weights of a partial model asynchronously.
             * \details Saves the weights of `model` in a weights file and appends that file to the internal list of
             * weight files. This does not update the metadata file. If this method is called for the first time, the
             * total number of labels and features is extracted. All subsequent calls verify that the partial models
             * have the same number of labels and features.
             * This function operates asynchronously, using std::async to launch the actual writing of the weights.
             *
             * \param model The model whose weights to save.
             * \param file_path If given, this will be used as the file path for the weights file. Otherwise, an
             * automatically generated file name is used.
             * \throw std::logic_error if `model` is incompatible because it has a different number of features/labels
             * than the other partial models, or if there is an overlap with already saved weights.
             * \return A future that becomes ready after the weight file has been written. Its value is the new
             * weight file entry that has been added to the list of weight files.
             */
            std::future<WeightFileEntry> add_model(const std::shared_ptr<const Model>& model,
                                                   const std::optional<std::string>& file_path={});

            using PartialModelIO::insert_sub_file;

            /*!
             * \brief Updates the metadata file.
             * \details This ensures that all weight files that have been created due to `add_model` calls will be
             * listed in the metadata file. Use this function to checkpoint partial saving. If all partial models have
             * been added, call \ref finalize() instead, which also verifies the completeness of the data.
             */
            void update_meta_file();

            /*!
             * \brief Checks that all weights have been written and updates the metadata file.
             * \details This function checks that the `PartialModelSaver` has received weight vectors for every label.
             * If that is true, it updates the metadata file.
             * \throw std::logic_error If there are missing weight vectors.
             * \todo Do we need this function? It seems like a nice idea to have a check if everything is done, but
             * currently we don't use it.
             */
            void finalize();

            /*!
             * \brief Get an interval labels for which weights are missing.
             * \details If all weights are present, the returned pair is `(num_weights, num_weights)`. Otherwise, it
             * is a half-open interval over label ids for which the `PartialModelSaver` doesn't have weights available.
             */
             [[nodiscard]] std::pair<label_id_t, label_id_t> get_missing_weights() const;

             /*!
              * \brief Checks if there are any weight vectors for the given interval.
              * \details This can be used to make sure that training is only run for label ids for which there is no
              * weight vector yet.
              */
              [[nodiscard]] bool any_weight_vector_for_interval(label_id_t begin, label_id_t end) const;

        private:
            SaveOption m_Options;
            path m_MetaFileName;
        };

        /*!
         * \brief This class allows loading only a subset of the weights of a large model.
         * \details In a setting with an extreme number of labels, it might not be possible
         * to load the entire set of weights into the memory at once.
         */
        class PartialModelLoader : public PartialModelIO {
        public:
            enum ESparseMode {
                DEFAULT,
                FORCE_SPARSE,
                FORCE_DENSE
            };

            /*!
             * \brief Create a new PartialModelLoader for the given metadata file.
             * \details This operation will only open and parse the metadata file, but no
             * weights will be read in.
             * \param meta_file Path to the metadata file.
             * \param mode Should the returned models store their weights in a sparse matrix, a dense matrix, or
             * use the representation as given by the weights file (DEFAULT).
             */
            explicit PartialModelLoader(path meta_file, ESparseMode mode=DEFAULT);

            /*!
             * \brief The path to the metadata file.
             */
            const path& meta_file_path() const { return m_MetaFileName; }

            /*!
             * \brief Loads part of the model.
             * \details If the desired interval `[label_begin, label_end)` does not match directly with the
             * weight files, a model that encompasses more weights will be used. Each weights file which has some
             * overlap with the interval will be loaded entirely.
             * \param label_begin First label whose weight is guaranteed to be part of the returned model.
             * \param label_end Element id after the last weight vector that is included in the returned model.
             * \return A model that contains weight at least for the interval `[label_begin, label_end)`.
             */
            [[nodiscard]] std::shared_ptr<Model> load_model(label_id_t label_begin, label_id_t label_end) const;

            /*!
             * \brief Loads the model from the weights file at the given index.
             * \param index The index of the sub-model to be loaded.
             */
            [[nodiscard]] std::shared_ptr<Model> load_model(int index) const;

            /*!
             * \brief Validates that all weight files exist.
             */
            bool validate() const;

            /// Returns the number of availabel weight files.
            [[nodiscard]] long num_weight_files() const;

            struct SubModelRangeSpec {
                weight_file_iter_t FilesBegin;
                weight_file_iter_t FilesEnd;
                label_id_t LabelsBegin{0};
                label_id_t LabelsEnd{0};
            };

            [[nodiscard]] SubModelRangeSpec get_loading_range(label_id_t label_begin, label_id_t label_end) const;
        private:
            path m_MetaFileName;
            ESparseMode m_SparseMode;
        };
    }

    using model::WeightFormat;
    using model::SaveOption;
    using model::save_model;
    using model::load_model;
    using model::PartialModelSaver;
    using model::PartialModelLoader;
}

#endif //DISMEC_MODEL_IO_H

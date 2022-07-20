// Copyright (c) 2021, Aalto University, developed by Erik Schultheis
// All rights reserved.
//
// SPDX-License-Identifier: MIT

#ifndef DISMEC_STATS_BASE_H
#define DISMEC_STATS_BASE_H

#include "matrix_types.h"
#include <memory>
#include <string>
#include <nlohmann/json_fwd.hpp>

namespace dismec::stats {
    class StatisticsCollection;

    /*!
     * \brief A tag container combines a name with a shared pointer, which points to the tag value
     * \details This is a lightweight wrapper around a shared pointer, which is used to manage tags.
     * It can be in an empty and a full state. In the empty state, it has only a name, but not associated
     * value (i.e. m_Value is nullptr), and in the full state it has both. Because updating and reading
     * the value are expected to happen in performance-critical parts of the code, there is no check
     * for empty state in release mode.
     *
     * Any statistics the needs a tag should create empty tags with the corresponding name during its creation
     * phase, and then fill in the full tag from the given `StatisticsCollection` during the `setup()` call.
     *
     * For now, only integer values are accepted as tag value.
     */
    class TagContainer {
    public:
        /// returns the name of the associated tag
        [[nodiscard]] const std::string& get_name() const { return m_Name; }

        /// Returns the current value of the tag. Requires the container to not be empty.
        [[nodiscard]] int get_value() const {
            assert(!is_empty());
            return *m_Value;
        }

        // Returns whether the container is currently empty.
        [[nodiscard]] bool is_empty() const { return m_Value == nullptr; }

        /// Updates the value of the tag. Requires the container to not be empty.
        void set_value(int value) {
            assert(!is_empty());
            *m_Value = value;
        }

        static TagContainer create_empty_container(std::string name) {
            return TagContainer(std::move(name), nullptr);
        }
        static TagContainer create_full_container(std::string name) {
            return TagContainer(std::move(name), std::make_shared<int>());
        }

    private:
        explicit TagContainer(std::string name, std::shared_ptr<int> val) : m_Name(std::move(name)), m_Value( std::move(val) ) {}

        std::string m_Name;
        std::shared_ptr<int> m_Value;
    };

    /// TODO maybe we should solve this with a variant which does the dispatch of expected type and tag
    class Statistics {
    public:
        virtual ~Statistics() = default;

        // this overload is provided to prevent the int -> long / float ambiguity
        void record(int integer) { record(long(integer)); }

        virtual void record(long integer) { throw std::logic_error("Not implemented"); }
        virtual void record(real_t real) { throw std::logic_error("Not implemented"); }
        virtual void record(const DenseRealVector& vector) { throw std::logic_error("Not implemented"); }

        [[nodiscard]] virtual std::unique_ptr<Statistics> clone() const = 0;

        /*!
         * \brief This function has to be called before the `Statistics` is used to collect data for the first time.
         * \details This will look up any tags that might be used within the statistics in `source`.
         * \throws std::runtime_error if a tag is required by the statistics but cannot be found in `source`.
         * \param source The statistics collection from which the tags can be read.
         */
        virtual void setup(const StatisticsCollection& source) {  };

        /*!
         * \brief Merges this statistics of another one of the same type and settings.
         * \details This operation is used to perform the reduction of the thread-local statistics into a single
         * global statistics. For this to be possible, the merged statistics need to have not only the same type,
         * but also the same settings (e.g. number of bins in a histogram). This is true for the thread local copies,
         * which are clones of one another.
         * \param other The statistics object to be merged into this. Must be of same type and have the same settings,
         * e.g. a clone of this.
         */
        virtual void merge(const Statistics& other) = 0;

        /// Converts the statistics current value into a json object.
        [[nodiscard]] virtual nlohmann::json to_json() const = 0;
    };

    /*!
     * \brief Generates a `stats::Statistics` object based on a json configuration.
     * \param source A json object the describes the statistics.
     * \todo document the json format.
     */
    std::unique_ptr<stats::Statistics> make_stat_from_json(const nlohmann::json& source);

    /*!
     * \brief Helper class for implementing `Statistics` classes.
     * \tparam Derived CRTP parameter. Needs to be a final class.
     * \details This class provides two default implementations to ease writing derived classes. The `merge()`
     * function checks that the other given statistics is of the same type, and then calls an overload of `merge()`
     * that expects a `Derived` reference.
     * The `record()` function for vector parameters is implemented so that it called the scalar `record()` for each
     * vector component. Note that we cannot efficiently implement this default behaviour in `Statistics`, because then
     * each record call would be virtual. Since we do know the `Derived` type here, no virtual call is necessary.
     *
     * For this to work, we require that `Derived` be a final class.
     */
    template<class Derived>
    class StatImplBase : public Statistics {
    public:
        void merge(const Statistics& other) override {
            static_assert(std::is_final_v<Derived>, "Derived needs to be declared final, because further derived classes would break the merge code.");
            static_cast<Derived*>(this)->merge(dynamic_cast<const Derived&>(other));
        }

        void record(const DenseRealVector& vector) override {
            for(int i = 0; i < vector.size(); ++i) {
                static_cast<Derived*>(this)->record(vector.coeff(i));
            }
        }
    };

}

#endif //DISMEC_STATS_BASE_H

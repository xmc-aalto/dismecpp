// Copyright (c) 2021, Aalto University, developed by Erik Schultheis
// All rights reserved.
//
// SPDX-License-Identifier: MIT

#ifndef DISMEC_COLLECTION_H
#define DISMEC_COLLECTION_H

#include "stats_base.h"
#include <vector>
#include <type_traits>
#include <memory>
#include "matrix_types.h"
#include "stat_id.h"

namespace stats
{
    /*!
     * \brief This class manages a collection of named `Statistics` objects.
     * \details The functionality of this class is designed to be performance efficient, but nonetheless it should not
     * be used in the inner loops. It is further designed so that statistics collection can be disabled at runtime,
     * in which case the performance impact should be almost negligible.
     *
     * This is achieved by assigning each named statistics also an integer id, which is used in the performance critical
     * code path, which consists of the `is_enabled()` and `record()` functions. The check whether a particular statistics
     * shall be collected is performed by `is_enabled()`, which is a simple lookup in a boolean vector, and defined
     * in the header so that it can be inlined. The `record()` functions first check if the statistics is enabled, and
     * if not no further action is performed. They are also defined in the header, so this check will get inlined too. In
     * all, if statistics are disabled, each call to record should boil down to an inlined access to a boolean vector.
     * For anything but the innermost loops, this should be fast enough.
     *
     * The actual gathering of the statistics requires an additional vector lookup, followed by a virtual function call
     * into the actual statistics implementation. This should also not impose too much of a performance problem in most
     * cases. A different situation arises if the statistics we want to gather is not calculated as part of the regular
     * computations, and is expensive to calculate. For that purpose, there exists an overload of `record()` that takes
     * a callable as an argument, which is evaluated only if the statistics is enabled.
     *
     * \attention The definition of the statistics integral identifiers is the responsibility of the user of this class.
     * Since they only have to be unique with respect to a single `StatisticsCollection`, my strategy is to define the
     * ids as constants at the beginning of the implementation file which uses the collection.
     *
     * For extracting the statistics, no knowledge of the ids is required. This is because we consider that process to
     * be non-performance-critical, and as such we can perform lookups based on the statistic's name. This means that
     * often that the IDs can be considered an implementation detail of the code that records the statistics, which can
     * be hidden from consumer code.
     */
    class StatisticsCollection {
    public:
        // declaration and registration of stats
        /*!
         * \brief Declares a new statistics. Defines the corresponding index and name.
         * \details The new statistics is disabled by default, and no `Statistics` object that gathers its values is
         * associated. This can be changed by calling `register_stat()`. The indices and names need to be unique.
         * \param index Index of the new statistics.
         * \param meta Metadata of the new statistics.
         * \note A current limitation of the implementation is that it assumes that statistics are declared in order of
         * their ids, which are assumed to be contiguous.
         * \throws std::invalid_argument If a stat with the same name or id already has been declared.
         */
        void declare_stat(stat_id_t index, StatisticMetaData meta);

        /*!
         * \brief Declares a new tag value.
         * \param index The unique index to be used for the tag.
         * \param name The unique name to be used for the tag.
         */
        void declare_tag(tag_id_t index, std::string name);

        /*!
         * \brief Registers a `Statistics` object to the named slot.
         * \details This function can also be used to unregister a `Statistics` by providing `nullptr` as the second
         * argument. In that case, the statistics is automatically disabled, otherwise it is enabled.
         * \param name Name for which `stat` should be registered.
         * \param stat The `Statistics` object to register.
         * \throws std::invalid_argument If no slot `name` has been declared, or if there already is a registered stat
         * for the slot and `stat` is not `nullptr`.
         */
        void register_stat(const std::string& name, std::unique_ptr<Statistics> stat);

        // access stats
        //! \brief Gets a vector with the declarations for all statistics.
        [[nodiscard]] const std::vector<StatisticMetaData>& get_statistics_meta() const { return m_MetaData; }

        /*!
         * \brief Returns the `Statistics` object corresponding to the slot `name`.
         * \throws std::invalid_argument if `name` has not been declared, or has no assigned `Statistics` object.
         */
        [[nodiscard]] const Statistics& get_stat(const std::string& name) const;

        /*!
         * \brief Gets the tag with the given name.
         * \details Returns a `TagContainer` whose value is a pointer to the interval value stored in this
         * collection. Thus, any `Statistics` object that needs a tag can lookup the tag by using the name
         * in its setup phase, and then during recording only needs to use the `TagContainer` to get the current
         * value, which is much faster.
         * \throws std::out_of_range if `name` has neither been declared nor provided by another collection.
         */
        [[nodiscard]] TagContainer get_tag_by_name(const std::string& name) const;

        //! \brief Gets a vector which contains all the tags owned by this collection.
        [[nodiscard]] const std::vector<TagContainer>& get_all_tags() const { return m_TagValues; }

        /*!
         * \brief Registers all the tags of the other collection as read-only tags in this collection.
         * \details Read-only tags can be accessed by `get_tag_by_name()`, so that statistics can use them in their
         * setup, but cannot be modified by `set_tag()`. They don't even have an associated tag_id inside this
         * collection.
         */
         void provide_tags(const StatisticsCollection& other);

        // enable / disable
        /*!
         * \brief Explicitly enable the collection of statistics for the given index.
         * \details Collection of data can only be enabled for statistics for which a corresponding `Statistics`
         * object has been provided using `register_stat()`.
         * \throws std::logic_error If you try to enable collection of a statistic without a registered `Statistics` object.
         * \sa disable(), is_enabled()
         */
        void enable(stat_id_t stat);

        /// \brief Explicitly disable the collection of statistics for the given index.
        /// \sa enable(), is_enabled()
        void disable(stat_id_t stat);

        /*!
         * \brief Quickly checks whether collection of data is enabled for the given statistics.
         * \details This function is defined in the header so it can be inlined.
         * \param stat The id of the statistics for which the check if performed.
         * \sa enable(), disable()
         */
        [[nodiscard]] bool is_enabled(stat_id_t stat) const {
#ifdef NDEBUG
            return m_Enabled[stat.to_index()];
#else
            // additional checking for debug builds.
            return m_Enabled.at(stat.to_index());
#endif
        }

        /*!
         * \brief Enables collecting data based on the name of a statistics.
         * \throws std::invalid_argument if not such stat exists.
         */
        void enable(const std::string& stat);

        /*!
         * \brief Disables collecting data based on the name of a statistics.
         * \throws std::invalid_argument if not such stat exists.
         */
        void disable(const std::string& stat);

        /*!
         * \brief Returns whether a stat with the given name is declared
         */
        [[nodiscard]] bool has_stat(const std::string& name) const;

        /*!
         * \brief Checks whether gathering data is enabled based on the statistic's name.
         * \details Because this function has to perform the lookup for the name, it is much slower than
         * `is_enabled()`. Therefore, even though it performs the exact same function, it has not been
         * provided an overload, but given an extra, longer name to prevent accidental performance problems.
         */
        [[nodiscard]] bool is_enabled_by_name(const std::string& name) const;

        // recording
        /*!
         * \brief Records an already computed value.
         * \tparam T The type of the recorded value. Needs to be compatible with `Statistics::record()`.
         * \param stat The id of the stat for which the value is recorded.
         * \param value The actual data.
         * \sa Statistics::record()
         */
        template<class T, std::enable_if_t<!std::is_invocable_v<T>, bool> = true>
        void record(stat_id_t stat, T&& value) {
            if(is_enabled(stat)) {
                do_record(stat.to_index(), std::forward<T>(value));
            }
        }

        /*!
         * \brief Records a value that is computed only when needed.
         * \tparam F A callable type. Will typically be the type of same lambda.
         * \param stat The id of the stat to be recorded.
         * \param callable A callable that produces a value that is compatible with `Statistics::record()`. Will
         * only be called if the statistics has been enabled.
         * \details A typical usage example would look like this
         * \code
         * collection.record(id, [&](){ return expensive_function(intermediate); });
         * \endcode
         * Here, `intermediate` is a value that is calculated as part of the regular program flow,
         * but for recording statistics we need `expensive_function(intermediate)`.
         * The invocation of the `callable` itself is wrapped into an immediately invoked lambda annotated as cold,
         * so that the expensive calculation will not get inlined into the main program flow and slow down the code
         * path when the statistics is disabled.
         * \sa Statistics::record().
         */
        template<class F, std::enable_if_t<std::is_invocable_v<F>, bool> = true>
        void record(stat_id_t stat, F&& callable) {
            if(is_enabled(stat)) {
                // we wrap callable into an immediately invoked lambda, which is annotated as cold, so that
                // the code-path with stat disabled is optimized.
                do_record(stat.to_index(), [&]() __attribute__((cold)) { return callable(); }());
            }
        }

        /*!
         * \brief Sets the tag to the given integer value
         */
         void set_tag(tag_id_t tag, int value) {
#ifdef NDEBUG
            m_TagValues[tag.to_index()].set_value(value);
#else
            m_TagValues.at(tag.to_index()).set_value(value);
#endif
         }


    private:
        template<class T>
        void do_record(int index, T&& value) {
            m_Statistics[index]->record(std::forward<T>(value));
        }

        std::vector<bool> m_Enabled;
        std::vector<StatisticMetaData> m_MetaData;
        std::vector<std::unique_ptr<Statistics>> m_Statistics;

        std::vector<TagContainer> m_TagValues;
        std::unordered_map<std::string, TagContainer> m_TagLookup;
    };
}

#endif //DISMEC_COLLECTION_H

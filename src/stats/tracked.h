// Copyright (c) 2021, Aalto University, developed by Erik Schultheis
// All rights reserved.
//
// SPDX-License-Identifier: MIT

#ifndef DISMEC_STATS_TRACKED_H
#define DISMEC_STATS_TRACKED_H

#include <memory>
#include "stat_id.h"

namespace dismec::stats {
    class StatisticsCollection;
    class Statistics;
    class ScopeTimer;

    /*!
     * \brief A base class to be used for all types that implement some for of statistics tracking
     * \details Provides the protected methods `record()`, `declare_stat()`, and `make_timer()` that
     * relay the corresponding action to the underlying `stats::StatisticsCollection`, to deriving classes.
     * Publicly available is only the option to register a `Statistics` object for a given statistics name,
     * via `register_stat()`, and read-only access to the accumulator.
     *
     * The implementation of Tracked is a bit tricky because it is trying to achieve too things:
     *   - Since this is expected to be used as base class in many fundamental types of the library, we want to
     *   keep the include dependencies as little as possible. In order to define the interface, it is not necessary
     *   to know any details of the statistics gathering implementation.
     *   - Introduce as little overhead as possible. This is only relevant to the statistics gathering functions
     *   `record()` and `make_timer()` which will appear in performance-critical code. For that reason, they are defined
     *   inline.
     *
     * These two goals are at odds in a naive implementation: In order to define an inline function, we cannot have an
     * incomplete type that is accessed or returned. This is circumvented by defining these functions as templates,
     * in such a way that the problematic call is (nominally) dependent on one of the template parameters, and thus
     * the types only need be available when the function is actually called. This means that the `collection.h`
     * and `timer.h` headers need only be included in the implementation files.
     *
     * Note that this class is non-virtual. It is used as a convenient way of adding functionality to other types. It
     * adds very convenient protected and public member functions. Its destructor is defined protected, so that one
     * does not accidentally try to use it polymorphically.
     */
    class Tracked {
    public:
        /// Default constructor, creates the internal `stats::StatisticsCollection`.
        Tracked();

        /*!
         * \brief Registers a tracker for the statistics `name`.
         * \note This function will not be called in performance-critical code, and is thus not defined inline.
         * \sa stats::StatisticsCollection::register_stat
         */
        void register_stat(const std::string& name, std::unique_ptr<Statistics> stat);

        /*!
         * \brief Gets an ownership-sharing reference to the `StatisticsCollection`.
         * \details We return a `std::shared_ptr`, instead of a plain reference, to accommodate the usage scenario
         * in which some global object collects references to all the (thread_local) individual tracked objects,
         * and combines their results after the threads have ended. By storing as a `std::shared_ptr`, we can just
         * collect all sub-objects during the setup phase, and do the merging after the threads have finished, without
         * worrying about the lifetime of the involved objects.
         * If we were to merge as each thread finishes and we are sure the objects are still alive, the
         * (possibly costly) merging process would need to be mutex protected.
         * To prevent this, we use shared ownership, but allow the global object only read access.
         */
        [[nodiscard]] std::shared_ptr<StatisticsCollection> get_stats() const {
            return m_Collection;
        }
    private:
        /*!
         * \brief Given an object T, and some dummy template parameters Args, returns T unchanged.
         * \details The purpose of this function is to get a dependent value from a direct value, so
         * that any calls on it will only be looked at in the second phase of template instantiation.
         */
        template<class T, class... Args>
        T& make_dependent(T& t) {
            return t;
        }

    protected:
        /*!
         * \brief Non-virtual destructor. Declared protected, so we don't accidentally try to do a polymorphic delete.
         */
        ~Tracked();

        /*!
         * \brief Record statistics. This function just forwards all its arguments to the internal `StatisticsCollection`.
         * \sa stats::StatisticsCollection::record
         */
        template<class T>
        void record(stat_id_t stat, T&& value) {
            // make m_Collection dependent, so it only needs to be a complete type where we actually call this function
            make_dependent<StatisticsCollection&, T>(*m_Collection).record(stat, std::forward<T>(value));
        }

        /*!
         * \brief Declares a new statistics. This function just forwards all its arguments to the internal `StatisticsCollection`.
         * \note Since this function will not be called in performance critical code, it is not implemented inline.
         * \sa stats::StatisticsCollection::declare_stat.
         */
        void declare_stat(stat_id_t index, StatisticMetaData meta);

        /*!
         * \brief Declares a new tag. This function just forwards all its arguments to the internal `StatisticsCollection`.
         * \note Since this function will not be called in performance critical code, it is not implemented inline.
         * \sa stats::StatisticsCollection::declare_tag.
         */
        void declare_tag(tag_id_t index, std::string name);


        /*!
         * \brief Set value of tag. This function just forwards all its arguments to the internal `StatisticsCollection`.
         * \tparam Args Dummy template parameter pack. Needs to be empty
         * \sa stats::StatisticsCollection::set_tag
         */
        template<class... Args>
        void set_tag(tag_id_t tag, long value) {
            static_assert(sizeof...(Args) == 0, "Unsupported extra args");
            // make m_Collection dependent, so it only needs to be a complete type where we actually call this function
            make_dependent<StatisticsCollection&, Args...>(*m_Collection).set_tag(tag, value);
        }

        /*!
         * \brief Creates a new `ScopeTimer` using `stats::record_scope_time`.
         * \tparam Args Dummy template parameter pack. Needs to be empty, but has to be used in this definition so that
         * the resulting call to `record_scope_time()` is only looked up in the second phase, where the function is
         * actually instantiated.
         * \sa stats::record_scope_time
         */
        template<class... Args>
        auto make_timer(stat_id_t id, Args... args) {
            // check that we don't accidentally use the dummy args in any way
            static_assert(sizeof...(Args) == 0, "Unsupported extra args");
            return record_scope_time(*m_Collection, id, args...);
        }

    private:
        /// The internal collection. Filled with a new `StatisticsCollection` in the constructor.
        std::shared_ptr<StatisticsCollection> m_Collection;
    };
}

#endif //DISMEC_STATS_TRACKED_H

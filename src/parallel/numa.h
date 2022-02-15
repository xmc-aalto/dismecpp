// Copyright (c) 2021, Aalto University, developed by Erik Schultheis
// All rights reserved.
//
// SPDX-License-Identifier: MIT

#ifndef DISMEC_NUMA_H
#define DISMEC_NUMA_H

#include <memory>
#include <vector>
#include <functional>
#include <any>
#include "spdlog/spdlog.h"

namespace parallel {

    /// Pint the calling thread to the NUMA node on which `data` resides.
    void pin_to_data(const void* data);

    /*!
     * \brief Base class for \ref NUMAReplicator
     * \details This class is not intended to be used directly, but only as an implementation details
     * for collecting the non-templated parts of \ref NUMAReplicator. It contains the replication logic,
     * but stores all data in a type-erased manner. To actually access the data the \ref NUMAReplicator
     * has to be used.
     * A derived class needs to implement a `get_clone()` function which creates a copy of the data
     * to be replicated. This function will be called in a context in which the allocation policy has
     * been set to allocate on a specific NUMA node, and need not concern itself with these details.
     *
     */
    class NUMAReplicatorBase {
    public:
        /// this returns true if NUMA is available. Otherwise, there will be no replication.
        [[nodiscard]] bool has_numa() const { return m_HasNUMA; }
        /// This returns the number of NUMA nodes.
        [[nodiscard]] int num_numa() const { return m_Copies.size(); }

    protected:
        NUMAReplicatorBase();
        virtual ~NUMAReplicatorBase() = default;

        /// Uses the `get_clone()` function of the implementing class to generate NUMA-local copies for each NUMA node.
        void make_copies();

        /// Returns the `std::any` that holds the copy for the calling thread. If NUMA is not available, returns a
        /// reference to a static, empty object.
        [[nodiscard]] const std::any& access_local() const;

    private:
        bool m_HasNUMA = false;     //!< Whether NUMA functions are available

        /// This vector contains one copy of data for each NUMA node.
        std::vector<std::any> m_Copies;

        /// This function needs to be implemented by a derived class.
        [[nodiscard]] virtual std::any get_clone() const = 0;
    };

    /*!
     * \brief Helper class to ensure that each NUMA node has its own copy of some immutable data
     * \tparam T The type of the data to be duplicated among NUMA nodes. Must be copyable.
     * \details We assume that we have the data available as `std::shared_ptr<T>`, and that `T`
     * is copyable. The class provides one function,
     * `get_local()` which returns a `shared_ptr` to the data on the NUMA node of the calling
     * process.
     *
     * Most of the implementation of this class is independent of the contained type, and provided
     * by \ref NUMAReplicatorBase.
     */
    template<class T>
    class NUMAReplicator : public NUMAReplicatorBase {
        using ptr_t = std::shared_ptr<const T>;
    public:
        /// Initializes the NUMAReplicator and distributes the data to all NUMA nodes.
        explicit NUMAReplicator(std::shared_ptr<const T> data) : m_Data(std::move(data)) {
            make_copies();
        }

        /// Gets the local copy of the data. The calling thread should cache this value,
        /// to prevent the overhead of doing this lookup.
        std::shared_ptr<const T> get_local() const {
            const std::any& data = access_local();
            if(data.has_value()) {
                return std::any_cast<ptr_t>(data);
            }
            return m_Data;
        }
    private:
        ptr_t m_Data;

        [[nodiscard]] std::any get_clone() const override {
            return std::make_shared<const T>(*m_Data);
        }
    };


    /*!
     * \brief This class helps with distributing threads to the different CPU cores.
     * \details For best performance, we want to pin the different threads to different
     * CPU cores.
     *
     * This works as follows. The constructor of `ThreadDistributor` is given the number
     * of threads it shall distribute, and produces a list of CPU cores to which threads
     * will be assigned. Subsequent calls to `pin_this_thread()` will then look up the
     * core in this list and pin the current thread to it. We also set the memory
     * strategy to allocate on the local node.
     *
     * The way the threads are distributed is as follows: We choose NUMA nodes in a round-robin
     * fashion, and assign cores in the node in increasing order. For example for two nodes with
     * four cores each, the allocation order would be `0:0, 1:4, 0:1, 1:5, 0:2, 1:6, ...`.
     * This strategy may be suboptimal if the number of threads is of the same order as the number
     * of NUMA nodes (e.g. running 4 threads on mahti).
     *
     * TODO enable spread-out allocation within a NUMA node
     * TODO figure out balancing of HT
     */
    class ThreadDistributor {
    public:
        ThreadDistributor(int num_threads, std::shared_ptr<spdlog::logger> = {});
        void pin_this_thread(int thread_id);
    private:
        std::vector<int> m_TargetCPUs;       //!< List of CPUs to which the threads will be assigned

        std::shared_ptr<spdlog::logger> m_Logger;
    };
}

#endif //DISMEC_NUMA_H

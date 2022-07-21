// Copyright (c) 2021, Aalto University, developed by Erik Schultheis
// All rights reserved.
//
// SPDX-License-Identifier: MIT

#include "parallel/numa.h"
#include "spdlog/spdlog.h"
#include "spdlog/fmt/fmt.h"
#include "utils/throw_error.h"
#include "config.h"
#include <numa.h>
#include <numaif.h>
#include <thread>
#include <numeric>
#include <fstream>

namespace parallel = dismec::parallel;
using namespace parallel;

// https://stackoverflow.com/questions/61454437/programmatically-get-accurate-cpu-cache-hierarchy-information-on-linux

namespace {
    int numa_node_count() {
        // `numa_max_node()` gives the highest index -- so to have the correct count, we need to add one
        return numa_max_node() + 1;
    }

    int current_numa_node() {
        int cpu = sched_getcpu();
        return numa_node_of_cpu(cpu);
    }

    int lookup_numa_node(const void* ptr) {
        int numa_node = -1;
        get_mempolicy(&numa_node, nullptr, 0, const_cast<void*>(ptr), MPOL_F_NODE | MPOL_F_ADDR); // NOLINT(cppcoreguidelines-pro-type-const-cast)
        return numa_node;
    }

    template<class F>
    void handle_cpu_set(std::fstream& source, F&& action) {
        std::string buffer;
        std::getline(source, buffer);
        bitmask* shared = numa_parse_cpustring_all(buffer.c_str());
        for(int i = 0; i < numa_num_possible_cpus(); ++i) {
            if (numa_bitmask_isbitset(shared, i)) {
                action(cpu_id_t{i});
            }
        }
        numa_free_cpumask(shared);
    }

    template<class F>
    void for_each_sibling(cpu_id_t core, F&& action) {
        std::string file_name = fmt::format("/sys/devices/system/cpu/cpu{}/topology/thread_siblings_list",
                                            core.to_index());
        std::fstream topology_file(file_name, std::fstream::in);
        if(!topology_file.is_open()) {
            spdlog::error("Could not open topology file '{}'", file_name);
        }
        handle_cpu_set(topology_file, std::forward<F>(action));
    }

    template<class F>
    void for_each_shared_cache_sibling(cpu_id_t core, F&& action) {
        constexpr const int MAX_CACHE_LEVELS = 10;
        for(int index = 0; index < MAX_CACHE_LEVELS; ++index) {
            std::string file_name = fmt::format("/sys/devices/system/cpu/cpu{}/cache/index{}/shared_cpu_list",
                                                core.to_index(), index);
            std::fstream topology_file(file_name, std::fstream::in);
            if (!topology_file.is_open()) {
                return;     // ok, we've handled the last cache
            }
            handle_cpu_set(topology_file, std::forward<F>(action));
        }
    }
}

void parallel::pin_to_data(const void* data) {
    int target_node = lookup_numa_node(data);
    errno = 0;
    if(numa_run_on_node(target_node) == -1) {
        spdlog::error("Error pinning thread {} to node {}: {}", pthread_self(), target_node, strerror(errno));
    }
}


NUMAReplicatorBase::NUMAReplicatorBase() : m_HasNUMA(numa_available() >= 0)
{
}

void NUMAReplicatorBase::make_copies() {
    if(m_HasNUMA) {
        m_Copies.clear();
        m_Copies.resize(numa_node_count());

        for(int i = 0; i < numa_node_count(); ++i) {
            if(numa_bitmask_isbitset(numa_all_nodes_ptr, i) == 0) {
                spdlog::warn("NUMA node {} is disabled, no local data copy was created", i);
                continue;
            }
            int current_preferred = numa_preferred();
            numa_set_preferred(i);

            m_Copies.at(i) = get_clone();
            numa_set_preferred(current_preferred);
        }
    }
}

const std::any& NUMAReplicatorBase::access_local() const {
    if(m_HasNUMA) {
        return m_Copies.at(current_numa_node());
    } else {
        static std::any empty;
        return empty;
    }
}

namespace {
    // sets the cpu affinity of the current thread to i
    bool set_thread_affinity(cpu_id_t cpu) {
        // first, we set the current thread to CPU `i`
        cpu_set_t cpuset;
        CPU_ZERO(&cpuset);
        CPU_SET(cpu.to_index(), &cpuset);
        int rc = pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset);

        if (rc != 0) {
            spdlog::error("Error fixing thread {} to core {}: {}\n", pthread_self(), cpu.to_index(), strerror(rc));
            return false;
        }
        return true;
    }

    class NodeData {
    public:
        explicit NodeData(int id) : m_ID(id) {
        }

        [[nodiscard]] numa_node_id_t get_id() const {
            return m_ID;
        }

        void add_cpu(cpu_id_t cpu_id) {
            m_CPUs.push_back(cpu_id);
            m_LoadIndicator.push_back(0);
        }

        [[nodiscard]] bool empty() const {
            return m_CPUs.empty();
        }

        [[nodiscard]] int num_threads() const {
            return NumThreads;
        }

        [[nodiscard]] int max_load() const {
            auto max_el = std::max_element(begin(m_LoadIndicator), end(m_LoadIndicator));
            return *max_el;
        }

        [[nodiscard]] cpu_id_t place_thread() {
            auto min_core = std::min_element(begin(m_LoadIndicator), end(m_LoadIndicator));
            long index = std::distance(begin(m_LoadIndicator), min_core);
            m_LoadIndicator[index] += COST_PLACE_THREAD;
            for_each_sibling(m_CPUs[index], [&](cpu_id_t sibling) {
                auto found = std::find(begin(m_CPUs), end(m_CPUs), sibling);
                if(found == end(m_CPUs)) {
                    return;         // I guess this means that we are not allowed to run on the sibling, so this is fine
                }
                *(begin(m_LoadIndicator) + std::distance(begin(m_CPUs), found)) += COST_PLACE_HYPER_THREAD;
            });

            for_each_shared_cache_sibling(m_CPUs[index], [&](cpu_id_t sibling) {
                auto found = std::find(begin(m_CPUs), end(m_CPUs), sibling);
                if(found == end(m_CPUs)) {
                    return;         // I guess this means that we are not allowed to run on the sibling, so this is fine
                }
                *(begin(m_LoadIndicator) + std::distance(begin(m_CPUs), found)) += 1;
            });

            ++NumThreads;
            return m_CPUs[index];
        }
    private:
        numa_node_id_t m_ID;
        std::vector<cpu_id_t> m_CPUs;              //!< Vector of CPU ids that are on this NUMA node
        /*!
         * \brief How much work have we placed on that CPU.
         * \details Currently uses the following heuristic: Placing a thread on the core increases the load by a value
         * of 10, and increases the load of any core that is shared by hyperthreading by 5.
         */
        std::vector<int> m_LoadIndicator;

        int NumThreads = 0;
    };

    /*!
     * \brief Returns the list of available NUMA nodes.
     * \details If no NUMA information is available, we put all cores into a
     * NUMA 0 node.
     * If CPUs are available for which the corresponding memory node
     * is not available (this happens on my desktop),
     * we assign these CPUs to NUMA node 0.
     */
    std::vector<NodeData> get_available_nodes() {
        std::vector<NodeData> nodes_avail;
        // first, we check which CPU indices are available
        if(numa_available() >= 0) {
            for(int i = 0; i < numa_num_possible_nodes(); ++i) {
                nodes_avail.emplace_back(i);
            }
            for(int i = 0; i < numa_num_possible_cpus(); ++i) {
                if (numa_bitmask_isbitset(numa_all_cpus_ptr, i) != 0) {
                    // OK, CPU is available
                    int node = numa_node_of_cpu(i);
                    if(numa_bitmask_isbitset(numa_all_nodes_ptr, node) == 0) {
                        spdlog::warn("Node {} of CPU {} is not available.", node, i);
                    }
                    nodes_avail.at(node).add_cpu(cpu_id_t{i});
                }
            }

        } else {
            // if we don't have numa available, assume all CPUs are available and correspond to node 0
            nodes_avail.emplace_back(0);
            for(unsigned i = 0; i < std::thread::hardware_concurrency(); ++i) {
                nodes_avail.at(0).add_cpu(cpu_id_t{i});
            }
        }

        // remove all NUMA nodes for which we don't have any CPUs
        nodes_avail.erase(std::remove_if(begin(nodes_avail), end(nodes_avail),
                                         [](auto&& d){ return d.empty(); }),
                          end(nodes_avail));
        return nodes_avail;
    }
}

ThreadDistributor::ThreadDistributor(long num_threads, std::shared_ptr<spdlog::logger> logger) :
        m_Logger(std::move(logger))
{
    if(num_threads < 0) {
        THROW_EXCEPTION(std::invalid_argument, "Negative number of threads {} given!", num_threads)
    }
    std::vector<NodeData> nodes_avail = get_available_nodes();

    if(m_Logger) {
        m_Logger->info("Distributing {} threads to {} NUMA nodes.",
                       num_threads, nodes_avail.size());
    }
    while(m_TargetCPUs.size() < static_cast<std::size_t>(num_threads)) {
        for(auto& node : nodes_avail) {
            m_TargetCPUs.push_back(node.place_thread());
        }
    }

    if(m_Logger) {
        for (auto& node : nodes_avail) {
            m_Logger->info("Node {}: {} threads, load {}", node.get_id().to_index(), node.num_threads(), node.max_load());
        }
    }
}

void ThreadDistributor::pin_this_thread(thread_id_t thread_id) {
    auto target_cpu = m_TargetCPUs.at(thread_id.to_index());
    if(!set_thread_affinity(target_cpu)) {
        throw std::runtime_error(fmt::format("Could not pin thread {} to CPU {}",
                                             pthread_self(), target_cpu.to_index()));
    }

    // if we successfully pinned a thread to a CPU, we also pin its memory allocation to the corresponding
    // NUMA node
    if(numa_available() >= 0) {
        numa_set_localalloc();
    }

    if(m_Logger) {
        m_Logger->info("Pinned thread {} ({}) to Core {} on Node {}",
                       thread_id.to_index(), pthread_self(),
                       target_cpu.to_index(),
                       numa_available() >= 0 ? current_numa_node() : -1);
    }
}
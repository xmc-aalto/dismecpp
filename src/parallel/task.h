// Copyright (c) 2021, Aalto University, developed by Erik Schultheis
// All rights reserved.
//
// SPDX-License-Identifier: MIT

#ifndef DISMEC_TASK_H
#define DISMEC_TASK_H

#include "thread_id.h"

namespace dismec::parallel {
    /*!
     * \brief Base class for all parallelized operations
     * \details Any computation that should be parallelized using
     * the `ParallelRunner` has to be implemented as a subclass of `TaskGenerator`.
     * This requires implementing two functions `num_tasks()` which returns the
     * number of tasks the generator provides, and `run_tasks()`, which shall
     * execute the actual computation for a given task. The `run_tasks()` function
     * has to be re-entrant when called with different non-overlapping `[begin, end)` intervals.
     */
    class TaskGenerator {
    public:
        using thread_id_t = dismec::parallel::thread_id_t;

        virtual ~TaskGenerator() = default;
        [[nodiscard]] virtual long num_tasks() const = 0;

        virtual void run_tasks(long begin, long end, thread_id_t thread_id) = 0;

        /*!
         * \brief Called to notify the `TaskGenerator` about the number of threads.
         * \details This function is called from the main thread, before distributed work
         * is started. It gives the `TaskGenerator` a chance to allocate working memory for
         * each thread, so these allocations don't need to be done and repeated in `run_task()`.
         * \note For memory that is used inside the computations done by each thread, the `init_thread()`
         * should be used. This will be called from inside the thread that will do the actual computations, so
         * that when using this on a `NUMA` system, first-touch policy has a chance to place the allocation in the
         * correct RAM. In that case, this function should only allocate an array of pointers that will be filled
         * in by `init_thread()`.
         * \param num_threads Number of threads that will be used.
         * \param chunk_size A hint for the size of chunks used when running this task. Note that if the total number of
         * tasks is not a multiple of the `chunk_size`, there may be some calls to `run_tasks()`
         * with less than `chunk_size` tasks.
         */
        virtual void prepare(long num_threads, long chunk_size) {};

        /*!
         * \brief Called once a thread has spun up, but before it runs its first task.
         * \details This function is called from inside the thread that also will run the tasks.
         */
        virtual void init_thread(thread_id_t thread_id) {};

        /*!
         * \brief Called after all threads have finished their tasks.
         * \details This function is called from the main thread after all worker threads have finished
         * their work. It can be used to perform single threaded reductions or clean up
         * per-thread buffers.
         */
        virtual void finalize() {}
    };
}

#endif //DISMEC_TASK_H

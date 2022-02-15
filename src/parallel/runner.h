// Copyright (c) 2021, Aalto University, developed by Erik Schultheis
// All rights reserved.
//
// SPDX-License-Identifier: MIT

#ifndef DISMEC_RUNNER_H
#define DISMEC_RUNNER_H

#include <functional>
#include <memory>
#include <chrono>
#include "spdlog/spdlog.h"

namespace parallel {
    class TaskGenerator;

    struct RunResult {
        bool IsFinished = false;        //!< If this is true, then all tasks have been run successfully
        long NextTask = -1;             //!< If running timed out before all tasks were done, this is the task
        //!< with which a subsequent run should start.
        // timing info
        std::chrono::seconds Duration;  //!< How long did this run take.
    };

    class ParallelRunner {
    public:
        /// @param num_threads Number of threads to use. Value <= 0 indicate auto-detect,
        /// using `std::thread::hardware_concurrency()`
        explicit ParallelRunner(long num_threads, long chunk_size=1);

        void set_chunk_size(long chunk_size);
        void set_time_limit(std::chrono::milliseconds time_limit);

        /// sets the logger object that is used for reporting. Set to nullptr for quiet mode.
        void set_logger(std::shared_ptr<spdlog::logger> logger);

        /*!
         * Runs the tasks provided by the `TaskGenerator` in parallel.
         * @param tasks The task generator that returns a runnable function for each task index.
         * @param start The first task id to run.
         */
        [[nodiscard]] RunResult run(TaskGenerator& tasks, long start=0);

    private:

        // helpers
        void log_start(long begin, long end);
        void log_finished(long begin, long end);

        long m_NumThreads;
        long m_ChunkSize = 1;
        std::chrono::milliseconds m_TimeLimit;
        std::shared_ptr<spdlog::logger> m_Logger;

        bool m_BindThreads = true;
    };
}

#endif //DISMEC_RUNNER_H

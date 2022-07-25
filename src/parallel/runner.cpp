// Copyright (c) 2021, Aalto University, developed by Erik Schultheis
// All rights reserved.
//
// SPDX-License-Identifier: MIT

#include <algorithm>
#include <thread>
#include <atomic>
#include "config.h"
#include "parallel/runner.h"
#include "parallel/task.h"
#include "parallel/numa.h"
#include "utils/conversion.h"

using namespace dismec;
using namespace dismec::parallel;

ParallelRunner::ParallelRunner(long num_threads, long chunk_size) :
        m_NumThreads(num_threads), m_ChunkSize(chunk_size),
        m_TimeLimit(std::numeric_limits<std::chrono::milliseconds::rep>::max()) {

}

void ParallelRunner::set_chunk_size(long chunk_size) {
    m_ChunkSize = chunk_size;
}

void ParallelRunner::set_logger(std::shared_ptr<spdlog::logger> logger) {
    m_Logger = std::move(logger);
}

namespace {
    template<class T>
    auto to_ms(T&& arg) {
        return std::chrono::duration_cast<std::chrono::milliseconds>(arg);
    }
}

RunResult ParallelRunner::run(TaskGenerator& tasks, long start) {
    using std::chrono::milliseconds;
    using std::chrono::steady_clock;

    long num_threads = m_NumThreads;
    if(num_threads <= 0) {
        num_threads = to_long(std::thread::hardware_concurrency());
    }
    if(num_threads > 2*std::thread::hardware_concurrency() + 1) {
        spdlog::warn("You have specified many more threads ({}) than your hardware appears to support ({}). Number"
                     "of threads has been capped at hardware concurrency.",
                     num_threads, std::thread::hardware_concurrency());
        num_threads = static_cast<long>(std::thread::hardware_concurrency());
    }

    long num_tasks = tasks.num_tasks() - start;
    long num_chunks = num_tasks / m_ChunkSize;
    if(num_tasks % m_ChunkSize != 0) {
        num_chunks += 1;
    }
    num_threads = std::min(num_threads, num_chunks);

    std::atomic<std::size_t> cpu_time{0};

    // we need an atomic counter to make sure that all sub-problems are touched exactly once
    std::atomic<long> sub_counter{0};
    // long sub_counter = start;
    // nothing we can do if the counter isn't lock free, but notify the user.
    if(!sub_counter.is_lock_free()) {
        spdlog::warn("Counter implementation is not lock-free. This might result in degraded performance in case of many threads");
    }

    auto start_time = steady_clock::now();

    std::vector<std::thread> workers;
    workers.reserve(num_threads);
    if(m_Logger)
        m_Logger->info("spawning {} threads to run {} tasks", num_threads, num_tasks);

    tasks.prepare(num_threads, m_ChunkSize);
    ThreadDistributor distribute(num_threads, m_Logger);

    for(int thread = 0; thread < num_threads; ++thread) {
        workers.emplace_back([&, thread_id=thread_id_t(thread)]()
        {
             if(m_BindThreads) {
                 distribute.pin_this_thread(thread_id);
             }

             tasks.init_thread(thread_id);

             while(to_ms(steady_clock::now() - start_time) < m_TimeLimit) {
                 // get a new sub-problem
                 // see also https://stackoverflow.com/questions/41206861/atomic-increment-and-return-counter
                 long search_pos = sub_counter++;
                 if(search_pos >= num_chunks) {
                     return;
                 }

                 auto task_start_time = steady_clock::now();

                 long begin_task = search_pos * m_ChunkSize + start;
                 long end_task = std::min((search_pos + 1) * m_ChunkSize, (long)num_tasks) + start;

                 log_start(begin_task, end_task);
                 tasks.run_tasks(begin_task, end_task, thread_id);
                 log_finished(begin_task, end_task);

                 cpu_time.fetch_add( to_ms(steady_clock::now() - task_start_time).count());
             }
        });
    }

    // OK, now we just have to wait for the threads to finnish
    for(auto& t : workers) {
        t.join();
    }

    tasks.finalize();

    auto wall_time = to_ms(steady_clock::now() - start_time);

    if(m_Logger) {
        if(sub_counter >= num_chunks) {
            m_Logger->info("Threads finished after {}s (per thread {}s).", wall_time.count() / 1000,
                           cpu_time / 1000 / num_threads);
        } else {
            m_Logger->info("Computation timeout ({}s) reached after {} tasks ({}s -- {}s per thread)",
                           m_TimeLimit.count() / 1000,
                           sub_counter, wall_time.count() / 1000, cpu_time / 1000 / num_threads);
        }
    }

    // display a warning if threads need to get new work more than every 5 ms.
    if((cpu_time * m_ChunkSize) / num_tasks < MIN_TIME_PER_CHUNK_MS) {
        spdlog::warn("The average time per chunk of work is only {}µs, consider increasing chunk size (currently {}) to "
                     "reduce parallelization overhead.", (1000 * cpu_time * m_ChunkSize) / num_tasks, m_ChunkSize);
    }

    return {sub_counter >= num_chunks, sub_counter * m_ChunkSize + start,
            std::chrono::duration_cast<std::chrono::seconds>(wall_time)};
}

void ParallelRunner::log_start(long begin, long end) {
    if(!m_Logger) return;
    if(begin == end - 1) {
        m_Logger->trace("Starting task {}", begin);
    } else {
        m_Logger->trace("Starting tasks {}-{}", begin, end-1);
    }
}

void ParallelRunner::log_finished(long begin, long end) {
    if(!m_Logger) return;
    if(begin == end - 1) {
        m_Logger->trace("Finished task {}", begin);
    } else {
        m_Logger->trace("Finished tasks {}-{}", begin, end-1);
    }
}

void ParallelRunner::set_time_limit(std::chrono::milliseconds time_limit) {
    if(time_limit.count() <= 0) {
        m_TimeLimit = std::chrono::milliseconds(std::numeric_limits<std::chrono::milliseconds::rep>::max());
    } else {
        m_TimeLimit = time_limit;
    }
}

#include "doctest.h"

namespace {
    struct DummyTask: TaskGenerator {
        DummyTask() : check(10000, 0) {

        }
        void run_tasks(long begin, long end, thread_id_t thread_id) override {
            for(long t = begin; t < end; ++t) {
                check.at(t) += 1;
                if(do_work) {
                    std::this_thread::sleep_for(std::chrono::milliseconds(10));
                }
            }
        }
        [[nodiscard]] long num_tasks() const override {
            return check.size();
        }

        std::vector<int> check;
        bool do_work=false;
    };
}

TEST_CASE("run parallel") {
    ParallelRunner runner{-1};
    DummyTask task;
    auto res = runner.run(task);
    REQUIRE(res.IsFinished);

    // make sure each task ran exactly once
    for(int s = 0; s < ssize(task.check); ++s) {
        REQUIRE_MESSAGE(task.check[s] == 1, "error at index " << s);
    }
}

TEST_CASE("run chunked parallel with start pos")
{
    ParallelRunner runner{-1, 32};
    DummyTask task;
    auto res = runner.run(task, 5);
    REQUIRE(res.IsFinished);

    // make sure that skipped tasks are not run, but all others are
    for(int s = 0; s < 5; ++s) {
        REQUIRE(task.check[s] == 0);
    }
    for(int s = 5; s < ssize(task.check); ++s) {
        REQUIRE_MESSAGE(task.check[s] == 1, "error at index " << s);
    }
}

TEST_CASE("run parallel with timeout") {
    ParallelRunner runner{-1, 16};
    DummyTask task;
    task.do_work = true;
    runner.set_time_limit(std::chrono::milliseconds(50));
    auto res = runner.run(task, 5);
    REQUIRE_FALSE(res.IsFinished);

    // check that NextTask correctly identifies until where we have done our work
    for(int s = 5; s < res.NextTask; ++s) {
        REQUIRE(task.check[s] == 1);
    }for(int s = res.NextTask; s < ssize(task.check); ++s) {
        REQUIRE(task.check[s] == 0);
    }
}

// TODO check chunks, starts etc
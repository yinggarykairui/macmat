#pragma once
#include <cstddef>
#include <string>
#include <vector>

#include "macmat/multiply.hpp"

namespace macmat::bench {

struct Config {
    Impl impl = Impl::Blocked;
    std::size_t m = 512, k = 512, n = 512;
    std::size_t tile = 32;
    unsigned threads = 0;      // 0 = hardware concurrency
    int samples = 15;
    int warmup = 3;
    double min_sample_ms = 5.0;  // batch calls until one sample is at least this long
    unsigned seed = 42;
};

struct Result {
    Config cfg;
    unsigned threads_used = 0;
    int batch = 1;                       // calls per timed sample
    std::vector<double> sample_ms;       // per-call latency for each sample
    double median_ms = 0, q1_ms = 0, q3_ms = 0, min_ms = 0, max_ms = 0;
    double gflops = 0;                   // from median latency
    double checksum = 0;                 // consumed result, keeps the compiler honest
};

struct Environment {
    std::string chip, macos, compiler, flags, power_mode, accelerate_threads_env, timestamp;
};

Environment probe_environment();
Result run(const Config& cfg);
std::string csv_header();
std::string csv_row(const Result& r, const Environment& env, int repeat, int order_index);
void print_summary(const Result& r);

}  // namespace macmat::bench

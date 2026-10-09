#include "bench.hpp"

#include <sys/sysctl.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <sstream>

#ifndef MACMAT_CXX_FLAGS
#define MACMAT_CXX_FLAGS "unknown"
#endif

namespace macmat::bench {

namespace {

std::string sysctl_string(const char* name) {
    char buf[256];
    std::size_t len = sizeof(buf);
    if (sysctlbyname(name, buf, &len, nullptr, 0) != 0) return "unknown";
    return std::string(buf, len > 0 ? len - 1 : 0);
}

std::string shell_line(const char* cmd) {
    FILE* p = popen(cmd, "r");
    if (!p) return "unknown";
    char buf[256];
    std::string out;
    if (std::fgets(buf, sizeof(buf), p)) out = buf;
    pclose(p);
    while (!out.empty() && (out.back() == '\n' || out.back() == ' ')) out.pop_back();
    return out.empty() ? "unknown" : out;
}

double percentile(std::vector<double> v, double q) {
    std::sort(v.begin(), v.end());
    const double pos = q * (v.size() - 1);
    const std::size_t lo = static_cast<std::size_t>(pos);
    const std::size_t hi = std::min(lo + 1, v.size() - 1);
    const double frac = pos - lo;
    return v[lo] * (1.0 - frac) + v[hi] * frac;
}

inline void run_once(const Config& cfg, const Matrix& A, const Matrix& B, Matrix& C,
                     const MultiplyOptions& opts) {
    multiply(cfg.impl, A, B, C, opts);
}

}  // namespace

Environment probe_environment() {
    Environment e;
    e.chip = sysctl_string("machdep.cpu.brand_string");
    e.macos = sysctl_string("kern.osproductversion");
    e.compiler = "Apple clang " __clang_version__;
    e.flags = MACMAT_CXX_FLAGS;
    const std::string lp = shell_line("pmset -g 2>/dev/null | awk '/lowpowermode/{print $2}'");
    e.power_mode = lp == "1" ? "low-power" : (lp == "0" ? "automatic" : "unknown");
    const char* v = std::getenv("VECLIB_MAXIMUM_THREADS");
    e.accelerate_threads_env = v ? v : "unset";
    char ts[32];
    const std::time_t now = std::time(nullptr);
    std::strftime(ts, sizeof(ts), "%Y-%m-%dT%H:%M:%S", std::localtime(&now));
    e.timestamp = ts;
    return e;
}

Result run(const Config& cfg) {
    using clock = std::chrono::steady_clock;
    Result r;
    r.cfg = cfg;
    MultiplyOptions opts;
    opts.tile = cfg.tile;
    opts.threads = cfg.threads;
    r.threads_used = (cfg.impl == Impl::Parallel || cfg.impl == Impl::MicroParallel)
                         ? static_cast<unsigned>(std::min<std::size_t>(
                               cfg.threads == 0 ? default_thread_count() : cfg.threads, cfg.m))
                         : 1;

    // Inputs allocated and initialised outside timing. C is allocated once and
    // reused; each kernel overwrites it (and any zero-fill it needs is timed).
    const Matrix A = Matrix::random(cfg.m, cfg.k, cfg.seed);
    const Matrix B = Matrix::random(cfg.k, cfg.n, cfg.seed + 1);
    Matrix C(cfg.m, cfg.n);

    // Warm-up: pages touched, caches populated, Accelerate's internal pools started.
    double probe_ms = 0;
    for (int w = 0; w < std::max(1, cfg.warmup); ++w) {
        const auto t0 = clock::now();
        run_once(cfg, A, B, C, opts);
        const auto t1 = clock::now();
        probe_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    }
    // Batch tiny operations so that timer overhead (~tens of ns) is negligible.
    r.batch = probe_ms >= cfg.min_sample_ms ? 1
                                            : static_cast<int>(std::ceil(cfg.min_sample_ms / std::max(probe_ms, 1e-6)));
    r.batch = std::clamp(r.batch, 1, 1000000);

    r.sample_ms.reserve(cfg.samples);
    for (int s = 0; s < cfg.samples; ++s) {
        const auto t0 = clock::now();
        for (int b = 0; b < r.batch; ++b) run_once(cfg, A, B, C, opts);
        const auto t1 = clock::now();
        r.sample_ms.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count() / r.batch);
        // Consume the result outside timing so the work cannot be eliminated.
        r.checksum += static_cast<double>(C.data()[s % C.size()]);
    }
    for (std::size_t i = 0; i < C.size(); i += 97) r.checksum += C.data()[i];

    r.median_ms = percentile(r.sample_ms, 0.5);
    r.q1_ms = percentile(r.sample_ms, 0.25);
    r.q3_ms = percentile(r.sample_ms, 0.75);
    r.min_ms = *std::min_element(r.sample_ms.begin(), r.sample_ms.end());
    r.max_ms = *std::max_element(r.sample_ms.begin(), r.sample_ms.end());
    const double flop = 2.0 * static_cast<double>(cfg.m) * static_cast<double>(cfg.k) * static_cast<double>(cfg.n);
    r.gflops = flop / (r.median_ms * 1e-3) / 1e9;
    return r;
}

std::string csv_header() {
    return "impl,M,K,N,tile,threads,batch,samples,median_ms,q1_ms,q3_ms,iqr_ms,min_ms,max_ms,gflops,"
           "repeat,order_index,chip,macos,compiler,flags,power_mode,veclib_max_threads,timestamp";
}

std::string csv_row(const Result& r, const Environment& env, int repeat, int order_index) {
    std::ostringstream os;
    os.precision(6);
    os << impl_name(r.cfg.impl) << ',' << r.cfg.m << ',' << r.cfg.k << ',' << r.cfg.n << ','
       << r.cfg.tile << ',' << r.threads_used << ',' << r.batch << ',' << r.sample_ms.size() << ','
       << r.median_ms << ',' << r.q1_ms << ',' << r.q3_ms << ',' << (r.q3_ms - r.q1_ms) << ','
       << r.min_ms << ',' << r.max_ms << ',' << r.gflops << ',' << repeat << ',' << order_index << ','
       << '"' << env.chip << "\",\"" << env.macos << "\",\"" << env.compiler << "\",\"" << env.flags
       << "\",\"" << env.power_mode << "\",\"" << env.accelerate_threads_env << "\",\"" << env.timestamp << '"';
    return os.str();
}

void print_summary(const Result& r) {
    std::printf("%-10s %5zux%5zux%5zu tile=%-4zu thr=%-2u batch=%-6d median=%10.4f ms  IQR=%9.4f ms  %8.2f GFLOP/s\n",
                impl_name(r.cfg.impl), r.cfg.m, r.cfg.k, r.cfg.n, r.cfg.tile, r.threads_used, r.batch,
                r.median_ms, r.q3_ms - r.q1_ms, r.gflops);
}

}  // namespace macmat::bench

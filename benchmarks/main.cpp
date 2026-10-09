// macmat command-line tool.
//   macmat bench --impl blocked --m 512 --k 512 --n 512 --tile 32 --samples 15 --output results/x.csv
//   macmat suite --output results/suite.csv --repeat 2 [--impls naive,reordered,...] [--tile 32]
//   macmat info
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <stdexcept>
#include <string>
#include <sys/stat.h>
#include <vector>

#include "bench.hpp"

using namespace macmat;
using namespace macmat::bench;

namespace {

void usage() {
    std::puts(
        "usage:\n"
        "  macmat bench --impl <naive|reordered|blocked|parallel|accelerate|micro|micro_parallel>\n"
        "               --m M --k K --n N\n"
        "               [--tile T] [--threads P] [--samples S] [--warmup W] [--min-sample-ms X]\n"
        "               [--output file.csv] [--repeat-id R] [--order-index I]\n"
        "  macmat suite [--impls a,b,c] [--tile T] [--threads P] [--samples S] [--repeat R]\n"
        "               [--output file.csv]   (runs the standard workload, rotating impl order)\n"
        "  macmat info");
}

std::map<std::string, std::string> parse_flags(int argc, char** argv, int start) {
    std::map<std::string, std::string> f;
    for (int i = start; i < argc; ++i) {
        std::string a = argv[i];
        if (a.rfind("--", 0) != 0) throw std::invalid_argument("unexpected argument: " + a);
        a = a.substr(2);
        if (i + 1 >= argc) throw std::invalid_argument("flag --" + a + " needs a value");
        f[a] = argv[++i];
    }
    return f;
}

template <typename T>
T get_num(const std::map<std::string, std::string>& f, const std::string& key, T def) {
    auto it = f.find(key);
    if (it == f.end()) return def;
    return static_cast<T>(std::strtod(it->second.c_str(), nullptr));
}

bool file_exists(const std::string& p) {
    struct stat st;
    return stat(p.c_str(), &st) == 0 && st.st_size > 0;
}

void append_csv(const std::string& path, const Result& r, const Environment& env, int repeat, int order) {
    if (path.empty()) return;
    const bool fresh = !file_exists(path);
    std::ofstream out(path, std::ios::app);
    if (!out) throw std::runtime_error("cannot open output file " + path);
    if (fresh) out << csv_header() << '\n';
    out << csv_row(r, env, repeat, order) << '\n';
}

void apply_common(Config& cfg, const std::map<std::string, std::string>& f) {
    cfg.tile = get_num<std::size_t>(f, "tile", cfg.tile);
    cfg.threads = get_num<unsigned>(f, "threads", cfg.threads);
    cfg.samples = get_num<int>(f, "samples", cfg.samples);
    cfg.warmup = get_num<int>(f, "warmup", cfg.warmup);
    cfg.min_sample_ms = get_num<double>(f, "min-sample-ms", cfg.min_sample_ms);
    cfg.seed = get_num<unsigned>(f, "seed", cfg.seed);
    if (cfg.samples < 1) throw std::invalid_argument("--samples must be >= 1");
}

int cmd_bench(const std::map<std::string, std::string>& f) {
    Config cfg;
    auto it = f.find("impl");
    if (it == f.end()) throw std::invalid_argument("--impl is required");
    auto impl = parse_impl(it->second);
    if (!impl) throw std::invalid_argument("unknown implementation: " + it->second);
    cfg.impl = *impl;
    cfg.m = get_num<std::size_t>(f, "m", cfg.m);
    cfg.k = get_num<std::size_t>(f, "k", cfg.k);
    cfg.n = get_num<std::size_t>(f, "n", cfg.n);
    apply_common(cfg, f);
    const Environment env = probe_environment();
    const Result r = run(cfg);
    print_summary(r);
    auto out = f.find("output");
    append_csv(out == f.end() ? "" : out->second, r, env, get_num<int>(f, "repeat-id", 0),
               get_num<int>(f, "order-index", 0));
    return 0;
}

int cmd_suite(const std::map<std::string, std::string>& f) {
    struct Shape { std::size_t m, k, n; const char* family; };
    const Shape shapes[] = {
        {32, 32, 32, "small"},       {64, 64, 64, "small"},
        {128, 128, 128, "medium"},   {256, 256, 256, "medium"},
        {512, 512, 512, "large"},    {1024, 1024, 1024, "large"},
        {1024, 128, 512, "rect"},    {128, 1024, 64, "rect"},
        {127, 255, 63, "irregular"},
    };
    std::vector<Impl> impls;
    auto it = f.find("impls");
    if (it == f.end()) {
        impls.assign(std::begin(kAllImpls), std::end(kAllImpls));
    } else {
        std::string s = it->second;
        std::size_t pos = 0;
        while (pos <= s.size()) {
            std::size_t comma = s.find(',', pos);
            if (comma == std::string::npos) comma = s.size();
            auto impl = parse_impl(s.substr(pos, comma - pos));
            if (!impl) throw std::invalid_argument("unknown implementation in --impls: " + s.substr(pos, comma - pos));
            impls.push_back(*impl);
            pos = comma + 1;
        }
    }
    const int repeats = get_num<int>(f, "repeat", 2);
    auto out_it = f.find("output");
    const std::string output = out_it == f.end() ? "" : out_it->second;
    const Environment env = probe_environment();

    for (int rep = 0; rep < repeats; ++rep) {
        // Rotate implementation order on each repeat so that thermal state and
        // frequency ramp do not systematically favour one implementation.
        std::vector<Impl> order;
        for (std::size_t i = 0; i < impls.size(); ++i) order.push_back(impls[(i + rep) % impls.size()]);
        std::printf("== repeat %d, order:", rep);
        for (Impl impl : order) std::printf(" %s", impl_name(impl));
        std::printf("\n");
        for (const Shape& s : shapes) {
            for (std::size_t oi = 0; oi < order.size(); ++oi) {
                Config cfg;
                cfg.impl = order[oi];
                cfg.m = s.m; cfg.k = s.k; cfg.n = s.n;
                apply_common(cfg, f);
                const Result r = run(cfg);
                print_summary(r);
                std::fflush(stdout);
                append_csv(output, r, env, rep, static_cast<int>(oi));
            }
        }
    }
    return 0;
}

int cmd_info() {
    const Environment env = probe_environment();
    std::printf("chip: %s\nmacos: %s\ncompiler: %s\nflags: %s\npower_mode: %s\n"
                "hardware_concurrency: %u\nVECLIB_MAXIMUM_THREADS: %s\ntimestamp: %s\n",
                env.chip.c_str(), env.macos.c_str(), env.compiler.c_str(), env.flags.c_str(),
                env.power_mode.c_str(), default_thread_count(), env.accelerate_threads_env.c_str(),
                env.timestamp.c_str());
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) { usage(); return 2; }
    const std::string cmd = argv[1];
    try {
        if (cmd == "bench") return cmd_bench(parse_flags(argc, argv, 2));
        if (cmd == "suite") return cmd_suite(parse_flags(argc, argv, 2));
        if (cmd == "info") return cmd_info();
        usage();
        return 2;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
}

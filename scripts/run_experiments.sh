#!/bin/zsh
# Runs every MacMat experiment sequentially and writes CSVs into results/.
# Usage: scripts/run_experiments.sh [build_dir]   (default: build)
# Keep the machine otherwise idle while this runs (about 10 minutes on an M4).
set -euo pipefail
cd "$(dirname "$0")/.."
BUILD=${1:-build}
BIN="$BUILD/macmat"
SAMPLES=${SAMPLES:-15}
mkdir -p results
# Set CLEAN=1 to discard previous CSVs; otherwise phases whose CSV exists are skipped.
if [[ "${CLEAN:-0}" == "1" ]]; then
  rm -f results/suite_tile32.csv results/tile_sweep.csv results/thread_scaling.csv \
        results/accelerate_threads.csv results/suite_best.csv
fi

{
  echo "date: $(date -Iseconds)"
  "$BIN" info
  echo "hardware: $(sysctl -n hw.perflevel0.physicalcpu) performance cores, $(sysctl -n hw.perflevel1.physicalcpu) efficiency cores"
  echo "L1d (P-core): $(sysctl -n hw.perflevel0.l1dcachesize) bytes; L2 (P-cluster): $(sysctl -n hw.perflevel0.l2cachesize) bytes; L2 (E-cluster): $(sysctl -n hw.perflevel1.l2cachesize) bytes"
  echo "cache line: $(sysctl -n hw.cachelinesize) bytes; memory: $(sysctl -n hw.memsize) bytes"
  echo "power: $(pmset -g batt | head -1)"
} > results/environment.txt
cat results/environment.txt

echo "### 1. Standard suite, tile 32, 3 repeats with rotated order"
[[ -f results/suite_tile32.csv ]] || "$BIN" suite --repeat 3 --tile 32 --samples $SAMPLES --output results/suite_tile32.csv

echo "### 2. Tile sweep (blocked)"
[[ -f results/tile_sweep.csv ]] || for shape in "32 32 32" "64 64 64" "128 128 128" "256 256 256" "512 512 512" "1024 1024 1024" "1024 128 512" "128 1024 64" "127 255 63"; do
  set -- ${=shape}
  for tile in 16 32 64 128 256; do
    "$BIN" bench --impl blocked --m $1 --k $2 --n $3 --tile $tile --samples $SAMPLES --output results/tile_sweep.csv
  done
done

BEST_TILE=$(python3 -I scripts/best_tile.py results/tile_sweep.csv)
echo "best tile by mean GFLOP/s over the large shapes: $BEST_TILE"
echo "best_tile: $BEST_TILE" >> results/environment.txt

echo "### 3. Thread scaling (parallel, tile $BEST_TILE)"
[[ -f results/thread_scaling.csv ]] || for shape in "64 64 64" "128 128 128" "256 256 256" "512 512 512" "1024 1024 1024" "1024 128 512" "128 1024 64"; do
  set -- ${=shape}
  for threads in 1 2 4 8 10; do
    "$BIN" bench --impl parallel --m $1 --k $2 --n $3 --tile $BEST_TILE --threads $threads --samples $SAMPLES --output results/thread_scaling.csv
  done
done

echo "### 4. Accelerate: default threading vs VECLIB_MAXIMUM_THREADS=1"
[[ -f results/accelerate_threads.csv ]] || for shape in "128 128 128" "512 512 512" "1024 1024 1024" "1024 128 512"; do
  set -- ${=shape}
  "$BIN" bench --impl accelerate --m $1 --k $2 --n $3 --samples $SAMPLES --output results/accelerate_threads.csv
  VECLIB_MAXIMUM_THREADS=1 "$BIN" bench --impl accelerate --m $1 --k $2 --n $3 --samples $SAMPLES --output results/accelerate_threads.csv
done

echo "### 5. Standard suite again with the best tile, 2 repeats"
[[ -f results/suite_best.csv ]] || "$BIN" suite --repeat 2 --tile $BEST_TILE --samples $SAMPLES --output results/suite_best.csv

echo "done: $(date -Iseconds)"

# ---- Extra experiment: register-blocked micro-kernel ------------------------
echo "### 6. Micro-kernel k-block sweep"
[[ -f results/micro_tile_sweep.csv ]] || for shape in "256 256 256" "512 512 512" "1024 1024 1024" "1024 128 512" "128 1024 64" "127 255 63"; do
  set -- ${=shape}
  for tile in 16 32 64 128 256 512 1024; do
    "$BIN" bench --impl micro --m $1 --k $2 --n $3 --tile $tile --samples $SAMPLES --output results/micro_tile_sweep.csv
  done
done
BEST_MICRO_TILE=$(python3 -I scripts/best_tile.py results/micro_tile_sweep.csv)
echo "best micro k-block: $BEST_MICRO_TILE"
grep -q best_micro_tile results/environment.txt || echo "best_micro_tile: $BEST_MICRO_TILE" >> results/environment.txt

echo "### 7. Micro-kernel thread scaling"
[[ -f results/micro_thread_scaling.csv ]] || for shape in "64 64 64" "128 128 128" "256 256 256" "512 512 512" "1024 1024 1024" "1024 128 512" "128 1024 64"; do
  set -- ${=shape}
  for threads in 1 2 4 8 10; do
    "$BIN" bench --impl micro_parallel --m $1 --k $2 --n $3 --tile $BEST_MICRO_TILE --threads $threads --samples $SAMPLES --output results/micro_thread_scaling.csv
  done
done

echo "### 8. 2048^3: working set (48 MB) exceeds the 16 MB L2"
if [[ ! -f results/large_2048.csv ]]; then
  "$BIN" bench --impl reordered --m 2048 --k 2048 --n 2048 --samples $SAMPLES --output results/large_2048.csv
  for tile in 64 128 256; do
    "$BIN" bench --impl blocked --m 2048 --k 2048 --n 2048 --tile $tile --samples $SAMPLES --output results/large_2048.csv
  done
  for tile in 64 256 1024; do
    "$BIN" bench --impl micro --m 2048 --k 2048 --n 2048 --tile $tile --samples $SAMPLES --output results/large_2048.csv
  done
  "$BIN" bench --impl parallel --m 2048 --k 2048 --n 2048 --tile $BEST_TILE --samples $SAMPLES --output results/large_2048.csv
  "$BIN" bench --impl micro_parallel --m 2048 --k 2048 --n 2048 --tile $BEST_MICRO_TILE --samples $SAMPLES --output results/large_2048.csv
  "$BIN" bench --impl accelerate --m 2048 --k 2048 --n 2048 --samples $SAMPLES --output results/large_2048.csv
fi

echo "### 9. Standard suite for the micro-kernel variants, 2 repeats"
[[ -f results/suite_micro.csv ]] || "$BIN" suite --impls micro,micro_parallel --repeat 2 --tile $BEST_MICRO_TILE --samples $SAMPLES --output results/suite_micro.csv
echo "done (extra): $(date -Iseconds)"

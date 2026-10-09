#!/bin/zsh
# CPU-profile the baseline and the best implementations on the same workload.
# With full Xcode installed this records Instruments Time Profiler traces via
# xctrace; otherwise it falls back to the Command Line Tools' `sample`, which is
# also a periodic call-stack sampler (1 ms interval) and reports the same
# "where does CPU time go" breakdown as text.
#   scripts/profile.sh [build_dir]
set -euo pipefail
cd "$(dirname "$0")/.."
BIN="${1:-build}/macmat"
OUT=results/profiles
mkdir -p "$OUT"
WORKLOAD=(--m 1024 --k 1024 --n 1024 --warmup 1)

have_xctrace() { xcrun --find xctrace >/dev/null 2>&1 && xctrace version >/dev/null 2>&1; }

# sample counts chosen so each run lasts ~5 s (naive ~0.9 s/call, parallel ~17 ms, micro_parallel ~7 ms)
for spec in "naive 32 6" "parallel 256 300" "micro_parallel 1024 700"; do
  set -- ${=spec}
  impl=$1; tile=$2; samples=$3
  if have_xctrace; then
    rm -rf "$OUT/${impl}_1024.trace"
    xctrace record --template 'Time Profiler' --output "$OUT/${impl}_1024.trace" \
      --launch -- "$BIN" bench --impl "$impl" --tile "$tile" --samples "$samples" "${WORKLOAD[@]}"
    xctrace export --input "$OUT/${impl}_1024.trace" \
      --xpath '/trace-toc/run[@number="1"]/data/table[@schema="time-profile"]' \
      > "$OUT/${impl}_1024_time_profile.xml"
  else
    echo "xctrace unavailable (needs Xcode); using /usr/bin/sample for $impl"
    "$BIN" bench --impl "$impl" --tile "$tile" --samples "$samples" "${WORKLOAD[@]}" > "$OUT/${impl}_1024_bench.txt" &
    pid=$!
    sleep 0.5
    sample "$pid" 4 1 -mayDie -file "$OUT/${impl}_1024_sample.txt" >/dev/null
    wait $pid || true
  fi
done
ls -la "$OUT"

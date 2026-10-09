"""Prints the tile size with the highest mean GFLOP/s across the large and
rectangular shapes in a tile-sweep CSV (used by run_experiments.sh)."""
import csv
import sys
from collections import defaultdict

rows = list(csv.DictReader(open(sys.argv[1])))
big = [r for r in rows if int(r["M"]) * int(r["K"]) * int(r["N"]) >= 512 ** 3 / 8]
score = defaultdict(list)
for r in big:
    score[int(r["tile"])].append(float(r["gflops"]))
best = max(score, key=lambda t: sum(score[t]) / len(score[t]))
print(best)

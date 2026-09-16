#!/bin/sh
# Construction time against the number of workers.
#   bench/scaling.sh [-n 1000000 -alpha 4 -d 2 -data uniform]
# Threads: $THREADS (default "1 2 4 8 14"), or the machine's core count.
set -e
cd "$(dirname "$0")/.."
BUILD=${BUILD:-build}
make -s BUILD="$BUILD" bench
for t in ${THREADS:-1 2 4 8 14}; do
  PARLAY_NUM_THREADS=$t "$BUILD"/bench/bench_build -seq 0 -rounds 2 "$@" | sed "s/^/[$t threads] /"
done

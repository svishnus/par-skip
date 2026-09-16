#!/bin/sh
# Measures every milestone of the implementation with the same benchmark and
# writes bench/results/history.csv (milestone,n,metric,value), then the
# thread scaling and the query modes of the current tree. bench/plot.py turns
# the CSV into docs/plots/*.svg.
#
#   bench/history.sh            # ~10 minutes on 14 cores; needs the submodule
#   NS="100000 200000" bench/history.sh
#
# Each milestone is checked out into a temporary worktree (sharing this
# checkout's ParlayLib) and built with the flags of its own Makefile.
# Uniform points in [0,1)^2, L2, alpha 4, seed 1.
set -eu
cd "$(dirname "$0")/.."
NS=${NS:-"100000 200000 500000 1000000"}
THREADS=${THREADS:-"1 2 4 8 14"}
OUT=bench/results/history.csv
TMP=$(mktemp -d)
trap 'for w in "$TMP"/*/; do git worktree remove --force "$w" >/dev/null 2>&1 || true; done; rm -rf "$TMP"' EXIT
mkdir -p bench/results

# label:commit:advflag  (advflag: 1 when bench_build/bench_query take -adv)
MILESTONES="
p1-sequential-Alg2:8f3c960:0
p2-parallel-Alg6:8f3c960:0
p4-advance-pointers:1c49d72:1
p4b-review-fixes:a357189:1
p5-reserved-arrays:HEAD:1
"

# Runs a command under /usr/bin/time, writing its output plus the peak
# resident set to $log (macOS -l or GNU -v).
timed() {
  log=$1; shift
  if [ "$(uname)" = Darwin ]; then /usr/bin/time -l "$@" > "$log" 2>&1; else /usr/bin/time -v "$@" > "$log" 2>&1; fi
}
rss_of() { awk '/maximum resident set size/ {print $1} /Maximum resident set size/ {print $NF * 1024}' "$1"; }

echo "milestone,n,metric,value" > "$OUT"
for m in $MILESTONES; do
  label=${m%%:*}; rest=${m#*:}; commit=${rest%%:*}; adv=${rest#*:}
  wt="$TMP/$label"
  git worktree add -q "$wt" "$commit"
  rm -rf "$wt/external/parlaylib" && ln -s "$PWD/external/parlaylib" "$wt/external/parlaylib"
  make -s -C "$wt" bench >/dev/null
  case $label in p1*) seq=1; rounds=1;; *) seq=0; rounds=2;; esac  # parallel: best of two runs
  case $adv in 1) advflag="-adv 1";; *) advflag="";; esac
  for n in $NS; do
    log="$TMP/$label-$n.log"
    # shellcheck disable=SC2086
    timed "$log" "$wt/build/bench/bench_build" -n "$n" -alpha 4 -d 2 -rounds $rounds -seq $seq $advflag
    rss=$(rss_of "$log")
    if [ "$seq" = 1 ]; then
      t=$(awk '/^ *sequential/ {print $2}' "$log"); echo "$label,$n,build_seconds,$t" >> "$OUT"
    else
      t=$(awk '/^ *parallel/ {print $2}' "$log"); echo "$label,$n,build_seconds,$t" >> "$OUT"
    fi
    echo "$label,$n,rss_bytes,$rss" >> "$OUT"
    lists=$(awk '/^ *structure/ {print $2}' "$log"); echo "$label,$n,lists_per_point,$lists" >> "$OUT"
    if [ "$n" = 1000000 ]; then
      # shellcheck disable=SC2086
      "$wt/build/bench/bench_query" -n "$n" -alpha 4 -d 2 -q 100000 -k 10 $advflag > "$log.q"
      echo "$label,$n,nearest_per_second,$(awk '/^ *nearest/ {v=$7} END {print v * 1e6}' "$log.q")" >> "$OUT"
      echo "$label,$n,knn10_per_second,$(awk '/^ *knn/ {print $8 * 1e6}' "$log.q")" >> "$OUT"
    fi
    echo "  $label n=$n: $t s" >&2
  done
done

# current tree: scaling with the number of workers, both navigation modes
make -s bench >/dev/null
for t in $THREADS; do
  for adv in 0 1; do
    v=$(PARLAY_NUM_THREADS=$t build/bench/bench_build -n 1000000 -alpha 4 -d 2 -rounds 2 -seq 0 -adv $adv | awk '/^ *parallel/ {print $2}')
    echo "current-adv$adv,$t,build_seconds_by_workers,$v" >> "$OUT"
    echo "  workers=$t adv=$adv: $v s" >&2
  done
done
for adv in 0 1; do
  build/bench/bench_query -n 1000000 -alpha 4 -d 2 -q 100000 -k 10 -adv $adv > "$TMP/q$adv.log"
  echo "current-adv$adv,1000000,nearest_per_second,$(awk '/^ *nearest/ {v=$7} END {print v * 1e6}' "$TMP/q$adv.log")" >> "$OUT"
  echo "current-adv$adv,1000000,knn10_per_second,$(awk '/^ *knn/ {print $8 * 1e6}' "$TMP/q$adv.log")" >> "$OUT"
done
echo "wrote $OUT" >&2

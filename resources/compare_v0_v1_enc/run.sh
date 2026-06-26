#!/usr/bin/env bash
# Reproduce the encoder-0 latency sweep: build the 3 benchmarks (f32 / v0 / v5),
# run a thread sweep, and aggregate into summary.md + summary.csv (this folder).
#
# The NAS/QNAP mount is noexec, so by default we stage a runnable copy in a local
# SANDBOX, build+run there, and copy results.csv back here. On a runnable checkout,
# omit SANDBOX to build+run in place.
#
#   # noexec mount (stage to local disk):
#   SANDBOX=/tmp/encbench FFTW_INC=/path/to/fftw3/local/include ./run.sh
#   # runnable checkout (in place):
#   FFTW_INC=/path/to/fftw3/local/include ./run.sh
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${ROOT:-$HERE/../..}"
FFTW_INC="${FFTW_INC:-$ROOT/../fftw3/local/include}"
SANDBOX="${SANDBOX:-}"
MPS="${MPS:-1 2 4 8 16}"
WARM="${WARM:-5}"
ITERS="${ITERS:-50}"

build_and_run() {                 # $1 = project root that holds sources/headers/data/resources
    local P="$1"
    make -C "$P/resources/compare_v0_v1_enc" ROOT=../.. FFTW_INC="$FFTW_INC"
    local R="$P/resources/compare_v0_v1_enc/results.csv"
    echo "version,mp,iters,mean_ssm_us,mean_mlp_us,mean_tot_us,min_ssm_us,min_mlp_us,min_tot_us" > "$R"
    local B="$P/resources/compare_v0_v1_enc"
    for mp in $MPS; do
        OMP_NUM_THREADS=$mp "$B/bench_f32" "$mp" "$WARM" "$ITERS" "$P/data/weight_f32"        >> "$R"
        OMP_NUM_THREADS=$mp "$B/bench_v0"  "$mp" "$WARM" "$ITERS" "$P/data/weight_sq_int8_v0" >> "$R"
        OMP_NUM_THREADS=$mp "$B/bench_v5"  "$mp" "$WARM" "$ITERS" "$P/data/weight_sq_int8_v5" >> "$R"
        echo "  done mp=$mp"
    done
    [ "$R" = "$HERE/results.csv" ] || cp "$R" "$HERE/results.csv"
}

if [ -n "$SANDBOX" ]; then
    echo "Staging runnable copy in $SANDBOX (noexec-mount mode)"
    rm -rf "$SANDBOX"; mkdir -p "$SANDBOX/resources"
    cp -a "$ROOT/headers" "$ROOT/sources" "$SANDBOX/"
    cp -a "$HERE" "$SANDBOX/resources/compare_v0_v1_enc"
    for v in f32 sq_int8_v0 sq_int8_v5; do
        mkdir -p "$SANDBOX/data/weight_$v"
        cp -a "$ROOT/data/weight_$v/pvss_ds_weight_enc0_mag" "$SANDBOX/data/weight_$v/"
    done
    build_and_run "$SANDBOX"
else
    build_and_run "$ROOT"
fi

python3 "$HERE/aggregate.py" "$HERE/results.csv"
echo "Done -> $HERE/summary.md , $HERE/summary.csv"

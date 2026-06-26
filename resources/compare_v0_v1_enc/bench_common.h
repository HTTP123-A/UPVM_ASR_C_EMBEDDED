/* Shared helpers for the encoder-0 latency micro-benchmark.
 * (resources/compare_v0_v1_enc — the only folder this benchmark is allowed to add to.) */
#ifndef BENCH_COMMON_H
#define BENCH_COMMON_H

#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <omp.h>

/* (b - a) in nanoseconds */
static inline double ts_ns(const struct timespec *a, const struct timespec *b)
{
    return (double)(b->tv_sec - a->tv_sec) * 1e9 + (double)(b->tv_nsec - a->tv_nsec);
}

/* Deterministic xorshift64 fill into [-1, 1]. Same seed => the f32, v0 and v5
 * benchmarks all see byte-identical input activations, so the comparison is fair. */
static inline void fill_rand(float *p, size_t n, uint64_t seed)
{
    uint64_t s = seed ? seed : 0x9E3779B97F4A7C15ULL;
    for (size_t i = 0; i < n; ++i) {
        s ^= s << 13; s ^= s >> 7; s ^= s << 17;
        p[i] = (float)((double)(s >> 11) / (double)(1ULL << 53)) * 2.0f - 1.0f;
    }
}

#endif /* BENCH_COMMON_H */

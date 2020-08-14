// Author:  Niels A.D.
// Project: HGB (https://github.com/nielsAD/hgb)
// License: Mozilla Public License, v2.0

#pragma once

#include "util/ubiquitous.h"
#include "util/openmp.h"

typedef long double time_mark_t;
typedef long double time_diff_t;

static inline time_mark_t time_mark() {
    return (time_mark_t) omp_get_wtime();
}

static inline time_diff_t time_since(const time_mark_t mark) {
    return time_mark() - mark;
}

static inline time_diff_t time_ms(const time_diff_t diff) {
    return diff * (time_diff_t)1e3;
}

static inline time_diff_t time_us(const time_diff_t diff) {
    return diff * (time_diff_t)1e6;
}

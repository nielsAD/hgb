// Author:  Niels A.D.
// Project: HGB (https://github.com/nielsAD/hgb)
// License: Mozilla Public License, v2.0

#include "alg/pagerank/omp_codelets.h"
#include "util/math.h"

void omp_pagerank_read_col(pr_float *restrict _dst, const pr_float *restrict _src, const graph_size_t dst_offset, const graph_size_t dst_cols, const graph_size_t src_cols, const graph_size_t size)
{
          pr_float *restrict dst = PAGERANK_ASSUME_ALIGNED(_dst);
    const pr_float *restrict src = PAGERANK_ASSUME_ALIGNED(_src);

    const graph_size_t src_rows = DIVIDE_BY_INC(size, src_cols);

    dst = &dst[dst_offset];

    OMP_PRAGMA(omp parallel for)
    for (graph_size_t row_idx = 0; row_idx < src_rows; row_idx++)
    {
        OMP_PRAGMA(omp simd)
        for (graph_size_t col_idx = 0; col_idx < src_cols; col_idx++)
            dst[row_idx*dst_cols + col_idx] = src[row_idx*src_cols + col_idx];
    }
}

void omp_pagerank_fill_arr(pr_float *_arr, const pr_float val, const graph_size_t size)
{
    pr_float *restrict arr = PAGERANK_ASSUME_ALIGNED(_arr);

    OMP_PRAGMA(omp parallel for)
    for (graph_size_t idx = 0; idx < size; idx++)
        arr[idx] = val;
}

void omp_pagerank_add_arr(pr_float *_a, const pr_float *_b, const graph_size_t size)
{
          pr_float *restrict a = PAGERANK_ASSUME_ALIGNED(_a);
    const pr_float *restrict b = PAGERANK_ASSUME_ALIGNED(_b);

    OMP_PRAGMA(omp parallel for)
    for (graph_size_t idx = 0; idx < size; idx++)
        a[idx] += b[idx];
}

pr_float omp_pagerank_sum_arr(const pr_float *_arr, const graph_size_t size)
{
    const pr_float *restrict arr = PAGERANK_ASSUME_ALIGNED(_arr);

    pr_float sum = 0.0;

    OMP_PRAGMA(omp parallel for reduction(+:sum))
    for (graph_size_t idx = 0; idx < size; idx++)
        sum += arr[idx];

    return sum;
}

pr_float omp_pagerank_baserank(const pr_float *_src, const graph_size_t *_deg, const graph_size_t vcount)
{
    const pr_float     *restrict src = PAGERANK_ASSUME_ALIGNED(_src);
    const graph_size_t *restrict deg = PAGERANK_ASSUME_ALIGNED(_deg);

    pr_float rank = 0.0;

    OMP_PRAGMA(omp parallel for reduction(+:rank))
    for (graph_size_t node = 0; node < vcount; node++)
        if (UNLIKELY(deg[node] == 0))
            rank += src[node];

    return rank;
}

void omp_pagerank_baserank_mapped(pr_float *_tmp, const pr_float *_src, const graph_size_t *_deg, const graph_size_t vcount)
{
          pr_float     *restrict tmp = PAGERANK_ASSUME_ALIGNED(_tmp);
    const pr_float     *restrict src = PAGERANK_ASSUME_ALIGNED(_src);
    const graph_size_t *restrict deg = PAGERANK_ASSUME_ALIGNED(_deg);

    OMP_PRAGMA(omp parallel for)
    for (graph_size_t node = 0; node < vcount; node++)
    {
        //ISO C -> comparison yields 1 if equal
        tmp[node] = (deg[node] == 0) * src[node];
    }
}

void omp_pagerank_update_rank_pull(pr_float *_dst, const pr_float *_src, const graph_size_t *_rid, const graph_size_t *_cid, const graph_size_t *_deg, const graph_size_t vcount)
{
          pr_float     *restrict dst = PAGERANK_ASSUME_ALIGNED(_dst);
    const pr_float     *restrict src = PAGERANK_ASSUME_ALIGNED(_src);
    const graph_size_t *restrict rid = PAGERANK_ASSUME_ALIGNED(_rid);
    const graph_size_t *restrict cid = PAGERANK_ASSUME_ALIGNED(_cid);
    const graph_size_t *restrict deg = PAGERANK_ASSUME_ALIGNED(_deg);

    OMP_PRAGMA(omp parallel for schedule(guided))
    for (graph_size_t node = 0; node < vcount; node++)
    {
        const graph_size_t efr = rid[node];
        const graph_size_t eto = rid[node + 1];

        pr_float new_rank = 0.0;

        OMP_PRAGMA(omp simd reduction(+:new_rank))
        for (graph_size_t edge = efr; edge < eto; edge++)
            new_rank += src[cid[edge]] / deg[cid[edge]];

        dst[node] += new_rank;
    }
}

void omp_pagerank_update_rank_push(pr_float *_dst, const pr_float *_src, const graph_size_t *_rid, const graph_size_t *_cid, const graph_size_t *_deg, const graph_size_t vcount)
{
          pr_float     *restrict dst = PAGERANK_ASSUME_ALIGNED(_dst);
    const pr_float     *restrict src = PAGERANK_ASSUME_ALIGNED(_src);
    const graph_size_t *restrict rid = PAGERANK_ASSUME_ALIGNED(_rid);
    const graph_size_t *restrict cid = PAGERANK_ASSUME_ALIGNED(_cid);
    const graph_size_t *restrict deg = PAGERANK_ASSUME_ALIGNED(_deg);

    OMP_PRAGMA(omp parallel for schedule(guided))
    for (graph_size_t node = 0; node < vcount; node++)
    {
        const graph_size_t efr = rid[node];
        const graph_size_t eto = rid[node + 1];

        const pr_float send_rank = src[node] / deg[node];

        for (graph_size_t edge = efr; edge < eto; edge++)
            OMP_PRAGMA(omp atomic)
            dst[cid[edge]] += send_rank;
    }
}

void omp_pagerank_update_tmp(pr_float *_tmp, const pr_float *_src, const graph_size_t *_deg, const graph_size_t vcount)
{
          pr_float     *restrict tmp = PAGERANK_ASSUME_ALIGNED(_tmp);
    const pr_float     *restrict src = PAGERANK_ASSUME_ALIGNED(_src);
    const graph_size_t *restrict deg = PAGERANK_ASSUME_ALIGNED(_deg);

    OMP_PRAGMA(omp parallel for)
    for (graph_size_t node = 0; node < vcount; node++)
        tmp[node] = src[node] / deg[node];
}

void omp_pagerank_update_rank_tmp_pull(pr_float *_dst, const pr_float *_tmp, const graph_size_t *_rid, const graph_size_t *_cid, const graph_size_t vcount)
{
          pr_float     *restrict dst = PAGERANK_ASSUME_ALIGNED(_dst);
    const pr_float     *restrict tmp = PAGERANK_ASSUME_ALIGNED(_tmp);
    const graph_size_t *restrict rid = PAGERANK_ASSUME_ALIGNED(_rid);
    const graph_size_t *restrict cid = PAGERANK_ASSUME_ALIGNED(_cid);

    OMP_PRAGMA(omp parallel for schedule(guided))
    for (graph_size_t node = 0; node < vcount; node++)
    {
        const graph_size_t efr = rid[node];
        const graph_size_t eto = rid[node + 1];

        pr_float new_rank = 0.0;

        OMP_PRAGMA(omp simd reduction(+:new_rank))
        for (graph_size_t edge = efr; edge < eto; edge++)
            new_rank += tmp[cid[edge]];

        dst[node] += new_rank;
    }
}

void omp_pagerank_update_rank_tmp_push(pr_float *_dst, const pr_float *_tmp, const graph_size_t *_rid, const graph_size_t *_cid, const graph_size_t vcount)
{
          pr_float     *restrict dst = PAGERANK_ASSUME_ALIGNED(_dst);
    const pr_float     *restrict tmp = PAGERANK_ASSUME_ALIGNED(_tmp);
    const graph_size_t *restrict rid = PAGERANK_ASSUME_ALIGNED(_rid);
    const graph_size_t *restrict cid = PAGERANK_ASSUME_ALIGNED(_cid);

    OMP_PRAGMA(omp parallel for schedule(guided))
    for (graph_size_t node = 0; node < vcount; node++)
    {
        const graph_size_t efr = rid[node];
        const graph_size_t eto = rid[node + 1];

        for (graph_size_t edge = efr; edge < eto; edge++)
            OMP_PRAGMA(omp atomic)
            dst[cid[edge]] += tmp[node];
    }
}

void omp_pagerank_update_rank_tmp_pull_binsearch(pr_float *_dst, const pr_float *_tmp, const graph_size_t *_rid, const graph_size_t *_cid, const graph_size_t vcount)
{
          pr_float     *restrict dst = PAGERANK_ASSUME_ALIGNED(_dst);
    const pr_float     *restrict tmp = PAGERANK_ASSUME_ALIGNED(_tmp);
    const graph_size_t *restrict rid = PAGERANK_ASSUME_ALIGNED(_rid);
    const graph_size_t *restrict cid = PAGERANK_ASSUME_ALIGNED(_cid);

    OMP_PRAGMA(omp parallel)
    {
        const graph_size_t thread_idx = omp_get_thread_num();
        if (thread_idx < vcount)
        {
            const graph_size_t ecount = rid[vcount];

            const graph_size_t thread_num = omp_get_num_threads();
            const graph_size_t thread_len = DIVIDE_BY_INC(ecount, thread_num);

            const graph_size_t efr = thread_idx * thread_len;
            const graph_size_t eto = MIN(efr + thread_len, ecount);

            size_t vfr = 0;
            if (!BINARY_SEARCH(efr, _rid, 0, vcount, vfr))
            {
                // Process partial node on the left side
                const graph_size_t eto = rid[vfr];

                pr_float new_rank = 0.0;

                OMP_PRAGMA(omp simd reduction(+:new_rank))
                for (graph_size_t edge = efr; edge < eto; edge++)
                    new_rank += tmp[cid[edge]];

                OMP_PRAGMA(omp atomic)
                dst[vfr - 1] += new_rank;
            }

            size_t vto = 0;
            if (!BINARY_SEARCH(eto, _rid, 0, vcount, vto))
            {
                // Process partial node on the right side
                vto--;
                const graph_size_t efr = rid[vto];

                pr_float new_rank = 0.0;

                OMP_PRAGMA(omp simd reduction(+:new_rank))
                for (graph_size_t edge = efr; edge < eto; edge++)
                    new_rank += tmp[cid[edge]];

                OMP_PRAGMA(omp atomic)
                dst[vto] += new_rank;
            }

            // Process assigned nodes
            for (graph_size_t node = vfr; node < vto; node++)
            {
                const graph_size_t efr = rid[node];
                const graph_size_t eto = rid[node + 1];

                pr_float new_rank = 0.0;

                OMP_PRAGMA(omp simd reduction(+:new_rank))
                for (graph_size_t edge = efr; edge < eto; edge++)
                    new_rank += tmp[cid[edge]];

                dst[node] += new_rank;
            }
        }
    }
}

void omp_pagerank_update_rank_tmp_push_binsearch(pr_float *_dst, const pr_float *_tmp, const graph_size_t *_rid, const graph_size_t *_cid, const graph_size_t vcount)
{
          pr_float     *restrict dst = PAGERANK_ASSUME_ALIGNED(_dst);
    const pr_float     *restrict tmp = PAGERANK_ASSUME_ALIGNED(_tmp);
    const graph_size_t *restrict rid = PAGERANK_ASSUME_ALIGNED(_rid);
    const graph_size_t *restrict cid = PAGERANK_ASSUME_ALIGNED(_cid);

    OMP_PRAGMA(omp parallel)
    {
        const graph_size_t thread_idx = omp_get_thread_num();
        if (thread_idx < vcount)
        {
            const graph_size_t ecount = rid[vcount];

            const graph_size_t thread_num = omp_get_num_threads();
            const graph_size_t thread_len = DIVIDE_BY_INC(ecount, thread_num);

            const graph_size_t efr = thread_idx * thread_len;
            const graph_size_t eto = MIN(efr + thread_len, ecount);

            size_t vfr = 0;
            if (!BINARY_SEARCH(efr, _rid, 0, vcount, vfr))
            {
                // Process partial node on the left side
                const graph_size_t eto = rid[vfr];

                for (graph_size_t edge = efr; edge < eto; edge++)
                    OMP_PRAGMA(omp atomic)
                    dst[cid[edge]] += tmp[vfr - 1];
            }

            size_t vto = 0;
            if (!BINARY_SEARCH(eto, _rid, 0, vcount, vto))
            {
                // Process partial node on the right side
                vto--;
                const graph_size_t efr = rid[vto];

                for (graph_size_t edge = efr; edge < eto; edge++)
                    OMP_PRAGMA(omp atomic)
                    dst[cid[edge]] += tmp[vto];
            }

            // Process assigned nodes
            for (graph_size_t node = vfr; node < vto; node++)
            {
                const graph_size_t efr = rid[node];
                const graph_size_t eto = rid[node + 1];

                for (graph_size_t edge = efr; edge < eto; edge++)
                    OMP_PRAGMA(omp atomic)
                    dst[cid[edge]] += tmp[node];
            }
        }
    }
}

void omp_pagerank_update_dest(pr_float *_dst, const pr_float base_rank, const pr_float damping, const graph_size_t vcount)
{
    pr_float *restrict dst = PAGERANK_ASSUME_ALIGNED(_dst);

    OMP_PRAGMA(omp parallel for)
    for (graph_size_t node = 0; node < vcount; node++)
        dst[node] = (damping * dst[node]) + base_rank;
}

void omp_pagerank_calc_dest(pr_float *_dst, pr_float *_tmp, const pr_float base_rank, const pr_float damping, const graph_size_t vcount)
{
          pr_float *restrict dst = PAGERANK_ASSUME_ALIGNED(_dst);
    const pr_float *restrict tmp = PAGERANK_ASSUME_ALIGNED(_tmp);

    OMP_PRAGMA(omp parallel for)
    for (graph_size_t node = 0; node < vcount; node++)
        dst[node] = (damping * tmp[node]) + base_rank;
}

pr_float omp_pagerank_calc_diff(const pr_float *_src, const pr_float *_dst, const graph_size_t vcount)
{
    const pr_float *restrict src = PAGERANK_ASSUME_ALIGNED(_src);
    const pr_float *restrict dst = PAGERANK_ASSUME_ALIGNED(_dst);

    pr_float diff = 0.0;

    OMP_PRAGMA(omp parallel for reduction(+:diff))
    for (graph_size_t node = 0; node < vcount; node++)
        diff += pr_abs(src[node] - dst[node]);

    return diff;
}

void omp_pagerank_calc_diff_mapped(pr_float *_tmp, const pr_float *_src, const pr_float *_dst, const graph_size_t vcount)
{
          pr_float *restrict tmp = PAGERANK_ASSUME_ALIGNED(_tmp);
    const pr_float *restrict src = PAGERANK_ASSUME_ALIGNED(_src);
    const pr_float *restrict dst = PAGERANK_ASSUME_ALIGNED(_dst);

    OMP_PRAGMA(omp parallel for)
    for (graph_size_t node = 0; node < vcount; node++)
        tmp[node] = pr_abs(src[node] - dst[node]);
}

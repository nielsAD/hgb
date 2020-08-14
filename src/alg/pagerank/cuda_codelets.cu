// Author:  Niels A.D.
// Project: HGB (https://github.com/nielsAD/hgb)
// License: Mozilla Public License, v2.0

#include "alg/pagerank/cuda_codelets.h"

#define kernel __global__ static
#define shared __shared__

#define launch_kernel(kernel,...) { \
    int gs, bs; CUDA_ASSERT(cudaOccupancyMaxPotentialBlockSize(&gs, &bs, kernel, 0, 0)); \
    kernel<<<DIVIDE_BY_INC(LAST_ARG(__VA_ARGS__), bs), bs, 0, stream>>>(__VA_ARGS__); \
    CUDA_ASSERT(cudaPeekAtLastError()); \
}

#define launch_kernel_max(kernel,...) { \
    int gs, bs; CUDA_ASSERT(cudaOccupancyMaxPotentialBlockSize(&gs, &bs, kernel, 0, 0)); \
    kernel<<<gs, bs, 0, stream>>>(__VA_ARGS__); \
    CUDA_ASSERT(cudaPeekAtLastError()); \
}

void cuda_pagerank_read_col(const cudaStream_t stream, pr_float *dst, const pr_float *src, const graph_size_t dst_offset, const graph_size_t dst_cols, const graph_size_t src_cols, const graph_size_t size)
{
    CUDA_ASSERT(cudaMemcpy2DAsync(
        &dst[dst_offset],
        dst_cols * sizeof(*dst),
        src,
        src_cols * sizeof(*src),
        src_cols * sizeof(*src),
        DIVIDE_BY_INC(size, src_cols),
        cudaMemcpyDeviceToHost,
        stream
    ));
}

kernel void _cuda_pagerank_fill_arr_single(
          pr_float *arr,
    const pr_float  val,
    const graph_size_t size)
{
    const graph_size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size)
        arr[idx] = val;
}

void cuda_pagerank_fill_arr(const cudaStream_t stream, pr_float *arr, const pr_float val, const graph_size_t size)
{
    launch_kernel(_cuda_pagerank_fill_arr_single, arr, val, size);
}

kernel void _cuda_pagerank_redux_add_single(
          pr_float *restrict a,
    const pr_float *restrict b,
    const graph_size_t size)
{
    const graph_size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size)
        a[idx] += b[idx];
}

void cuda_pagerank_add_arr(const cudaStream_t stream, pr_float *restrict a, const pr_float *restrict b, const graph_size_t size)
{
    launch_kernel(_cuda_pagerank_redux_add_single, a, b, size);
}

kernel void _cuda_pagerank_redux_sum(
          pr_float *res,
    const pr_float *arr,
    const graph_size_t size)
{
    pr_float tmp_res = 0.0;
    for (graph_size_t idx = 0; idx < size; idx++)
        tmp_res += arr[idx];
    *res += tmp_res;
}

void cuda_pagerank_sum_arr(const cudaStream_t stream, pr_float *res, const pr_float *arr, const graph_size_t size)
{
    _cuda_pagerank_redux_sum<<<1, 1, 0, stream>>>
        (res, arr, size);
    CUDA_ASSERT(cudaPeekAtLastError());
}

kernel void _cuda_pagerank_redux_sum_offset(
          pr_float *res,
    const pr_float *arr,
    const graph_size_t offset,
    const graph_size_t size)
{
    pr_float tmp_res = 0.0;
    for (graph_size_t idx = 0; idx < size; idx++)
        tmp_res += arr[idx];
    res[offset] = tmp_res;
}

void cuda_pagerank_sum_arr_offset(const cudaStream_t stream, pr_float *res, const pr_float *arr, const graph_size_t offset, const graph_size_t size)
{
    _cuda_pagerank_redux_sum_offset<<<1, 1, 0, stream>>>
        (res, arr, offset, size);
    CUDA_ASSERT(cudaPeekAtLastError());
}

//https://github.com/sschaetz/nvidia-opencl-examples/tree/master/OpenCL/src/oclReduction
kernel void _cuda_pagerank_redux_sum_parallel(
          pr_float *res,
    const pr_float *arr,
    const graph_size_t size)
{
    #define lsi 128
    shared volatile pr_float scr[lsi];

    const graph_size_t lid = threadIdx.x;
    //const graph_size_t lsi = blockDim.x;
    const graph_size_t gid = blockIdx.x;
    const graph_size_t gsi = lsi*gridDim.x;

    pr_float sum = 0.0;

    for (graph_size_t idx = gid*lsi + lid; idx < size; idx += gsi)
        sum += arr[idx];

    scr[lid] = sum;
    __syncthreads();

    for(graph_size_t rdx = lsi >> 1; rdx > 0; rdx >>= 1)
    {
        if (lid < rdx)
            scr[lid] += scr[lid + rdx];
        __syncthreads();
    }

    if (lid == 0)
        res[gid] = scr[0];

    #undef lsi
}

void cuda_pagerank_parallel_sum_arr(const cudaStream_t stream, pr_float *res, const pr_float *arr, pr_float *scr, const graph_size_t arr_size, const graph_size_t scr_size)
{
    if (arr_size < 2*scr_size)
        return cuda_pagerank_sum_arr(stream, res, arr, arr_size);

    _cuda_pagerank_redux_sum_parallel<<<scr_size, 128, 0, stream>>>
        (scr, arr, arr_size);
    CUDA_ASSERT(cudaPeekAtLastError());

    cuda_pagerank_sum_arr(stream, res, scr, scr_size);
}

void cuda_pagerank_parallel_sum_arr_offset(const cudaStream_t stream, pr_float *res, const pr_float *arr, pr_float *scr, const graph_size_t offset, const graph_size_t arr_size, const graph_size_t scr_size)
{
    if (arr_size < 2*scr_size)
        return cuda_pagerank_sum_arr_offset(stream, res, arr, offset, arr_size);

    _cuda_pagerank_redux_sum_parallel<<<scr_size, 128, 0, stream>>>
        (scr, arr, arr_size);
    CUDA_ASSERT(cudaPeekAtLastError());

    cuda_pagerank_sum_arr_offset(stream, res, scr, offset, scr_size);
}

kernel void _cuda_pagerank_baserank_single(
          pr_float     *restrict res,
    const pr_float     *restrict src,
    const graph_size_t *restrict deg,
    const graph_size_t vcount)
{
    const graph_size_t node = blockIdx.x * blockDim.x + threadIdx.x;
    if (node < vcount)
    {
        //ISO C -> comparison yields 1 if equal
        res[node] = (deg[node] == 0) * src[node];
    }
}

void cuda_pagerank_baserank(const cudaStream_t stream, pr_float *restrict res, const pr_float *restrict src, const graph_size_t *restrict deg, const graph_size_t vcount)
{
    launch_kernel(_cuda_pagerank_baserank_single, res, src, deg, vcount);
}

kernel void _cuda_pagerank_baserank_redux(
          pr_float *redux,
    const pr_float damping,
    const graph_size_t vcount)
{
    pr_float r = *redux;

    r /= vcount;
    r *= damping;
    r += (1.0 - damping) / vcount;

    *redux = r;
}

void cuda_pagerank_baserank_redux(const cudaStream_t stream, pr_float *redux, const pr_float damping, const graph_size_t vcount)
{
    _cuda_pagerank_baserank_redux<<<1, 1, 0, stream>>>
        (redux, damping, vcount);
    CUDA_ASSERT(cudaPeekAtLastError());
}

kernel void _cuda_pagerank_fill_cols_pull(
          pr_float     *restrict dst,
    const graph_size_t *restrict rid,
    const graph_size_t *restrict cid,
    const graph_size_t *restrict deg,
    const graph_size_t vcount)
{
    const graph_size_t node = blockIdx.x * blockDim.x + threadIdx.x;

    if (node < vcount)
    {
        const graph_size_t efr = rid[node];
        const graph_size_t eto = rid[node + 1];

        for (graph_size_t edge = efr; edge < eto; edge++)
            dst[edge] = 1.0 / deg[cid[edge]];
    }
}

void cuda_pagerank_fill_cols_pull(const cudaStream_t stream, pr_float *restrict dst, const graph_size_t *restrict rid, const graph_size_t *restrict cid, const graph_size_t *restrict deg, const graph_size_t vcount)
{
    launch_kernel(_cuda_pagerank_fill_cols_pull, dst, rid, cid, deg, vcount);
}

kernel void _cuda_pagerank_fill_cols_push(
          pr_float     *restrict dst,
    const graph_size_t *restrict rid,
    const graph_size_t *restrict cid,
    const graph_size_t *restrict deg,
    const graph_size_t vcount)
{
    const graph_size_t node = blockIdx.x * blockDim.x + threadIdx.x;

    if (node < vcount)
    {
        const graph_size_t efr = rid[node];
        const graph_size_t eto = rid[node + 1];
        const pr_float     val = 1.0 / deg[node];

        for (graph_size_t edge = efr; edge < eto; edge++)
            dst[edge] = val;
    }
}

void cuda_pagerank_fill_cols_push(const cudaStream_t stream, pr_float *restrict dst, const graph_size_t *restrict rid, const graph_size_t *restrict cid, const graph_size_t *restrict deg, const graph_size_t vcount)
{
    launch_kernel(_cuda_pagerank_fill_cols_push, dst, rid, cid, deg, vcount);
}

kernel void _cuda_pagerank_update_rank_pull_single(
          pr_float     *restrict dst,
    const pr_float     *restrict src,
    const graph_size_t *restrict rid,
    const graph_size_t *restrict cid,
    const graph_size_t *restrict deg,
    const graph_size_t vcount)
{
    const graph_size_t node = blockIdx.x * blockDim.x + threadIdx.x;

    if (node < vcount)
    {
        const graph_size_t efr = rid[node];
        const graph_size_t eto = rid[node + 1];

        pr_float new_rank = 0.0;

        for (graph_size_t edge = efr; edge < eto; edge++)
            new_rank += src[cid[edge]] / deg[cid[edge]];

        dst[node] += new_rank;
    }
}

void cuda_pagerank_update_rank_pull(const cudaStream_t stream, pr_float *restrict dst, const pr_float *restrict src, const graph_size_t *restrict rid, const graph_size_t *restrict cid, const graph_size_t *restrict deg, const graph_size_t vcount)
{
    launch_kernel(_cuda_pagerank_update_rank_pull_single, dst, src, rid, cid, deg, vcount);
}

kernel void _cuda_pagerank_update_rank_push_single(
          pr_float     *restrict dst,
    const pr_float     *restrict src,
    const graph_size_t *restrict rid,
    const graph_size_t *restrict cid,
    const graph_size_t *restrict deg,
    const graph_size_t vcount)
{
    const graph_size_t node = blockIdx.x * blockDim.x + threadIdx.x;

    if (node < vcount)
    {
        const graph_size_t efr = rid[node];
        const graph_size_t eto = rid[node + 1];

        const pr_float send_rank = src[node] / deg[node];

        for (graph_size_t edge = efr; edge < eto; edge++)
            atomicAdd(&dst[cid[edge]], send_rank);
    }
}

void cuda_pagerank_update_rank_push(const cudaStream_t stream, pr_float *restrict dst, const pr_float *restrict src, const graph_size_t *restrict rid, const graph_size_t *restrict cid, const graph_size_t *restrict deg, const graph_size_t vcount)
{
    launch_kernel(_cuda_pagerank_update_rank_push_single, dst, src, rid, cid, deg, vcount);
}

kernel void _cuda_pagerank_update_tmp_single(
          pr_float     *restrict tmp,
    const pr_float     *restrict src,
    const graph_size_t *restrict deg,
    const graph_size_t vcount)
{
    const graph_size_t node = blockIdx.x * blockDim.x + threadIdx.x;
    if (node < vcount)
        tmp[node] = src[node] / deg[node];
}

void cuda_pagerank_update_tmp(const cudaStream_t stream, pr_float *restrict tmp, const pr_float *restrict src, const graph_size_t *restrict deg, const graph_size_t vcount)
{
    launch_kernel(_cuda_pagerank_update_tmp_single, tmp, src, deg, vcount);
}

kernel void _cuda_pagerank_update_rank_tmp_pull_single(
          pr_float     *restrict dst,
    const pr_float     *restrict tmp,
    const graph_size_t *restrict rid,
    const graph_size_t *restrict cid,
    const graph_size_t vcount)
{
    const graph_size_t node = blockIdx.x * blockDim.x + threadIdx.x;
    if (node < vcount)
    {
        const graph_size_t efr = rid[node];
        const graph_size_t eto = rid[node + 1];

        pr_float new_rank = 0.0;

        for (graph_size_t edge = efr; edge < eto; edge++)
            new_rank += tmp[cid[edge]];

        dst[node] += new_rank;
    }
}

void cuda_pagerank_update_rank_tmp_pull(const cudaStream_t stream, pr_float *restrict dst, const pr_float *restrict tmp, const graph_size_t *restrict rid, const graph_size_t *restrict cid, const graph_size_t vcount)
{
    launch_kernel(_cuda_pagerank_update_rank_tmp_pull_single, dst, tmp, rid, cid, vcount);
}

kernel void _cuda_pagerank_update_rank_tmp_push_single(
          pr_float     *restrict dst,
    const pr_float     *restrict tmp,
    const graph_size_t *restrict rid,
    const graph_size_t *restrict cid,
    const graph_size_t vcount)
{
    const graph_size_t node = blockIdx.x * blockDim.x + threadIdx.x;
    if (node < vcount)
    {
        const graph_size_t efr = rid[node];
        const graph_size_t eto = rid[node + 1];

        for (graph_size_t edge = efr; edge < eto; edge++)
            atomicAdd(&dst[cid[edge]], tmp[node]);
    }
}

void cuda_pagerank_update_rank_tmp_push(const cudaStream_t stream, pr_float *restrict dst, const pr_float *restrict tmp, const graph_size_t *restrict rid, const graph_size_t *restrict cid, const graph_size_t vcount)
{
    launch_kernel(_cuda_pagerank_update_rank_tmp_push_single, dst, tmp, rid, cid, vcount);
}

// Based on Toast++ (https://github.com/toastpp/toastpp)
template <graph_size_t VECTORS_PER_BLOCK, graph_size_t THREADS_PER_VECTOR>
kernel void _cuda_pagerank_update_rank_tmp_pull_warp(
          pr_float     *restrict dst,
    const pr_float     *restrict tmp,
    const graph_size_t *restrict rid,
    const graph_size_t *restrict cid,
    const graph_size_t vcount)
{
    shared volatile pr_float     scr[VECTORS_PER_BLOCK * THREADS_PER_VECTOR + THREADS_PER_VECTOR / 2];
    shared volatile graph_size_t space[VECTORS_PER_BLOCK][2];

    const graph_size_t THREADS_PER_BLOCK = VECTORS_PER_BLOCK * THREADS_PER_VECTOR;

    const graph_size_t thread_id   = THREADS_PER_BLOCK * blockIdx.x + threadIdx.x;    // global thread index
    const graph_size_t thread_lane = threadIdx.x & (THREADS_PER_VECTOR - 1);          // thread index within the vector
    const graph_size_t vector_id   = thread_id   /  THREADS_PER_VECTOR;               // global vector index
    const graph_size_t vector_lane = threadIdx.x /  THREADS_PER_VECTOR;               // vector index within the block
    const graph_size_t num_vectors = VECTORS_PER_BLOCK * gridDim.x;                   // total number of active vectors

    for (graph_size_t node = vector_id; node < vcount; node += num_vectors)
    {
        if(thread_lane < 2)
            space[vector_lane][thread_lane] = rid[node + thread_lane];

        const graph_size_t efr = space[vector_lane][0];
        const graph_size_t eto = space[vector_lane][1];

        // initialize local sum
        pr_float sum = 0;

        if (THREADS_PER_VECTOR == 32 && eto - efr > 32)
        {
            // ensure aligned memory access to Aj and Ax
            graph_size_t i = efr - (efr & (THREADS_PER_VECTOR - 1)) + thread_lane;

            // accumulate local sums
            if(i >= efr && i < eto)
                sum += tmp[cid[i]];

            // accumulate local sums
            for(i += THREADS_PER_VECTOR; i < eto; i += THREADS_PER_VECTOR)
                sum += tmp[cid[i]];
        }
        else
        {
            // accumulate local sums
            for(graph_size_t i = efr + thread_lane; i < eto; i += THREADS_PER_VECTOR)
                sum += tmp[cid[i]];
        }

        // store local sum in shared memory
        scr[threadIdx.x] = sum;

        // reduce local sums to row sum
        if (THREADS_PER_VECTOR > 16) scr[threadIdx.x] = sum = sum + scr[threadIdx.x + 16];
        if (THREADS_PER_VECTOR >  8) scr[threadIdx.x] = sum = sum + scr[threadIdx.x +  8];
        if (THREADS_PER_VECTOR >  4) scr[threadIdx.x] = sum = sum + scr[threadIdx.x +  4];
        if (THREADS_PER_VECTOR >  2) scr[threadIdx.x] = sum = sum + scr[threadIdx.x +  2];
        if (THREADS_PER_VECTOR >  1) scr[threadIdx.x] = sum = sum + scr[threadIdx.x +  1];

        // first thread writes the result
        if (thread_lane == 0)
            dst[node] += scr[threadIdx.x];
    }
}

template <graph_size_t THREADS_PER_VECTOR>
static inline void launch_cuda_pagerank_update_rank_tmp_pull_warp(const cudaStream_t stream, pr_float *restrict dst, const pr_float *restrict tmp, const graph_size_t *restrict rid, const graph_size_t *restrict cid, const graph_size_t vcount)
{
    const unsigned int THREADS_PER_BLOCK = 1024;
    const unsigned int VECTORS_PER_BLOCK = THREADS_PER_BLOCK / THREADS_PER_VECTOR;

    _cuda_pagerank_update_rank_tmp_pull_warp<VECTORS_PER_BLOCK, THREADS_PER_VECTOR><<<DIVIDE_BY_INC(vcount, VECTORS_PER_BLOCK), THREADS_PER_BLOCK, 0, stream>>> 
        (dst, tmp, rid, cid, vcount);
}

void cuda_pagerank_update_rank_tmp_pull_warp(const cudaStream_t stream, const graph_size_t avg, pr_float *restrict dst, const pr_float *restrict tmp, const graph_size_t *restrict rid, const graph_size_t *restrict cid, const graph_size_t vcount)
{
    if (avg <= 2) {
        launch_cuda_pagerank_update_rank_tmp_pull_warp<2>
            (stream, dst, tmp, rid, cid, vcount);
    } else if (avg <= 4) {
        launch_cuda_pagerank_update_rank_tmp_pull_warp<4>
            (stream, dst, tmp, rid, cid, vcount);
    } else if (avg <= 8) {
        launch_cuda_pagerank_update_rank_tmp_pull_warp<8>
            (stream, dst, tmp, rid, cid, vcount);
    } else if (avg <= 16) {
        launch_cuda_pagerank_update_rank_tmp_pull_warp<16>
            (stream, dst, tmp, rid, cid, vcount);
    } else {
        launch_cuda_pagerank_update_rank_tmp_pull_warp<32>
            (stream, dst, tmp, rid, cid, vcount);
    }

    CUDA_ASSERT(cudaPeekAtLastError());
}

// Based on Toast++ (https://github.com/toastpp/toastpp)
template <graph_size_t VECTORS_PER_BLOCK, graph_size_t THREADS_PER_VECTOR>
kernel void _cuda_pagerank_update_rank_tmp_push_warp(
          pr_float     *restrict dst,
    const pr_float     *restrict tmp,
    const graph_size_t *restrict rid,
    const graph_size_t *restrict cid,
    const graph_size_t vcount)
{
    shared volatile graph_size_t space[VECTORS_PER_BLOCK][2];

    const graph_size_t THREADS_PER_BLOCK = VECTORS_PER_BLOCK * THREADS_PER_VECTOR;

    const graph_size_t thread_id   = THREADS_PER_BLOCK * blockIdx.x + threadIdx.x;    // global thread index
    const graph_size_t thread_lane = threadIdx.x & (THREADS_PER_VECTOR - 1);          // thread index within the vector
    const graph_size_t vector_id   = thread_id   /  THREADS_PER_VECTOR;               // global vector index
    const graph_size_t vector_lane = threadIdx.x /  THREADS_PER_VECTOR;               // vector index within the block
    const graph_size_t num_vectors = VECTORS_PER_BLOCK * gridDim.x;                   // total number of active vectors

    for(graph_size_t node = vector_id; node < vcount; node += num_vectors)
    {
        if(thread_lane < 2)
            space[vector_lane][thread_lane] = rid[node + thread_lane];

        const graph_size_t efr = space[vector_lane][0];
        const graph_size_t eto = space[vector_lane][1];

        // accumulate local sums
        for(graph_size_t i = efr + thread_lane; i < eto; i += THREADS_PER_VECTOR)
            atomicAdd(&dst[cid[i]], tmp[node]);
    }
}


template <graph_size_t THREADS_PER_VECTOR>
static inline void launch_cuda_pagerank_update_rank_tmp_push_warp(const cudaStream_t stream, pr_float *restrict dst, const pr_float *restrict tmp, const graph_size_t *restrict rid, const graph_size_t *restrict cid, const graph_size_t vcount)
{
    const unsigned int THREADS_PER_BLOCK = 1024;
    const unsigned int VECTORS_PER_BLOCK = THREADS_PER_BLOCK / THREADS_PER_VECTOR;

    _cuda_pagerank_update_rank_tmp_push_warp<VECTORS_PER_BLOCK, THREADS_PER_VECTOR><<<DIVIDE_BY_INC(vcount, VECTORS_PER_BLOCK), THREADS_PER_BLOCK, 0, stream>>> 
        (dst, tmp, rid, cid, vcount);
}

void cuda_pagerank_update_rank_tmp_push_warp(const cudaStream_t stream, const graph_size_t avg, pr_float *restrict dst, const pr_float *restrict tmp, const graph_size_t *restrict rid, const graph_size_t *restrict cid, const graph_size_t vcount)
{
    if (avg <= 2) {
        launch_cuda_pagerank_update_rank_tmp_push_warp<2>
            (stream, dst, tmp, rid, cid, vcount);
    } else if (avg <= 4) {
        launch_cuda_pagerank_update_rank_tmp_push_warp<4>
            (stream, dst, tmp, rid, cid, vcount);
    } else if (avg <= 8) {
        launch_cuda_pagerank_update_rank_tmp_push_warp<8>
            (stream, dst, tmp, rid, cid, vcount);
    } else if (avg <= 16) {
        launch_cuda_pagerank_update_rank_tmp_push_warp<16>
            (stream, dst, tmp, rid, cid, vcount);
    } else {
        launch_cuda_pagerank_update_rank_tmp_push_warp<32>
            (stream, dst, tmp, rid, cid, vcount);
    }

    CUDA_ASSERT(cudaPeekAtLastError());
}

#define __shfl(...)      __shfl_sync(0xFFFFFFFF, __VA_ARGS__)
#define __shfl_down(...) __shfl_down_sync(0xFFFFFFFF, __VA_ARGS__)

// Based on LightSpMV (http://lightspmv.sourceforge.net)
template<graph_size_t THREADS_PER_VECTOR, graph_size_t MAX_NUM_VECTORS_PER_BLOCK>
kernel void _cuda_pagerank_update_rank_tmp_pull_dyn(
          graph_size_t* restrict idx,
          pr_float     *restrict dst,
    const pr_float     *restrict tmp,
    const graph_size_t* restrict rid,
    const graph_size_t* restrict cid,
    const graph_size_t vcount)
{
    graph_size_t row;
    const graph_size_t laneId       = threadIdx.x % THREADS_PER_VECTOR; // lane index in the vector
    const graph_size_t vectorId     = threadIdx.x / THREADS_PER_VECTOR; // vector index in the thread block
    const graph_size_t warpLaneId   = threadIdx.x & 31;                 // lane index in the warp
    const graph_size_t warpVectorId = warpLaneId / THREADS_PER_VECTOR;  // vector index in the warp

    shared volatile graph_size_t space[MAX_NUM_VECTORS_PER_BLOCK][2];

    //get the row index
    if (warpLaneId == 0)
        row = atomicAdd(idx, 32 / THREADS_PER_VECTOR);

    //broadcast the value to other threads in the same warp and compute the row index of each vector
    row = __shfl((int)row, 0) + warpVectorId;

    //check the row range
    while (row < vcount)
    {
        //use two threads to fetch the row offset
        if (laneId < 2)
            space[vectorId][laneId] = rid[row + laneId];

        graph_size_t efr = space[vectorId][0];
        graph_size_t eto = space[vectorId][1];

        graph_size_t i;
        pr_float sum = 0;

        if (THREADS_PER_VECTOR == 32)
        {
            //ensure aligned memory access
            i = efr - (efr & (THREADS_PER_VECTOR - 1)) + laneId;

            //process the unaligned part
            if (i >= efr && i < eto)
                sum += tmp[cid[i]];

            //process the aligned part
            for (i += THREADS_PER_VECTOR; i < eto; i += THREADS_PER_VECTOR)
                sum += tmp[cid[i]];
        }
        else
        {
            //regardless of the global memory access alignment
            for (i = efr + laneId; i < eto; i += THREADS_PER_VECTOR)
                sum += tmp[cid[i]];
        }

        //intra-vector reduction
        #pragma unroll
        for (i = THREADS_PER_VECTOR >> 1; i > 0; i >>= 1)
            sum += __shfl_down(sum, i, THREADS_PER_VECTOR);

        //save the results and get a new row
        if (laneId == 0)
            dst[row] += sum;

        //get a new row index
        if (warpLaneId == 0)
            row = atomicAdd(idx, 32 / THREADS_PER_VECTOR);

        //broadcast the row index to the other threads in the same warp and compute the row index of each vetor
        row = __shfl((int)row, 0) + warpVectorId;
    }
}

void cuda_pagerank_update_rank_tmp_pull_dyn(const cudaStream_t stream, const graph_size_t avg, graph_size_t *restrict scr, pr_float *restrict dst, const pr_float *restrict tmp, const graph_size_t *restrict rid, const graph_size_t *restrict cid, const graph_size_t vcount)
{
    static const graph_size_t MAX_NUM_THREADS_PER_BLOCK = 1024;

    CUDA_ASSERT(cudaMemsetAsync(scr, 0, sizeof(graph_size_t), stream));

    if (avg <= 2)
        launch_kernel_max((_cuda_pagerank_update_rank_tmp_pull_dyn<2, MAX_NUM_THREADS_PER_BLOCK / 2>), scr, dst, tmp, rid, cid, vcount)
    else if (avg <= 4)
        launch_kernel_max((_cuda_pagerank_update_rank_tmp_pull_dyn<4, MAX_NUM_THREADS_PER_BLOCK / 4>), scr, dst, tmp, rid, cid, vcount)
    else if (avg <= 64)
        launch_kernel_max((_cuda_pagerank_update_rank_tmp_pull_dyn<8, MAX_NUM_THREADS_PER_BLOCK / 8>), scr, dst, tmp, rid, cid, vcount)
    else
        launch_kernel_max((_cuda_pagerank_update_rank_tmp_pull_dyn<32, MAX_NUM_THREADS_PER_BLOCK / 32>), scr, dst, tmp, rid, cid, vcount)
}

// Based on LightSpMV (http://lightspmv.sourceforge.net)
template<graph_size_t THREADS_PER_VECTOR, graph_size_t MAX_NUM_VECTORS_PER_BLOCK>
kernel void _cuda_pagerank_update_rank_tmp_push_dyn(
          graph_size_t* restrict idx,
          pr_float     *restrict dst,
    const pr_float     *restrict tmp,
    const graph_size_t* restrict rid,
    const graph_size_t* restrict cid,
    const graph_size_t vcount)
{
    graph_size_t row;
    const graph_size_t laneId       = threadIdx.x % THREADS_PER_VECTOR; // lane index in the vector
    const graph_size_t vectorId     = threadIdx.x / THREADS_PER_VECTOR; // vector index in the thread block
    const graph_size_t warpLaneId   = threadIdx.x & 31;                 // lane index in the warp
    const graph_size_t warpVectorId = warpLaneId / THREADS_PER_VECTOR;  // vector index in the warp

    shared volatile graph_size_t space[MAX_NUM_VECTORS_PER_BLOCK][2];

    //get the row index
    if (warpLaneId == 0)
        row = atomicAdd(idx, 32 / THREADS_PER_VECTOR);

    //broadcast the value to other threads in the same warp and compute the row index of each vector
    row = __shfl((int)row, 0) + warpVectorId;

    //check the row range
    while (row < vcount)
    {
        //use two threads to fetch the row offset
        if (laneId < 2)
            space[vectorId][laneId] = rid[row + laneId];

        graph_size_t efr = space[vectorId][0];
        graph_size_t eto = space[vectorId][1];

        graph_size_t i;

        if (THREADS_PER_VECTOR == 32)
        {
            //ensure aligned memory access
            i = efr - (efr & (THREADS_PER_VECTOR - 1)) + laneId;

            //process the unaligned part
            if (i >= efr && i < eto)
                atomicAdd(&dst[cid[i]], tmp[row]);

            //process the aligned part
            for (i += THREADS_PER_VECTOR; i < eto; i += THREADS_PER_VECTOR)
                atomicAdd(&dst[cid[i]], tmp[row]);
        }
        else
        {
            //regardless of the global memory access alignment
            for (i = efr + laneId; i < eto; i += THREADS_PER_VECTOR)
                atomicAdd(&dst[cid[i]], tmp[row]);
        }

        //get a new row index
        if (warpLaneId == 0)
            row = atomicAdd(idx, 32 / THREADS_PER_VECTOR);

        //broadcast the row index to the other threads in the same warp and compute the row index of each vetor
        row = __shfl((int)row, 0) + warpVectorId;
    }
}

void cuda_pagerank_update_rank_tmp_push_dyn(const cudaStream_t stream, const graph_size_t avg, graph_size_t *restrict scr, pr_float *restrict dst, const pr_float *restrict tmp, const graph_size_t *restrict rid, const graph_size_t *restrict cid, const graph_size_t vcount)
{
    static const graph_size_t MAX_NUM_THREADS_PER_BLOCK = 1024;

    CUDA_ASSERT(cudaMemsetAsync(scr, 0, sizeof(graph_size_t), stream));

    if (avg <= 2)
        launch_kernel_max((_cuda_pagerank_update_rank_tmp_push_dyn<2, MAX_NUM_THREADS_PER_BLOCK / 2>), scr, dst, tmp, rid, cid, vcount)
    else if (avg <= 4)
        launch_kernel_max((_cuda_pagerank_update_rank_tmp_push_dyn<4, MAX_NUM_THREADS_PER_BLOCK / 4>), scr, dst, tmp, rid, cid, vcount)
    else if (avg <= 64)
        launch_kernel_max((_cuda_pagerank_update_rank_tmp_push_dyn<8, MAX_NUM_THREADS_PER_BLOCK / 8>), scr, dst, tmp, rid, cid, vcount)
    else
        launch_kernel_max((_cuda_pagerank_update_rank_tmp_push_dyn<32, MAX_NUM_THREADS_PER_BLOCK / 32>), scr, dst, tmp, rid, cid, vcount)
}

kernel void _cuda_pagerank_update_dest_single(
          pr_float *restrict dst,
    const pr_float *restrict brp,
    const pr_float damping,
    const graph_size_t vcount)
{
    const graph_size_t node = blockIdx.x * blockDim.x + threadIdx.x;

    if (node < vcount)
        dst[node] = (damping*dst[node]) + *brp;
}

void cuda_pagerank_update_dest(const cudaStream_t stream, pr_float *restrict dst, const pr_float *restrict brp, const pr_float damping, const graph_size_t vcount)
{
    launch_kernel(_cuda_pagerank_update_dest_single, dst, brp, damping, vcount);
}

kernel void _cuda_pagerank_update_dest_raw_single(
          pr_float *restrict dst,
    const pr_float brp,
    const pr_float damping,
    const graph_size_t vcount)
{
    const graph_size_t node = blockIdx.x * blockDim.x + threadIdx.x;

    if (node < vcount)
        dst[node] = (damping*dst[node]) + brp;
}

void cuda_pagerank_update_dest_raw(const cudaStream_t stream, pr_float *restrict dst, const pr_float brp, const pr_float damping, const graph_size_t vcount)
{
    launch_kernel(_cuda_pagerank_update_dest_raw_single, dst, brp, damping, vcount);
}

kernel void _cuda_pagerank_calc_dest_single(
          pr_float *restrict dst,
    const pr_float *restrict tmp,
    const pr_float *restrict brp,
    const pr_float damping,
    const graph_size_t vcount)
{
    const graph_size_t node = blockIdx.x * blockDim.x + threadIdx.x;

    if (node < vcount)
        dst[node] = (damping*tmp[node]) + *brp;
}

void cuda_pagerank_calc_dest(const cudaStream_t stream, pr_float *restrict dst, const pr_float *restrict tmp, const pr_float *restrict brp, const pr_float damping, const graph_size_t vcount)
{
    launch_kernel(_cuda_pagerank_calc_dest_single, dst, tmp, brp, damping, vcount);
}

kernel void _cuda_pagerank_calc_dest_raw_single(
          pr_float *restrict dst,
    const pr_float *restrict tmp,
    const pr_float brp,
    const pr_float damping,
    const graph_size_t vcount)
{
    const graph_size_t node = blockIdx.x * blockDim.x + threadIdx.x;

    if (node < vcount)
        dst[node] = (damping*tmp[node]) + brp;
}

void cuda_pagerank_calc_dest_raw(const cudaStream_t stream, pr_float *restrict dst, const pr_float *restrict tmp, const pr_float brp, const pr_float damping, const graph_size_t vcount)
{
    launch_kernel(_cuda_pagerank_calc_dest_raw_single, dst, tmp, brp, damping, vcount);
}

kernel void _cuda_pagerank_calc_diff_single(
          pr_float *restrict dif,
    const pr_float *restrict src,
    const pr_float *restrict dst,
    const graph_size_t vcount)
{
    const graph_size_t node = blockIdx.x * blockDim.x + threadIdx.x;
    if (node < vcount)
        dif[node] = fabs(src[node] - dst[node]);
}

void cuda_pagerank_calc_diff(const cudaStream_t stream, pr_float *restrict dif, const pr_float *restrict src, const pr_float *restrict dst, const graph_size_t vcount)
{
    launch_kernel(_cuda_pagerank_calc_diff_single, dif, src, dst, vcount);
}

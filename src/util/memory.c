// Author:  Niels A.D.
// Project: HGB (https://github.com/nielsAD/hgb)
// License: Mozilla Public License, v2.0

#include "util/memory.h"
#include "util/math.h"
#include "util/string.h"

#include "util/cuda.h"
#include "util/starpu.h"

#include "memory_ocl.c.inc"
#include "memory_mmap.c.inc"

size_t                MEMORY_ALIGNMENT_DEFAULT      = 128;
memory_manager_enum_t MEMORY_PINNED_MANAGER_DEFAULT = E_MM_DEFAULT;

size_t memory_get_default_alignment()
{
    return MEMORY_ALIGNMENT_DEFAULT;
}

void memory_set_default_alignment(const size_t alignment)
{
    if (alignment > 0)
        MEMORY_ALIGNMENT_DEFAULT = alignment;
}

memory_manager_enum_t memory_get_default_pinned_manager(void)
{
    return MEMORY_PINNED_MANAGER_DEFAULT;
}

void memory_set_default_pinned_manager(const memory_manager_enum_t manager)
{
    if (manager < E_MM_MAX)
        MEMORY_PINNED_MANAGER_DEFAULT = manager;
}

void *memory_aligned_alloc(const size_t size, const size_t alignment)
{
    assert(alignment > 0);

    void *ptr = NULL;
    UNUSED int err = posix_memalign(&ptr, alignment, ROUND_TO_MULT(size, alignment));
    assert(err == 0);

    assert(ptr != NULL || size <= 0);
    return ptr;
}

void *memory_aligned_calloc(const size_t size, const size_t alignment)
{
    void *ptr = memory_aligned_alloc(size, alignment);
    memset(ptr, 0, size);

    return ptr;
}

void *memory_aligned_realloc(void *ptr, const size_t size, const size_t alignment)
{
    assert(alignment > 0);

    if (ptr == NULL)
        ptr = memory_aligned_alloc(size, alignment);
    else if (size == 0)
    {
        memory_aligned_free(ptr);
        ptr = NULL;
    }
    else
    {
        ptr = realloc(ptr, ROUND_TO_MULT(size, alignment));
        assert(ptr != NULL);

        if ((uintptr_t)ptr & (alignment - 1))
        {
            void *new_ptr = memory_aligned_alloc(size, alignment);

            memcpy(new_ptr, ptr, size);

            memory_aligned_free(ptr);
            ptr = new_ptr;
        }
    }

    return ptr;
}

void memory_aligned_free(void *ptr)
{
    free(ptr);
}

void *memory_alloc(const size_t size)
{
    return memory_aligned_alloc(size, MEMORY_ALIGNMENT_DEFAULT);
}

void *memory_calloc(const size_t size)
{
    return memory_aligned_calloc(size, MEMORY_ALIGNMENT_DEFAULT);
}

void *memory_realloc(void *ptr, const size_t size)
{
    return memory_aligned_realloc(ptr, size, MEMORY_ALIGNMENT_DEFAULT);
}

void memory_free(void *ptr)
{
    memory_aligned_free(ptr);
}

void *memory_pinned_alloc_mgr(const memory_manager_enum_t manager, const size_t size)
{
    if (size <= 0)
        return NULL;

    switch(manager)
    {
        case E_MM_DEFAULT:
            return memory_alloc(size);

        case E_MM_MMAP:
            return memory_mapped_alloc(size, -1);

        case E_MM_OPENCL:
            return ocl_host_alloc(size);

        case E_MM_CUDA:
        {
            void *res;
            CUDA_ASSERT(cudaHostAlloc(&res, size, cudaHostAllocPortable));
            assert(res != NULL);
            return res;
        }

        case E_MM_STARPU:
        {
            void *res;
            STARPU_CHECK_RETURN_VALUE(starpu_malloc(&res, size), "starpu_malloc");
            assert(res != NULL);
            return res;
        }

        default:
            assert(false && "unknown memory manager");
            return NULL;
    }
}

void *memory_pinned_calloc_mgr(const memory_manager_enum_t manager, const size_t size)
{
    void *ptr = memory_pinned_alloc_mgr(manager, size);
    memset(ptr, 0, size);

    return ptr;
}

void *memory_pinned_realloc_mgr(const memory_manager_enum_t manager, void *ptr, const size_t old_size, const size_t new_size)
{
    switch(manager)
    {
        case E_MM_DEFAULT:
            return memory_realloc(ptr, new_size);

        case E_MM_MMAP:
            return memory_mapped_realloc(ptr, new_size);

        default:
        {
            void *new_ptr = memory_pinned_alloc_mgr(manager, new_size);

            if (ptr != NULL && old_size > 0 && new_size > 0)
                memcpy(new_ptr, ptr, MIN(old_size, new_size));

            memory_pinned_free_mgr(manager, ptr);

            return new_ptr;
        }
    }
}

void *memory_pinned_realloc_managers(const memory_manager_enum_t manager_from, const memory_manager_enum_t manager_to, void *ptr, const size_t size)
{
    if (ptr == NULL || size == 0 || manager_from == manager_to)
        return ptr;

    void *new_ptr = memory_pinned_alloc_mgr(manager_to, size);
    memcpy(new_ptr, ptr, size);
    memory_pinned_free_mgr(manager_from, ptr);

    return new_ptr;
}

void memory_pinned_free_mgr(const memory_manager_enum_t manager, void *ptr)
{
    if (ptr != NULL)
        switch(manager)
        {
            case E_MM_MMAP:   memory_mapped_free(ptr);                                    break;
            case E_MM_OPENCL: ocl_free(ptr);                                              break;
            case E_MM_CUDA:   CUDA_ASSERT(cudaFreeHost(ptr));                             break;
            case E_MM_STARPU: STARPU_CHECK_RETURN_VALUE(starpu_free(ptr), "starpu_free"); break;
            default:          memory_free(ptr);                                           break;
        }
}

void *memory_pinned_alloc(const size_t size)
{
    return memory_pinned_alloc_mgr(MEMORY_PINNED_MANAGER_DEFAULT, size);
}

void *memory_pinned_calloc(const size_t size)
{
    return memory_pinned_calloc_mgr(MEMORY_PINNED_MANAGER_DEFAULT, size);
}

void *memory_pinned_realloc(void *ptr, const size_t old_size, const size_t new_size)
{
    return memory_pinned_realloc_mgr(MEMORY_PINNED_MANAGER_DEFAULT, ptr, old_size, new_size);
}

void memory_pinned_free(void *ptr)
{
    memory_pinned_free_mgr(MEMORY_PINNED_MANAGER_DEFAULT, ptr);
}

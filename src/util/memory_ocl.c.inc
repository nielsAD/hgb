// Author:  Niels A.D.
// Project: HGB (https://github.com/nielsAD/hgb)
// License: Mozilla Public License, v2.0

#include "util/opencl.h"

static cl_context       ocl_context = NULL;
static cl_command_queue ocl_queue   = NULL;

void memory_set_opencl_context(const cl_context context, const cl_command_queue queue)
{
    ocl_context = context;
    ocl_queue   = queue;
}

struct _ocl_memory_entry;
struct _ocl_memory_entry {
    volatile struct _ocl_memory_entry *next;
    void *ptr;
    cl_mem mem;
};

static volatile struct _ocl_memory_entry *_ocl_memory_entries = NULL;

static inline volatile struct _ocl_memory_entry *ocl_memory_entry_find(const void *ptr)
{
    volatile struct _ocl_memory_entry *head = _ocl_memory_entries;
    while (head != NULL && head->ptr != ptr)
        head = head->next;
    return head;
}

static inline volatile struct _ocl_memory_entry *ocl_memory_entry_insert_head(const cl_mem mem, void *ptr)
{
    assert(ptr != NULL);

    volatile struct _ocl_memory_entry *entry = (volatile struct _ocl_memory_entry*) malloc(sizeof(struct _ocl_memory_entry));
    entry->mem = mem;
    entry->ptr = ptr;

    volatile void* old;
    do {
        old = entry->next = _ocl_memory_entries;
    } while (!__sync_bool_compare_and_swap(&_ocl_memory_entries, old, entry));

    return entry;
}

static inline volatile struct _ocl_memory_entry *ocl_memory_entry_insert(const cl_mem mem, void *ptr)
{
    assert(ptr != NULL);

    volatile struct _ocl_memory_entry *entry;
    while ((entry = ocl_memory_entry_find(NULL)) != NULL)
        if (__sync_bool_compare_and_swap(&entry->ptr, NULL, ptr))
        {
            entry->mem = mem;
            return (struct _ocl_memory_entry*) entry;
        }

    return ocl_memory_entry_insert_head(mem, ptr);
}

static inline cl_mem ocl_memory_entry_remove(void *ptr)
{
    volatile struct _ocl_memory_entry *entry = ocl_memory_entry_find(ptr);
    if (entry != NULL)
    {
        cl_mem mem = entry->mem;
        if (__sync_bool_compare_and_swap(&entry->ptr, ptr, NULL))
            return mem;
    }
    return NULL;
}

static inline void *ocl_host_alloc(const size_t size)
{
    assert(ocl_context != NULL);
    assert(ocl_queue   != NULL);

    if (size <= 0)
        return NULL;

    cl_int err;
    cl_mem mem = clCreateBuffer(ocl_context, CL_MEM_READ_ONLY | CL_MEM_ALLOC_HOST_PTR, size, NULL, &err); OPENCL_ASSERT(err);
    void  *ptr = clEnqueueMapBuffer(ocl_queue, mem, CL_TRUE, CL_MAP_WRITE, 0, size, 0, NULL, NULL, &err); OPENCL_ASSERT(err);
    assert(ptr != NULL);

    ocl_memory_entry_insert(mem, ptr);
    return ptr;
}

static inline void ocl_free(void *ptr)
{
    if (ptr == NULL)
        return;

    assert(ocl_queue != NULL);

    cl_mem mem = ocl_memory_entry_remove(ptr);
    if (mem)
    {
        OPENCL_ASSERT(clEnqueueUnmapMemObject(ocl_queue, mem, ptr, 0, NULL, NULL));
        OPENCL_ASSERT(clReleaseMemObject(mem));
    }
}

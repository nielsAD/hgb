// Author:  Niels A.D.
// Project: HGB (https://github.com/nielsAD/hgb)
// License: Mozilla Public License, v2.0

#pragma once

#include "util/ubiquitous.h"
#include "util/file.h"
#include "util/opencl.h"

#define STATIC_ARR_SIZE(ar) (sizeof ar / sizeof ar[0])

#ifdef __cplusplus
extern "C" {
#endif

typedef enum memory_manager {
    E_MM_DEFAULT,
    E_MM_MMAP,
    E_MM_OPENCL,
    E_MM_CUDA,
    E_MM_STARPU,
    E_MM_MAX
} memory_manager_enum_t;

size_t memory_get_default_alignment(void) PURE_FUN;
void   memory_set_default_alignment(const size_t alignment);

memory_manager_enum_t memory_get_default_pinned_manager(void) PURE_FUN;
                 void memory_set_default_pinned_manager(const memory_manager_enum_t manager);

void memory_set_mapped_tmpdir(char *dir);
void memory_set_opencl_context(const cl_context context, const cl_command_queue queue);

void *memory_aligned_alloc(const size_t size, const size_t alignment);
void *memory_aligned_calloc(const size_t size, const size_t alignment);
void *memory_aligned_realloc(void *ptr, const size_t size, const size_t alignment);
void  memory_aligned_free(void *ptr);

void *memory_alloc(const size_t size);
void *memory_calloc(const size_t size);
void *memory_realloc(void *ptr, const size_t size);
void  memory_free(void *ptr);

void *memory_mapped_alloc(const size_t size, int fno);
void *memory_mapped_calloc(const size_t size, int fno);
void *memory_mapped_realloc(void *ptr, const size_t size);
void  memory_mapped_free(void *ptr);

void *memory_pinned_alloc_mgr(const memory_manager_enum_t manager, const size_t size);
void *memory_pinned_calloc_mgr(const memory_manager_enum_t manager, const size_t size);
void *memory_pinned_realloc_mgr(const memory_manager_enum_t manager, void *ptr, const size_t old_size, const size_t new_size);
void *memory_pinned_realloc_managers(const memory_manager_enum_t manager_from, const memory_manager_enum_t manager_to, void *ptr, const size_t size);
void  memory_pinned_free_mgr(const memory_manager_enum_t manager, void *ptr);

void *memory_pinned_alloc(const size_t size);
void *memory_pinned_calloc(const size_t size);
void *memory_pinned_realloc(void *ptr, const size_t old_size, const size_t new_size);
void  memory_pinned_free(void *ptr);

#ifdef __cplusplus
}
#endif

#define memory_talloc1(t)       (CAST_TO_POINTER_OF(t) memory_calloc(sizeof(t)))     // Clear mem for single
#define memory_talloc2(t,n)     (CAST_TO_POINTER_OF(t) memory_alloc(sizeof(t)*(n)))
#define memory_talloc3(t,w,h)   (CAST_TO_POINTER_OF(t) memory_alloc(sizeof(t)*(w)*(h)))
#define memory_talloc4(t,w,h,d) (CAST_TO_POINTER_OF(t) memory_alloc(sizeof(t)*(w)*(h)*(d)))
#define memory_talloc(type,...) VARARG(memory_talloc,type,##__VA_ARGS__)

#define memory_retalloc2(p,n)     (CAST_TO_TYPE_OF(p) memory_realloc(p,sizeof(*p)*(n)))
#define memory_retalloc3(p,w,h)   (CAST_TO_TYPE_OF(p) memory_realloc(p,sizeof(*p)*(w)*(h)))
#define memory_retalloc4(p,w,h,d) (CAST_TO_TYPE_OF(p) memory_realloc(p,sizeof(*p)*(w)*(h)*(d)))
#define memory_retalloc(ptr,...)  VARARG(memory_retalloc,ptr,##__VA_ARGS__)

#define memory_toggle(cond_tog,ptr,...) do {if (cond_tog) ptr=memory_talloc(*ptr,__VA_ARGS__); else {memory_free(ptr); ptr=NULL;}} while(false)

#define memory_pinned_talloc1(t)       (CAST_TO_POINTER_OF(t) memory_pinned_calloc(sizeof(t)))
#define memory_pinned_talloc2(t,n)     (CAST_TO_POINTER_OF(t) memory_pinned_alloc(sizeof(t)*(n)))
#define memory_pinned_talloc3(t,w,h)   (CAST_TO_POINTER_OF(t) memory_pinned_alloc(sizeof(t)*(w)*(h)))
#define memory_pinned_talloc4(t,w,h,d) (CAST_TO_POINTER_OF(t) memory_pinned_alloc(sizeof(t)*(w)*(h)*(d)))
#define memory_pinned_talloc(type,...) VARARG(memory_pinned_talloc,type,##__VA_ARGS__)

#define memory_pinned_retalloc(p,o,n)  (CAST_TO_TYPE_OF(p) memory_pinned_realloc(p,sizeof(*p)*(o),sizeof(*p)*(n)))
#define memory_pinned_toggle(cond_tog,ptr,...) do {if (cond_tog) ptr=memory_pinned_talloc(*ptr,##__VA_ARGS__); else {memory_pinned_free(ptr); ptr=NULL;}} while(false)

#define memory_pinned_talloc_if(cond_if,type,...)  ((cond_if) ? memory_pinned_talloc(type,##__VA_ARGS__) : memory_talloc(type,##__VA_ARGS__))
#define memory_pinned_retalloc_if(cond_if,ptr,o,n) ((cond_if) ? memory_pinned_retalloc(ptr,o,n) : memory_retalloc(ptr,n))
#define memory_pinned_toggle_if(cond_if,cond_tog,ptr,...) do {if (cond_if) memory_pinned_toggle(cond_tog,ptr,##__VA_ARGS__); else memory_toggle(cond_tog,ptr,##__VA_ARGS__); } while(false)
#define memory_pinned_free_if(cond_if,ptr)                do {if (cond_if) memory_pinned_free((void*)ptr); else memory_free((void*)ptr);} while(false)

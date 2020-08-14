// Author:  Niels A.D.
// Project: HGB (https://github.com/nielsAD/hgb)
// License: Mozilla Public License, v2.0

#include "util/file.h"
#include "util/string.h"
#include <sys/mman.h>

static char *mmap_tmp_dir = NULL;

void memory_set_mapped_tmpdir(char *dir)
{
    free(mmap_tmp_dir);
    mmap_tmp_dir = (dir == NULL) ? NULL : strdup(dir);
}

struct _mmap_entry;
struct _mmap_entry {
    volatile struct _mmap_entry *next;
    void *ptr;
    int fno;
    size_t size;
};

static volatile struct _mmap_entry *_mmap_entries = NULL;

static inline volatile struct _mmap_entry *mmap_entry_find(const void *ptr)
{
    volatile struct _mmap_entry *head = _mmap_entries;
    while (head != NULL && head->ptr != ptr)
        head = head->next;
    return head;
}

static inline volatile struct _mmap_entry *mmap_entry_insert_head(const int fno, void *ptr)
{
    assert(ptr != NULL);

    volatile struct _mmap_entry *entry = (volatile struct _mmap_entry*) malloc(sizeof(struct _mmap_entry));
    entry->fno = fno;
    entry->ptr = ptr;

    volatile void* old;
    do {
        old = entry->next = _mmap_entries;
    } while (!__sync_bool_compare_and_swap(&_mmap_entries, old, entry));

    return entry;
}

static inline volatile struct _mmap_entry *mmap_entry_insert(const int fno, void *ptr)
{
    assert(ptr != NULL);

    volatile struct _mmap_entry *entry;
    while ((entry = mmap_entry_find(NULL)) != NULL)
        if (__sync_bool_compare_and_swap(&entry->ptr, NULL, ptr))
        {
            entry->fno = fno;
            return (struct _mmap_entry*) entry;
        }

    return mmap_entry_insert_head(fno, ptr);
}

static inline int mmap_entry_remove(void *ptr)
{
    volatile struct _mmap_entry *entry = mmap_entry_find(ptr);
    if (entry != NULL)
    {
        const int fno = entry->fno;
        if (__sync_bool_compare_and_swap(&entry->ptr, ptr, NULL))
            return fno;
    }
    return -1;
}

void *memory_mapped_alloc(const size_t size, int fno)
{
    assert(size > 0);

    if (fno < 0)
    {
        char *path = strdup(mmap_tmp_dir == NULL ? ".tmp_mmap_XXXXXX" : mmap_tmp_dir);
        fno = mkstemp(path);
        assert(fno != -1 && "could not create temporary mmap file");

        unlink(path);
        free(path);
    }

    UNUSED bool res = file_allocate(fno, size) && lseek(fno, size, SEEK_SET) != -1;
    assert(res && "could not allocate size for mmap file");

    lseek(fno, size, SEEK_SET);

    void *ptr = mmap(NULL, size, PROT_READ | PROT_WRITE, MAP_SHARED, fno, 0);
    assert(ptr != MAP_FAILED);

    mmap_entry_insert(fno, ptr);
    return ptr;
}

void *memory_mapped_calloc(const size_t size, int fno)
{
    void *ptr = memory_mapped_alloc(size, fno);
    memset(ptr, 0, size);

    return ptr;
}

void memory_mapped_free(void *ptr)
{
    if (ptr == NULL)
        return;

    const int fno = mmap_entry_remove(ptr);
    if (fno >= 0)
    {
        munmap(ptr, lseek(fno, 0, SEEK_CUR));
        close(fno);
    }
}

void *memory_mapped_realloc(void *ptr, const size_t size)
{
    if (size < 1)
    {
        memory_mapped_free(ptr);
        return NULL;
    }

    const int fno = mmap_entry_remove(ptr);
    if (fno < 0)
        return memory_mapped_alloc(size, -1);

    const size_t oldsize = (size_t)lseek(fno, 0, SEEK_CUR);

    bool res;
    if (oldsize == size)
        return ptr;
    else if (oldsize < size)
        res = file_allocate(fno, size);
    else
        res = ftruncate(fno, size) == 0;

    res |= lseek(fno, size, SEEK_SET) != -1;
    assert(res && "could not allocate size for mmap file");

    void *new_ptr = mremap(ptr, oldsize, size, MREMAP_MAYMOVE);
    assert(new_ptr != MAP_FAILED);

    mmap_entry_insert(fno, new_ptr);
    return new_ptr;
}

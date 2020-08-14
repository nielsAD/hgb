// Author:  Niels A.D.
// Project: HGB (https://github.com/nielsAD/hgb)
// License: Mozilla Public License, v2.0

#include "util/mpi.h"
#include <stdio.h>
#include <time.h>

void mpi_display_error(const char *const func, const char *const file, int line, int error_code)
{
    int  error_class;
    char error_string[BUFSIZ];

    MPI_Error_class(error_code, &error_class);
    MPI_Error_string(error_class, error_string, NULL);
    fprintf(stderr, "oops in %s (%s:%d)... <%s> (class %d)", func, file, line, error_string, error_class);

    MPI_Error_string(error_code,  error_string, NULL);
    fprintf(stderr, " :: <%s> (id %d) \n", error_string, error_code);
}

void mpi_report_error(const char *const func, const char *const file, int line, int error_code)
{
    mpi_display_error(func, file, line, error_code);
    assert(false && "mpi_report_error");
}

int mpi_world_rank = -1;
int mpi_world_size = -1;

void mpi_initialize(int *argc, char **argv[])
{
    assert(mpi_world_rank == -1);

    int prov = MPI_THREAD_SINGLE;
    MPI_ASSERT(MPI_Init_thread(argc, argv, MPI_THREAD_FUNNELED, &prov));

    MPI_Comm_set_errhandler(MPI_COMM_WORLD, MPI_ERRORS_ARE_FATAL);

    MPI_Comm_size(MPI_COMM_WORLD, &mpi_world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_world_rank);

    // Synchronize random seed
    unsigned int seed = (unsigned int) time(NULL);
    MPI_Bcast(&seed, 1, MPI_UNSIGNED, 0, MPI_COMM_WORLD);
    srand(seed);
}

void mpi_finalize()
{
    assert(mpi_world_rank != -1);
    MPI_ASSERT(MPI_Finalize());

    mpi_world_rank = -1;
    mpi_world_size = -1;
}

int mpi_get_rank(void)
{
    assert(mpi_world_rank != -1);
    return mpi_world_rank;
}

int mpi_get_size(void)
{
    assert(mpi_world_size != -1);
    return mpi_world_size;
}

MPI_Comm mpi_get_strided_comm(MPI_Comm comm, const int offset, const int stride, const int last)
{
    MPI_Group old_group = MPI_GROUP_NULL;
    MPI_Comm_group(comm, &old_group);

    int range[] = {offset, last, stride};
    MPI_Group new_group = MPI_GROUP_NULL;
    MPI_Group_range_incl(old_group, 1, &range, &new_group);

    MPI_Comm new_comm = MPI_COMM_NULL;
    MPI_Comm_create(comm, new_group, &new_comm);

    MPI_Group_free(&old_group);
    MPI_Group_free(&new_group);

    return new_comm;
}

MPI_Comm mpi_get_sub_comm(MPI_Comm comm, const int size)
{
    return mpi_get_strided_comm(comm, 0, 1, size - 1);
}

void mpi_get_rowcol_comm(MPI_Comm comm, const int width, MPI_Comm *row, MPI_Comm *col)
{
    int rank = -1;
    MPI_Comm_rank(comm, &rank);

    const int rid = rank / width;
    const int cid = rank % width;
    MPI_Comm_split(comm, rid, cid, row);
    MPI_Comm_split(comm, cid, rid, col);
}

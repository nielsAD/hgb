// Author:  Niels A.D.
// Project: HGB (https://github.com/nielsAD/hgb)
// License: Mozilla Public License, v2.0

#pragma once

#include "util/ubiquitous.h"
#include <mpi.h>

#define MPI_DISPLAY_ERROR(ERR) { if (UNLIKELY((ERR) != MPI_SUCCESS)) mpi_display_error(__FUNCTION__, __FILE__, __LINE__, ERR); }
#define MPI_REPORT_ERROR(ERR)  { if (UNLIKELY((ERR) != MPI_SUCCESS)) mpi_report_error(__FUNCTION__, __FILE__, __LINE__, ERR); }
#define MPI_CHECK_ERROR MPI_REPORT_ERROR

#define MPI_SEND_RECV(send, recv) (((void*)(send)) == ((void*)(recv)) ? MPI_IN_PLACE : (void*)(send)), (recv)

#ifdef NDEBUG
	#define MPI_ASSERT(x) { if(x){} }
#else
	#define MPI_ASSERT MPI_REPORT_ERROR
#endif

#ifdef __cplusplus
extern "C" {
#endif

void mpi_display_error(const char *const func, const char *const file, int line, int error_code);
void mpi_report_error(const char *const func, const char *const file, int line, int error_code);

void mpi_initialize(int *argc, char **argv[]);
void mpi_finalize(void);

int mpi_get_rank(void) PURE_FUN;
int mpi_get_size(void) PURE_FUN;

MPI_Comm mpi_get_strided_comm(MPI_Comm comm, const int offset, const int stride, const int last);
MPI_Comm mpi_get_sub_comm(MPI_Comm comm, const int size);
void mpi_get_rowcol_comm(MPI_Comm comm, const int width, MPI_Comm *row, MPI_Comm *col);

#ifdef __cplusplus
}
#endif

static inline int mpi_get_root(void) {
    return 0;
}

static inline bool mpi_is_root(void)
{
    return mpi_get_rank() == mpi_get_root();
}

static inline MPI_Datatype _mpi_get_int_type(const size_t size, const bool is_signed)
{
         if (size == sizeof(char))          return (is_signed) ? MPI_CHAR      : MPI_UNSIGNED_CHAR;
    else if (size == sizeof(short int))     return (is_signed) ? MPI_SHORT     : MPI_UNSIGNED_SHORT;
    else if (size == sizeof(int))           return (is_signed) ? MPI_INT       : MPI_UNSIGNED;
    else if (size == sizeof(long int))      return (is_signed) ? MPI_LONG      : MPI_UNSIGNED_LONG;
    else if (size == sizeof(long long int)) return (is_signed) ? MPI_LONG_LONG : MPI_UNSIGNED_LONG_LONG;
    else                                    return MPI_DATATYPE_NULL;
}

static inline MPI_Datatype _mpi_get_float_type(const size_t size)
{
         if (size == sizeof(float))       return MPI_FLOAT;
    else if (size == sizeof(double))      return MPI_DOUBLE;
    else if (size == sizeof(long double)) return MPI_LONG_DOUBLE;
    else                                  return MPI_DATATYPE_NULL;
}

#define mpi_get_int_type(type)   (_mpi_get_int_type(sizeof(type), ((type)-1) <= 0))
#define mpi_get_float_type(type) (_mpi_get_float_type(sizeof(type)))

#define return_if_not_mpi_root(r) { if(!mpi_is_root()) return r; }
#define mpi_printf(s,...)         { if (mpi_is_root()) printf(s,##__VA_ARGS__); }
#define mpi_fprintf(f,s,...)      { if (mpi_is_root()) fprintf(f,s,##__VA_ARGS__); }

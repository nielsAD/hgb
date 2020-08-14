# Author:  Niels A.D.
# Project: HGB (https://github.com/nielsAD/hgb)
# License: Mozilla Public License, v2.0

# Depends on variables and rules in compile.mk

EXECUTABLE ?= $(firstword $(BIN_OUT))

MPIRUN ?= mpirun
PRUN   ?= prun
GPROF  ?= gprof
MEMCHK ?= valgrind

LOG_DIR  ?= log
PROF_OUT ?= gmon.out

PROF_EXT ?= prof
LOG_EXT  ?= log
ERR_EXT  ?= err

TIMEOUT ?= 00:15:00
PROFILE ?= "$(LOG_DIR)/$(shell hostname)_$(shell date "+%y_%m_%d_%H_%M_%S").$(PROF_EXT)"
LOGFILE ?= "$(LOG_DIR)/$(shell hostname)_$(shell date "+%y_%m_%d_%H_%M_%S").$(LOG_EXT)"
ERRFILE ?= "$(LOG_DIR)/$(shell hostname)_$(shell date "+%y_%m_%d_%H_%M_%S").$(ERR_EXT)"
LOGGLOB ?= "$(LOG_DIR)/$(shell hostname).$(LOG_EXT).bak"
ERRGLOB ?= "$(LOG_DIR)/$(shell hostname).$(ERR_EXT).bak"

MPI_NP   ?= $(shell nproc)
MPI_ARGS ?= --bind-to core --report-bindings
MPI_ENV  ?= $(shell env | grep "OPENMP")

PRUN_NP   ?= 1
PRUN_PP   ?= 1
PRUN_MPI  ?= --bind-to socket --report-bindings
PRUN_ARGS ?= -v -sge-script $(PRUN_ETC)/prun-openmpi

LOG_OUT += *.trace *.trace.csv *.rec *.data dag.dot* prof_file_* trace.html tasks.rec .activity.data.names starpu_idle_microsec.log
LOG_DIRS = $(addsuffix /,$(LOG_DIR))

_MEMCHK_ARGS  = --leak-check=full --track-origins=yes $(MEMCHK_ARG) $(MEMCHK_ARGS)
_MPI_OPTS     = $(addprefix -x , $(MPI_ENV)) $(MPI_ARG) $(MPI_ARGS)
_MPI_RUN_ARGS = -np $(MPI_NP) $(_MPI_OPTS)
_PRUN_ARGS    = -np $(PRUN_NP) -$(PRUN_PP) -t $(TIMEOUT) OMPI_OPTS="$(PRUN_MPI)" $(PRUN_ARG) $(PRUN_ARGS)

_TIMEOUT_PREF = timeout --foreground --signal SIGTERM --kill-after 15s $(shell date -ud "1970/01/01 $(TIMEOUT)" +%s)s
_LOGFILE_SUFF = | tee -a $(LOGFILE) $(LOGGLOB)
_ERRFILE_SUFF = | tee -a $(ERRFILE) $(ERRGLOB)

.PHONY: clean clean-logs distclean info memcheck mpirun profile prun run
.NOTPARALLEL: clean clean-logs distclean info
.PRECIOUS: $(LOG_DIRS)

memcheck: CFLAGS   += -g
memcheck: CXXFLAGS += -g
memcheck: RUN_PREFIX += $(MEMCHK) $(_MEMCHK_ARGS)
memcheck: debug run;

profile: CFLAGS   += -g -pg
profile: CXXFLAGS += -g -pg
profile: NVCFLAGS += -g -G -pg

mpirun: RUN_PREFIX += $(MPIRUN) $(_MPI_RUN_ARGS)
mpirun: run;

prun: RUN_PREFIX += $(PRUN) $(_PRUN_ARGS)
prun: run;

distclean:: clean-logs;

info::
	#directory LOG_DIR: $(LOG_DIR)

$(LOG_DIRS):
	@ mkdir -p $@

clean::
	@ rm -f $(PROF_OUT)
	@ rm -f $(LOG_OUT)
	@ rm -f $(addprefix $(LOG_DIR)/,$(LOG_OUT))

clean-logs:
	@ rm -f $(LOG_DIR)/*.$(PROF_EXT)
	@ rm -f $(LOG_DIR)/*.$(LOG_EXT)
	@ rm -f $(LOG_DIR)/*.$(ERR_EXT)

run: $(EXECUTABLE) | $(LOG_DIRS)
	@ date $(_ERRFILE_SUFF)
	( \
		( \
			$(RUN_ENV) \
			$(_TIMEOUT_PREF) \
			$(RUN_PREFIX) $(EXECUTABLE) $(ARG) $(ARGS) $(ARG_SUFFIX) \
			$(_LOGFILE_SUFF) \
		) \
		3>&1 1>&2 2>&3 $(_ERRFILE_SUFF) \
	) 3>&1 1>&2 2>&3

profile: release run
	$(GPROF) $(EXECUTABLE)* $(PROF_OUT) > $(PROFILE)

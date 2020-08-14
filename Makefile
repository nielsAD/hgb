# Author:  Niels A.D.
# Project: HGB (https://github.com/nielsAD/hgb)
# License: Mozilla Public License, v2.0

TARGETS  := bench_pagerank graphdeg graphgen graphheat graphpart graphsan graphstride strtoidx
PACKAGES := libstarpu igraph
INCLUDES := $(MKLROOT)/include
DEFINES  := _GNU_SOURCE MKL_ILP64

CC  := mpicc
CXX := mpic++
LD  := mpic++

CFLAGS := -march=native -std=gnu11 -Wall -Wextra
CFLAGS += -Wsuggest-attribute=pure -Wsuggest-attribute=const

CXXFLAGS := -march=native -std=gnu++11 -Wall -Wextra
CXXFLAGS += -Wsuggest-attribute=pure -Wsuggest-attribute=const

override CFLAGS   += -fopenmp
override CXXFLAGS += -fopenmp
override NVCFLAGS += -Xcompiler "-march=native -fopenmp"
override LDFLAGS  += -fopenmp -lm -lz -lOpenCL -lcusparse -lcublas -lcudart -lmkl_rt -L$(MKLROOT)/lib/intel64

MPI_ENV += STARPU_SCHED STARPU_CALIBRATE
MPI_ENV += $(shell env | grep "OPENMP") $(shell env | grep "STARPU")

export CUDA_PROPAGATE_HOST_FLAGS=OFF

STARPU_SCHED          ?= dmdar
STARPU_CALIBRATE      ?= 0
STARPU_GENERATE_TRACE ?= 0
STARPU_PREFETCH       ?= 1
STARPU_STATS	      ?= 1
STARPU_MEMORY_STATS   ?= 1
STARPU_BUS_STATS      ?= 1
STARPU_WORKER_STATS   ?= 1
STARPU_FXT_PREFIX     ?= $(abspath $(LOG_DIR))/

PLOT_DIR ?= $(LOG_DIR)/plot

RUN_ENV = \
	STARPU_SCHED=$(STARPU_SCHED) \
	STARPU_CALIBRATE=$(STARPU_CALIBRATE) \
	STARPU_GENERATE_TRACE=$(STARPU_GENERATE_TRACE) \
	STARPU_PREFETCH=$(STARPU_PREFETCH) \
	STARPU_STATS=$(STARPU_STATS) \
	STARPU_MEMORY_STATS=$(STARPU_MEMORY_STATS) \
	STARPU_BUS_STATS=$(STARPU_BUS_STATS) \
	STARPU_WORKER_STATS=$(STARPU_WORKER_STATS) \
	STARPU_FXT_PREFIX=$(STARPU_FXT_PREFIX)

include scripts/compile.mk
include scripts/run.mk

.PHONY: clean-plots default move-logs plot plot-generate trace
.NOTPARALLEL: move-logs plot-generate

.DEFAULT_GOAL := default
default: all

release: DEFINES += OCL_FLAGS="-cl-fast-relaxed-math"

trace: STARPU_GENERATE_TRACE=1
trace: run;

plot: trace plot-generate;

distclean:: clean-plots;

$(PLOT_DIR):
	@ mkdir -p $@

move-logs:
	-$(foreach L,$(LOG_OUT),mv -u $(L) $(LOG_DIR);)
	-$(foreach L,$(LOG_OUT),mv -u $(STARPU_FXT_PREFIX)$(L) $(LOG_DIR);)

plot-generate: move-logs | $(PLOT_DIR)
	-@ mv -u $(LOG_DIR)/*.trace $(PLOT_DIR)
	-@ mv -u $(LOG_DIR)/*.data $(PLOT_DIR)
	-@ mv -u $(LOG_DIR)/dag.dot* $(PLOT_DIR)
	cd $(PLOT_DIR); starpu_workers_activity activity.data
	cd $(PLOT_DIR); starpu_perfmodel_display -l | cut -d '<' -f 2 | sed 's/>$$//' | xargs -n 1 starpu_perfmodel_plot -s
	# cd $(PLOT_DIR); starpu_perfmodel_display -l | cut -d '<' -f 2 | sed 's/>$$//' | xargs -n 1 starpu_codelet_profile distrib.data
	# cd $(PLOT_DIR); starpu_codelet_histo_profile distrib.data
	cd $(PLOT_DIR); sfdp -x -Teps -Goverlap=scale -O dag.dot
	cd $(PLOT_DIR); find . -iname '*.gp'  -exec gnuplot {} \;
	# cd $(PLOT_DIR); find . -iname '*.eps' -exec convert -density 300 {} {}.png \;

clean-plots:
	@ rm -f $(PLOT_DIR)/*.eps
	@ rm -f $(PLOT_DIR)/*.gp
	@ rm -f $(PLOT_DIR)/*.png

# Author:  Niels A.D.
# Project: HGB (https://github.com/nielsAD/hgb)
# License: Mozilla Public License, v2.0
#
# input: $(TARGETS), $(PACKAGES), $(DEFINES)

CC  ?= cc
CXX ?= c++
NVC ?= nvcc
CPP ?= cpp
LD  ?= cc

SRC_DIR ?= src
INC_DIR ?= include
OBJ_DIR ?= obj
BIN_DIR ?= bin
SUB_DIR ?= $(shell cd $(SRC_DIR) && find . -type d -print)

INC_EXT ?= .h .hh .hp .hxx .hpp .h++ .H .HPP
  C_EXT ?= .c .i
CXX_EXT ?= .cc .cp .cxx .cpp .c++ .CPP .C
NVC_EXT ?= .cu

NVC_SM ?= 35

CFLAGS   ?= -march=native -std=gnu11   -Wall -Wextra
CXXFLAGS ?= -march=native -std=gnu++11 -Wall -Wextra
NVCFLAGS ?= -Xcompiler "-march=native"

override CFLAGS   += $(foreach P, $(PACKAGES), $(shell pkg-config --cflags $(P)))
override CXXFLAGS += $(foreach P, $(PACKAGES), $(shell pkg-config --cflags $(P)))
override NVCFLAGS += $(foreach P, $(PACKAGES), $(shell pkg-config --cflags $(P))) $(foreach sm,$(NVC_SM),-gencode arch=compute_$(sm),code=sm_$(sm)) 
override LDFLAGS  += $(foreach P, $(PACKAGES), $(shell pkg-config --libs $(P)))

HEADERS = $(subst /./,/,$(foreach D,$(INC_DIR),$(foreach S,$(SUB_DIR),$(foreach E,$(INC_EXT),$(wildcard $(D)/$(S)/*$(E))))))
SRC_C   = $(subst /./,/,$(foreach S,$(SUB_DIR),$(foreach E,  $(C_EXT),$(wildcard $(SRC_DIR)/$(S)/*$(E)))))
SRC_CXX = $(subst /./,/,$(foreach S,$(SUB_DIR),$(foreach E,$(CXX_EXT),$(wildcard $(SRC_DIR)/$(S)/*$(E)))))
SRC_NVC = $(subst /./,/,$(foreach S,$(SUB_DIR),$(foreach E,$(NVC_EXT),$(wildcard $(SRC_DIR)/$(S)/*$(E)))))
SRC_BIN = $(foreach T,$(TARGETS),$(foreach E,$(C_EXT) $(CXX_EXT) $(NVC_EXT),$(wildcard $(SRC_DIR)/$(T)$(E))))

OBJ_C   = $(patsubst $(SRC_DIR)/%,$(OBJ_DIR)/%.o,$(SRC_C))
OBJ_CXX = $(patsubst $(SRC_DIR)/%,$(OBJ_DIR)/%.o,$(SRC_CXX))
OBJ_NVC = $(patsubst $(SRC_DIR)/%,$(OBJ_DIR)/%.o,$(SRC_NVC))
OBJ_BIN = $(patsubst $(SRC_DIR)/%,$(OBJ_DIR)/%.o,$(SRC_BIN))
OBJ_OUT = $(OBJ_C) $(OBJ_CXX) $(OBJ_NVC)
OBJ_SRC = $(filter-out $(OBJ_BIN),$(OBJ_OUT))

DEP_C   = $(patsubst %.o,%.d,$(OBJ_C))
DEP_CXX = $(patsubst %.o,%.d,$(OBJ_CXX))
DEP_OUT = $(DEP_C) $(DEP_CXX)

VER_OUT = $(patsubst %.o,%.v,$(OBJ_OUT))
BIN_OUT = $(addprefix $(BIN_DIR)/,$(TARGETS))
DIR_OUT = $(sort $(subst /./,/,$(addsuffix /,$(BIN_DIR) $(OBJ_DIR) $(addprefix $(OBJ_DIR)/,$(SUB_DIR)))))

$(DEP_C):   CPP_FLAGS += $(CFLAGS)
$(DEP_CXX): CPP_FLAGS += $(CXXFLAGS)

-include $(addsuffix *,$(DEP_OUT))
.PHONY: all clean debug distclean force info release $(TARGETS)
.NOTPARALLEL: clean distclean info
.PRECIOUS: $(DIR_OUT) $(OBJ_OUT) $(DEP_OUT) $(VER_OUT) $(BIN_OUT)
.SECONDEXPANSION:
.SUFFIXES:

VER_HASH = " \
	[[[$(DEFINES)]]] \
	[[[$(INCLUDES)]]] [[[$(INC_DIR)]]] \
	[[[$(CC)]]]  [[[$(CFLAGS)]]] \
	[[[$(CXX)]]] [[[$(CXXFLAGS)]]] \
	[[[$(NVC)]]] [[[$(NVCFLAGS)]]] \
	[[[$(LD)]]]  [[[$(LDFLAGS)]]] \
"

$(TARGETS): %: $(BIN_DIR)/%;

all: $(BIN_OUT);

debug: CFLAGS   += -g3
debug: CXXFLAGS += -g3
debug: NVCFLAGS += -g -G
debug: $(BIN_OUT);

sanitize: CFLAGS   += -fsanitize=address -fsanitize=leak -fsanitize=undefined -O1 -fno-omit-frame-pointer -g
sanitize: CXXFLAGS += -fsanitize=address -fsanitize=leak -fsanitize=undefined -O1 -fno-omit-frame-pointer -g
sanitize: LDFLAGS  += -fsanitize=address -fsanitize=leak -fsanitize=undefined
sanitize: $(BIN_OUT);

release: CFLAGS   += -Ofast
release: CXXFLAGS += -Ofast
release: NVCFLAGS += -O3 -use_fast_math
release: DEFINES  += NDEBUG
release: $(BIN_OUT);

info::
	$(CC) $(CFLAGS) -v -Q --help=target
	$(CXX) $(CXXFLAGS) -v -Q --help=target
	$(NVC) -V
	$(CPP) --version
	$(LD) -v
	#directory INC_DIR: $(INC_DIR)
	#directory SRC_DIR: $(SRC_DIR)
	#directory SUB_DIR: $(SUB_DIR)
	#directory OBJ_DIR: $(OBJ_DIR)
	#directory BIN_DIR: $(BIN_DIR)

clean::
	@ rm -f $(OBJ_OUT)
	@ rm -f $(DEP_OUT)
	@ rm -f $(VER_OUT)
	@ rm -f $(addsuffix *,$(BIN_OUT))

distclean:: clean
	@$(foreach D,$(filter-out ./ .. .,$(DIR_OUT)),$(shell test -d $(D) && rmdir -p $(D)))

$(DIR_OUT):
	@ mkdir -p $@

$(VER_OUT): $(OBJ_DIR)/%.v: force | $$(dir $(OBJ_DIR)/%)
	@ echo '$(VER_HASH)' | cmp -s - $@ || echo '$(VER_HASH)' > $@

$(DEP_OUT): $(OBJ_DIR)/%.d: $(SRC_DIR)/%
	@ $(CPP) $(CPP_FLAGS) -MM -MP -MF $@ $(addprefix -D, $(DEFINES)) $(addprefix -I, $(INC_DIR)) $<
	@ sed -i 's|\(^$(notdir $(basename $*)).o:\)|$(OBJ_DIR)/$*.o:|' $@

$(OBJ_C): $(OBJ_DIR)/%.o: $(SRC_DIR)/% $(OBJ_DIR)/%.v $(OBJ_DIR)/%.d
	$(CC) $(CFLAGS) $(addprefix -D, $(DEFINES)) $(addprefix -I, $(INC_DIR) $(INCLUDES)) -c $< -o $@

$(OBJ_CXX): $(OBJ_DIR)/%.o: $(SRC_DIR)/% $(OBJ_DIR)/%.v $(OBJ_DIR)/%.d
	$(CXX) $(CXXFLAGS) $(addprefix -D, $(DEFINES)) $(addprefix -I, $(INC_DIR) $(INCLUDES)) -c $< -o $@

$(OBJ_NVC): $(OBJ_DIR)/%.o: $(SRC_DIR)/% $(OBJ_DIR)/%.v $(HEADERS)
	$(NVC) $(NVCFLAGS) $(addprefix -D, $(DEFINES)) $(addprefix -I, $(INC_DIR) $(INCLUDES)) -c $< -o $@

$(BIN_OUT): $(BIN_DIR)/%: $(OBJ_SRC) $$(filter $(OBJ_BIN),$(foreach E,$(C_EXT) $(CXX_EXT) $(NVC_EXT),$(OBJ_DIR)/%$(E).o)) | $(BIN_DIR)/
	$(LD) $^ $(LDFLAGS) -o $@

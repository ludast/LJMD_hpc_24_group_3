#-------------------------------------------------------------------------------
# OMP_NUM_THREADS=2 mpirun --use-hwthread-cpus -n 4 --map-by PE-LIST=0,1,2,3,4,5,6,7:HWTCPUS:ORDERED ./build/bin/a_mpi.out

CC=mpic++
CFLAGS= -I ${INC_DIR} -acc -Minfo
#CFLAGS= -I ${INC_DIR} -Wall -Werror -O3 -march=native -fopenmp # -DPRINT

# Release folders 
PROJECT_NAME = a_mpi.out
BUILD_DIR=./build
INC_DIR=./include
DAT_DIR=./build/dat

# Dependencies
SRC_DIR=$(INC_DIR)
SRCS=$(shell find $(SRC_DIR) -name '*.c')

# Objects
_OBJECTS = $(SRCS:$(SRC_DIR)/%=$(BUILD_DIR)/%)
OBJECTS = $(patsubst %.c,%.o,$(_OBJECTS)) ${BUILD_DIR}/main.o

# Target
BIN_DIR=$(BUILD_DIR)/bin
TARGET_EXEC=$(BIN_DIR)/$(PROJECT_NAME)


#-------------------------------------------------------------------------------
# RELEASE
release: $(TARGET_EXEC)

$(TARGET_EXEC): $(OBJECTS)
	$(CC) $(CFLAGS) $^ -o $@
$(BUILD_DIR)/%.o: $(SRC_DIR)/%.c
	$(CC) -c $(CFLAGS) $^ -o $@
$(BUILD_DIR)/main.o: main.cpp
	$(CC) -c $(CFLAGS) $^ -o $@

#-------------------------------------------------------------------------------
# DEBUG
#DEBUG_FLAGS= -g -O0 -fsanitize=leak -fsanitize=address -fsanitize=undefined
DEBUG_FLAGS= -I $(DEBUG_INC_DIR) -Wall -Werror -O0 -g

DEBUG_BUILD_DIR=./debug/
DEBUG_INC_DIR=$(INC_DIR)

DEBUG_SRC_DIR=$(SRC_DIR)
DEBUG_SRCS=$(SRCS)

_DEBUG_OBJECTS=$(DEBUG_SRCS:$(DEBUG_SRC_DIR)/%=$(DEBUG_BUILD_DIR)/%)
DEBUG_OBJECTS=$(patsubst %.c,%.o,$(_DEBUG_OBJECTS)) $(DEBUG_BUILD_DIR)/main.o

DEBUG_BIN_DIR=$(DEBUG_BUILD_DIR)/bin
DEBUG_TARGET_EXEC=$(DEBUG_BIN_DIR)/$(PROJECT_NAME)



debug: $(DEBUG_TARGET_EXEC)

$(DEBUG_TARGET_EXEC): $(DEBUG_OBJECTS)
	$(CC) $(DEBUG_FLAGS) $^ -o $@
$(DEBUG_BUILD_DIR)/%.o: $(DEBUG_SRC_DIR)/%.c
	$(CC) -c $(DEBUG_FLAGS) $^ -o $@
$(DEBUG_BUILD_DIR)/main.o: main.cpp
	$(CC) -c $(DEBUG_FLAGS) $^ -o $@


#-------------------------------------------------------------------------------
clean:
	rm -rf $(BUILD_DIR) $(DEBUG_BUILD_DIR)
	mkdir -p $(BUILD_DIR) $(BIN_DIR) $(DAT_DIR) $(DEBUG_BUILD_DIR) $(DEBUG_BIN_DIR)

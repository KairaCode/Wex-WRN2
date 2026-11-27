# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Ludovico Zanichelli

OUT      ?= build
ifeq ($(origin CC),default)
    CC := gcc
endif
AR       ?= ar
NVCC     ?= nvcc
OPENMP   ?= 0
CUDA_ARCH ?=

UNAME_S := $(shell uname -s 2>/dev/null || echo Unknown)

ifeq ($(OS),Windows_NT)
    PLATFORM := windows
else ifneq (,$(findstring MINGW,$(UNAME_S)))
    PLATFORM := windows
else ifneq (,$(findstring MSYS,$(UNAME_S)))
    PLATFORM := windows
else ifneq (,$(findstring CYGWIN,$(UNAME_S)))
    PLATFORM := windows
else ifeq ($(UNAME_S),Darwin)
    PLATFORM := macos
else ifeq ($(UNAME_S),Linux)
    PLATFORM := linux
else
    PLATFORM := posix
endif

BASE_CFLAGS := -std=c99 -O2 -Wall -Wextra -Wno-unused-parameter -Wno-unused-function \
               -Iinclude -Isrc -I../c
BASE_LDFLAGS :=

ifeq ($(OPENMP),1)
    BASE_CFLAGS  += -fopenmp
    BASE_LDFLAGS += -fopenmp
endif

ifeq ($(PLATFORM),windows)
    IO_SRC       := src/wex_io_win.c
    LIB_EXT      := .dll
    EXE_EXT      := .exe
    DLL_CFLAGS   := -DLIBWEX_BUILDING_DLL
    SHARED_FLAGS := -shared
    SYS_LDFLAGS  :=
    CUDA_EXT     := .dll
    CUDA_SHARED  := --shared
else ifeq ($(PLATFORM),macos)
    IO_SRC       := src/wex_io_posix.c
    LIB_EXT      := .dylib
    EXE_EXT      :=
    DLL_CFLAGS   := -fPIC
    SHARED_FLAGS := -dynamiclib -install_name @rpath/libwex.dylib
    SYS_LDFLAGS  := -lpthread -ldl
    CUDA_EXT     := .dylib
    CUDA_SHARED  := --shared -Xcompiler -fPIC
else ifeq ($(PLATFORM),linux)
    IO_SRC       := src/wex_io_linux.c
    LIB_EXT      := .so
    EXE_EXT      :=
    DLL_CFLAGS   := -fPIC
    SHARED_FLAGS := -shared
    SYS_LDFLAGS  := -lpthread -ldl
    CUDA_EXT     := .so
    CUDA_SHARED  := --shared -Xcompiler -fPIC
else
    IO_SRC       := src/wex_io_posix.c
    LIB_EXT      := .so
    EXE_EXT      :=
    DLL_CFLAGS   := -fPIC
    SHARED_FLAGS := -shared
    SYS_LDFLAGS  := -lpthread -ldl
    CUDA_EXT     := .so
    CUDA_SHARED  := --shared -Xcompiler -fPIC
endif

CORE_SRCS    := src/wex_store.c src/wex_cache.c src/wex_verify.c $(IO_SRC)
IO_TEST_SRCS := tests/test_libwex_io.c $(IO_SRC) src/wex_verify.c
C_TEST_SRCS  := tests/test_libwex_c.c $(CORE_SRCS)

HEADERS      := include/wex_store.h include/wex_loader.h \
                src/wex_cache.h src/wex_io.h src/wex_io_common.h src/wex_dev_ops.h

LIB_TARGET   := $(OUT)/libwex$(LIB_EXT)
STATIC_TARGET := $(OUT)/libwex.a
CUDA_TARGET  := $(OUT)/libwex_cuda$(CUDA_EXT)
TEST_IO_BIN  := $(OUT)/test_libwex_io$(EXE_EXT)
TEST_C_BIN   := $(OUT)/test_libwex_c$(EXE_EXT)

SHARED_OBJS  := $(patsubst src/%.c,$(OUT)/obj_shared/%.o,$(CORE_SRCS))
STATIC_OBJS  := $(patsubst src/%.c,$(OUT)/obj_static/%.o,$(CORE_SRCS))

NVCC_FLAGS   := -O2 -std=c++17 $(CUDA_SHARED) -Iinclude -Isrc -I../c
ifneq ($(CUDA_ARCH),)
    NVCC_FLAGS += -arch=$(CUDA_ARCH)
endif

.PHONY: all shared static cuda tests test clean

all: shared

shared: $(LIB_TARGET)

static: $(STATIC_TARGET)

cuda: $(CUDA_TARGET)

tests: $(LIB_TARGET) $(TEST_IO_BIN) $(TEST_C_BIN)

$(OUT) $(OUT)/obj_shared $(OUT)/obj_static:
	mkdir -p "$@"

$(OUT)/obj_shared/%.o: src/%.c $(HEADERS) | $(OUT)/obj_shared
	$(CC) $(BASE_CFLAGS) $(DLL_CFLAGS) $(CFLAGS) -c "$<" -o "$@"

$(OUT)/obj_static/%.o: src/%.c $(HEADERS) | $(OUT)/obj_static
	$(CC) $(BASE_CFLAGS) $(CFLAGS) -c "$<" -o "$@"

$(LIB_TARGET): $(SHARED_OBJS) | $(OUT)
	$(CC) $(SHARED_FLAGS) $(SHARED_OBJS) $(BASE_LDFLAGS) $(SYS_LDFLAGS) $(LDFLAGS) -o "$@"

$(STATIC_TARGET): $(STATIC_OBJS) | $(OUT)
	$(AR) rcs "$@" $(STATIC_OBJS)

$(CUDA_TARGET): src/wex_dev_cuda.cu src/wex_dev_ops.h | $(OUT)
	$(NVCC) $(NVCC_FLAGS) $(NVCCFLAGS) "$<" -o "$@"

$(TEST_IO_BIN): $(IO_TEST_SRCS) $(HEADERS) | $(OUT)
	$(CC) $(BASE_CFLAGS) $(CFLAGS) $(IO_TEST_SRCS) $(BASE_LDFLAGS) $(SYS_LDFLAGS) $(LDFLAGS) -o "$@"

$(TEST_C_BIN): $(C_TEST_SRCS) $(HEADERS) | $(OUT)
	$(CC) $(BASE_CFLAGS) $(CFLAGS) $(C_TEST_SRCS) $(BASE_LDFLAGS) $(SYS_LDFLAGS) $(LDFLAGS) -o "$@"

test: tests
	"$(TEST_IO_BIN)"
	cd "$(OUT)" && "./test_libwex_c$(EXE_EXT)" --lib "./libwex$(LIB_EXT)"

clean:
	rm -rf "$(OUT)"

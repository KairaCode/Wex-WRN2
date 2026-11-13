# SPDX-License-Identifier: Apache-2.0
# Derived from colibri c/Makefile (https://github.com/JustVugg/colibri), Copyright 2026 Vincenzo Fornaro.
# Modified by Ludovico Zanichelli, 2026.
# REFERENCE EXCERPT, not compilable on its own.
# Source: KAIRA src/colibri/c/Makefile at commit 88ddb07 (2026-11-07), as committed.
# Line numbers are of that commit; the live tree moves them. Code between ranges is omitted.
# Top-level Makefile rules for the WEX unit tests and for wex_decode.o in the test/segment builds.

# ==== Makefile:2447-2452  segment build links wex_decode.o ====
SEGMENT_V4_UNITS = $(filter-out COLI_V4_UNIT_PROMPT \
	COLI_V4_UNIT_GENERATE_STATS,$(V4_TARGET_UNITS)) $(V4_SEGMENT_UNIT)
SEGMENT_V4_OBJS = $(addprefix $(SEGMENT_BUILD_DIR)/,$(addsuffix .o,$(SEGMENT_V4_UNITS)))
SEGMENT_V4_REGISTRY_OBJ = $(SEGMENT_BUILD_DIR)/expert_store_registry.o
SEGMENT_ALL_OBJS = $(SEGMENT_ENGINE_OBJS) $(SEGMENT_V4_OBJS) $(SEGMENT_BUILD_DIR)/v4_route_tap.o \
	$(SEGMENT_V4_REGISTRY_OBJ) $(SEGMENT_BUILD_DIR)/wex_decode.o

# ==== Makefile:2488-2493  segment build rule for wex_decode.o ====
$(SEGMENT_V4_REGISTRY_OBJ): expert_store_registry.c | $(SEGMENT_BUILD_DIR)
	$(CC) $(SEGMENT_CPU_CFLAGS) -c expert_store_registry.c -o $@

$(SEGMENT_BUILD_DIR)/wex_decode.o: wex_decode.c | $(SEGMENT_BUILD_DIR)
	$(CC) $(SEGMENT_CPU_CFLAGS) -c wex_decode.c -o $@


# ==== Makefile:2590-2646  V4_TEST_LINK_OBJS (wex_decode.o), tests/test_wex_format, tests/test_expert_store_wex, test_deepseek_v4 deps ====
	COLI_V4_UNIT_ST.o \
	COLI_V4_UNIT_CONFIG.o \
	COLI_V4_UNIT_MATH.o \
	COLI_V4_UNIT_ROUTE_BF16.o \
	COLI_V4_UNIT_SPARSE_ATTENTION.o \
	COLI_V4_UNIT_RESOURCE_PLAN.o \
	COLI_V4_UNIT_PROMPT.o \
	$(filter-out COLI_V4_UNIT_EXPERT_STORE.o COLI_V4_UNIT_EXPERT.o,$(addsuffix .o,$(V4_TEST_UNITS))) \
	$(V4_HOT_TEST_OBJ) \
	$(V4_BATCH_TEST_OBJ) \
	COLI_V4_UNIT_EXPERT_ROWS16.o \
	COLI_V4_UNIT_NATIVE_QUANT.o \
	COLI_V4_UNIT_NATIVE_QUANT_DUAL.o \
	COLI_V4_UNIT_NATIVE_QUANT_ROWS16.o \
	wex_decode.o
# Windows engine objects are compiled with -DCOLI_V4_GPU_TIER (the CUDA tier
# is resolved from coli_cuda_dsv4*.dll at runtime), so the matvec references in
# native_quant.h resolve through the GPU unit + the DLL loader. Both are inert
# without a DLL next to the test binary.
ifneq ($(IS_WIN),)
V4_TEST_LINK_OBJS += $(V4_GPU_TEST_OBJ) backend_loader_dsv4.o
V4_TEST_EXTRA_TARGETS = $(V4_GPU_TEST_OBJ) backend_loader_dsv4.o
endif

tests/test_wex_format$(EXE): tests/test_wex_format.c
	$(CC) $(CFLAGS) $< -o $@ $(LDFLAGS)

# The WEX expert-store test links four units; keeping them as objects is what
# lets -MMD write a complete .d for tests/test_expert_store_wex.exe (#1741 rule:
# one unit per compile command).
tests/expert_store_registry.o: expert_store_registry.c
	$(CC) $(CFLAGS) -c $< -o $@
tests/expert_store_wex.o: expert_store_wex.c
	$(CC) $(CFLAGS) -c $< -o $@
tests/wex_decode.o: wex_decode.c
	$(CC) $(CFLAGS) -c $< -o $@

tests/test_expert_store_wex$(EXE): tests/test_expert_store_wex.c tests/expert_store_registry.o \
		tests/expert_store_wex.o tests/wex_decode.o
	$(CC) $(CFLAGS) $< -o $@ tests/expert_store_registry.o tests/expert_store_wex.o tests/wex_decode.o -pthread $(LDFLAGS)

V4_TEST_OBJ_TARGETS = deepseek-v4-test-objs \
	COLI_V4_UNIT_ST.o COLI_V4_UNIT_CONFIG.o COLI_V4_UNIT_MATH.o COLI_V4_UNIT_SPARSE_ATTENTION.o \
	COLI_V4_UNIT_ROUTE_BF16.o \
	COLI_V4_UNIT_RESOURCE_PLAN.o COLI_V4_UNIT_PROMPT.o COLI_V4_UNIT_NATIVE_QUANT.o \
	COLI_V4_UNIT_NATIVE_QUANT_DUAL.o COLI_V4_UNIT_NATIVE_QUANT_ROWS16.o \
	COLI_V4_UNIT_EXPERT_ROWS16.o \
	$(V4_HOT_TEST_OBJ) \
	$(V4_BATCH_TEST_OBJ) \
	wex_decode.o \
	$(V4_TEST_EXTRA_TARGETS)

tests/test_deepseek_v4$(EXE): tests/test_deepseek_v4.c sse41_kernels.h deepseek_v4.c deepseek_v4.h compat.h expert_store_wex.h wex_decode.c wex_decode.h wex_format.h wex_palette.h wex_neural_rans.h \
		Makefile.deepseek-v4.units Makefile.deepseek-v4
	$(MAKE) -f Makefile.deepseek-v4 ARCH=$(ARCH) $(V4_TEST_OBJ_TARGETS)
	$(CC) $(CFLAGS) -DCOLI_V4_TEST_HOOKS $< $(V4_TEST_LINK_OBJS) \
		-o $@ -pthread $(LDFLAGS)

# ==== Makefile:2655-2657  KAIRA-only test tests/test_v4_gpu_priority (WEX prefetch pool test + KAIRA GPU hooks) ====
tests/test_v4_gpu_priority$(EXE): tests/test_v4_gpu_priority.c tests/test_deepseek_v4$(EXE) | v4-test-objs
	$(CC) $(CFLAGS) -DCOLI_V4_TEST_HOOKS $< $(V4_TEST_LINK_OBJS) \
		-o $@ -pthread $(LDFLAGS)

# ==== Makefile:3284-3291  AUTODEP_OBJS lists the WEX test objects ====
ENGINE_RULES := $(foreach n,$(RULE_NAMES),$(if $(findstring /,$(n)),,$(if $(wildcard $(n).c),$(n))))
AUTODEP_TESTS := $(patsubst tests/%,%,$(filter tests/%,$(RULE_NAMES)))
AUTODEP_OBJS      = backend_vulkan.o vk_tier.o backend_loader.o backend_xdna.o qwen36_tier.o \
	tests/backend_xdna_lane.o tests/edge_runtime.o tests/edge_runtime_nocuda.o \
	tests/segment_runtime.o tests/segment_runtime_nocuda.o \
	tests/segment_conformance_fixtures.o tests/deepseek_v4_native_quant.o \
	tests/expert_store_registry.o tests/expert_store_wex.o tests/wex_decode.o \
	qwen36_qpack.o qpack.o tests/qwen36_qpack.o tests/qpack_qwen36.o tests/qpack.o \


# ==== Makefile:2667-2676 at commit cade51d (branch wnr3-20261110; the ranges above are 88ddb07)  KAIRA-only WNR3 engine-store test tests/test_v4_wex_nr3 (repo copy: c/tests/test_v4_wex_nr3.c) and its CPU stand-in helper tests/fake_wex_cuda.dll (c/tests/fake_wex_cuda.c) ====
# WNR3 records on the engine's own expert store (Windows): a CPU stand-in for
# kaira_wex_cuda.dll is built next to the test, which loads it by path.
ifneq ($(IS_WIN),)
tests/fake_wex_cuda.dll: tests/fake_wex_cuda.c wex_format.h wex_neural_rans.h
	$(CC) -O2 -shared -static-libgcc $< -o $@
V4_WEX_NR3_DEPS = tests/fake_wex_cuda.dll
endif
tests/test_v4_wex_nr3$(EXE): tests/test_v4_wex_nr3.c tests/test_deepseek_v4$(EXE) $(V4_WEX_NR3_DEPS) | v4-test-objs
	$(CC) $(CFLAGS) -DCOLI_V4_TEST_HOOKS $< $(V4_TEST_LINK_OBJS) \
		-o $@ -pthread $(LDFLAGS)

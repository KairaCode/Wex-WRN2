# SPDX-License-Identifier: Apache-2.0
# Derived from colibri c/Makefile.deepseek-v4 (https://github.com/JustVugg/colibri), Copyright 2026 Vincenzo Fornaro.
# Modified by Ludovico Zanichelli, 2026.
# REFERENCE EXCERPT, not compilable on its own.
# Source: KAIRA src/colibri/c/Makefile.deepseek-v4 at commit 88ddb07 (2026-11-07), as committed.
# Line numbers are of that commit; the live tree moves them. Code between ranges is omitted.
# Engine build rules that link the WEX objects into deepseek_v4.exe (Windows only).

# ==== Makefile.deepseek-v4:57-62  Windows block: V4_WIN_EXTRA_OBJS gains expert_store_wex.o wex_decode.o ====
	-DCOLI_V4_GPU_TIER \
	-DCOLI_V4_EXPERIMENTAL_DUAL_EXPERT_LOADER
# -lws2_32: the routing tap's loopback UDP socket (v4_route_tap.c).
LDFLAGS = -lm -fopenmp -static -pthread -lws2_32
V4_BINARY := deepseek_v4.exe
V4_WIN_EXTRA_OBJS = backend_loader_dsv4.o expert_store_wex.o wex_decode.o

# ==== Makefile.deepseek-v4:112-115  non-Windows block: no WEX objects ====
	-DCOLI_V4_EXPERIMENTAL_DUAL_EXPERT_LOADER
LDFLAGS = -lm -fopenmp -pthread
V4_BINARY := deepseek_v4
V4_WIN_EXTRA_OBJS =

# ==== Makefile.deepseek-v4:203-206  CUDA build: backend linked statically (dsv4_cuda_wex_direct_available returns 1) ====
V4_NVCCFLAGS = -O3 -std=c++20 --expt-relaxed-constexpr --expt-extended-lambda -Xcompiler=-fPIC \
               $(V4_CUDA_GENCODE) $(V4_DEEPGEMM_FLAGS)
CFLAGS += -DCOLI_V4_GPU_TIER
V4_WIN_EXTRA_OBJS += backend_cuda_dsv4.o

# ==== Makefile.deepseek-v4:269-270  V4_OBJS includes V4_WIN_EXTRA_OBJS ====
V4_OBJS = $(TARGET_OBJS) $(V4_WIN_EXTRA_OBJS) $(REGISTRY_OBJ) $(V4_VK_OBJ) $(V4_TAP_OBJ)
REGISTRY_OBJ = expert_store_registry.o

# ==== Makefile.deepseek-v4:352-378  rules: expert_store_wex.o, wex_decode.o, WEX headers as dependencies of the engine units and the HOT test unit ====
$(REGISTRY_OBJ): expert_store_registry.c expert_store_registry.h expert_store.h \
	$(V4_FLAGS_STAMP)
	$(CC) $(CFLAGS) -c expert_store_registry.c -o $@

expert_store_wex.o: expert_store_wex.c expert_store_wex.h wex_decode.h wex_format.h wex_palette.h wex_neural_rans.h expert_store.h $(V4_FLAGS_STAMP)
	$(CC) $(CFLAGS) -c expert_store_wex.c -o $@

# WEX record reconstruction (CPU codec + optional kaira_wex_cuda.dll). Its own
# object: the expert-store unit calls it, and the native tests link that unit
# without the WEX store backend and its registry/planned-store dependencies.
wex_decode.o: wex_decode.c wex_decode.h wex_format.h wex_palette.h wex_neural_rans.h $(V4_FLAGS_STAMP)
	$(CC) $(CFLAGS) -c wex_decode.c -o $@

$(TARGET_OBJS) $(TEST_UNIT_OBJS): %.o: deepseek_v4.c deepseek_v4.h \
	deepseek_v4_internal.h deepseek_v4_dspark.inc st.h json.h compat.h tensor.h quant.h fp8_format.h sse41_kernels.h \
	route_trace.h vk_tier.h deepseek_v4_chain.h vk_chain.h \
	native_quant.h native_quant_batch.h native_quant_dual.h \
	native_quant_fp4_rows16.h expert_store_registry.h expert_store_wex.h wex_decode.h wex_format.h wex_palette.h wex_neural_rans.h v4_route_tap.h v4_np_route.h v4_np_gates.h \
	$(V4_FLAGS_STAMP)
	$(CC) $(CFLAGS) -D$* -c deepseek_v4.c -o $@

$(V4_HOT_TEST_OBJ): deepseek_v4.c deepseek_v4.h deepseek_v4_internal.h \
		st.h json.h compat.h tensor.h quant.h fp8_format.h route_trace.h sse41_kernels.h \
		native_quant.h native_quant_fp4_rows16.h expert_store_registry.h expert_store_wex.h wex_decode.h wex_format.h wex_palette.h wex_neural_rans.h v4_route_tap.h \
		$(V4_FLAGS_STAMP)
	$(CC) $(CFLAGS) -DCOLI_V4_TEST_HOOKS \
		-DCOLI_V4_UNIT_EXPERT_STORE_HOT_ROWS16 -c deepseek_v4.c -o $@


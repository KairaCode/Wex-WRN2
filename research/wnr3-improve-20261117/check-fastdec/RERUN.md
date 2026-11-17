# check-fastdec (skeptic) — rerun

Own files: check_ref.c (production wex_nr3_decode vs the patched default entry point, full-slab memcmp) and logs.
Copy ../fastdec/* into a working directory and build with ../fastdec/build.sh first. check_ref usage is in its
main(): `check_ref MODEL ENTRIES KERNEL LANES records...` (KERNEL -1 = default entry point); build it once with
-march=haswell (kernel 3 default) and once with -march=x86-64-v4 -mavx512vbmi2 (kernel 4 default), linking
/tmp/wnr3-data/libwex_nr3_tools.so. Timing: abx.py as in ../fastdec/RERUN.md, adding a baseline built with
-DWEX_FAST_LANES=6 (logs/time_check_cpu1_vs_k1x6.txt).

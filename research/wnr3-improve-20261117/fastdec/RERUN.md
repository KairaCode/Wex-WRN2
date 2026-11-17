# fastdec (lane, faster decoder for today's format) — rerun

Copied without `out/` (binaries, generated negative-control sources) and without wex_fast_decode.orig.h (= the
repo's wex4/research/fast-cpu-decode/wex_fast_decode.h; copy it here as wex_fast_decode.orig.h) and two
intermediate kernel headers. `wex_fast_decode.patch` applies with `patch -p1` from a Wex-WRN2 checkout root.

    ./build.sh
    ./run_time.sh > logs/time_rerun.txt 2>&1
    AB_CPU=1 python3 abx.py 9 1 1 orig_k1x4=out/test_orig:10 new_avx2_k3x4=out/test_h:22 new_avx512_k4x3=out/test_z:26 -- \
        /tmp/wnr3-data/recs/rec_00_007.bin /tmp/wnr3-data/recs/rec_20_073.bin /tmp/wnr3-data/recs/rec_39_120.bin
    : > logs/equal_z_121.txt; ./run_equal2.sh
    ./run_mutate.sh
    python3 negcontrol.py 6000 99 /tmp/wnr3-data/recs/rec_00_007.bin /tmp/wnr3-data/recs/rec_20_073.bin /tmp/wnr3-data/recs/rec_39_120.bin
    for a in 0 1 2 4 8 16; do gcc -std=c11 -O3 -march=haswell -fopenmp -DWEXF_ABL=$a -I/workspace/Wex-WRN2/c test_abl.c -o out/abl_$a; done
    AB_CPU=1 python3 abx.py 7 1 1 full=out/abl_0:10 nohash=out/abl_1:10 nostore=out/abl_2:10 nosearch=out/abl_4:10 norenorm=out/abl_8:10 fixedctx=out/abl_16:10 -- \
        /tmp/wnr3-data/recs/rec_00_007.bin /tmp/wnr3-data/recs/rec_20_073.bin

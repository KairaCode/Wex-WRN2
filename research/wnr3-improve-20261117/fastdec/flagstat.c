/* Bucket-table statistics of the production model: for each bucket size 2^S, the buckets that hold 1, 2, 3+ symbols
 * over the reachable weight contexts (3 matrices x 7 srel x 37 (previous <= maximum) + block start). */
#include "wex_fast_decode_research.h"
#include <stdio.h>
int main(int argc, char **argv) {
    FILE *f = fopen(argv[1], "rb"); static unsigned char blob[WEX_NR2_MODEL_BYTES]; WexNr2Model model;
    if (!f || fread(blob, 1, sizeof blob, f) != sizeof blob || wex_nr2_model_open(blob, sizeof blob, &model)) return 2;
    for (unsigned S = 3; S <= 9; S++) {
        unsigned long n1 = 0, n2 = 0, n3 = 0, maxflag = 0, ctxs = 0; double slots3 = 0;
        for (unsigned m = 0; m < 3; m++) for (unsigned srel = 0; srel < 7; srel++) for (unsigned e = 0; e < 65; e++) {
            if (e < 64 && (e & 7) > (e >> 3)) continue;
            unsigned prev = e == 64 ? 8 : e & 7, mx = e == 64 ? 8 : e >> 3, c[17], fl = 0;
            const unsigned char *cdf = wex_nr2_weight_cdf(&model, m, wex_nr2_context(srel, prev, mx));
            for (unsigned i = 0; i <= 16; i++) c[i] = wex_nr_u16(cdf + 2 * i);
            ctxs++;
            for (unsigned lo = 0; lo < 16384; lo += 1u << S) {
                unsigned n = 0;
                for (unsigned s = 0; s < 16; s++) if (c[s + 1] > lo && c[s] < lo + (1u << S)) n++;
                if (n == 1) n1++; else if (n == 2) n2++; else { n3++; fl++; slots3 += 1u << S; }
            }
            if (fl > maxflag) maxflag = fl;
        }
        printf("S=%u: %lu contexts, buckets 1 sym %lu, 2 sym %lu, 3+ sym %lu (max %lu per context; %.3f%% of slots, uniform context weight)\n",
               S, ctxs, n1, n2, n3, maxflag, 100.0 * slots3 / (ctxs * 16384.0));
    }
    return 0;
}

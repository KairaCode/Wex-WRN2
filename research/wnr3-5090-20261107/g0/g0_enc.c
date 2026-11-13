/* WNR2-g0 encoder side (lab only). Includes the SCRATCH COPY wex_neural_rans_g0.h, whose only change is
 * that wex_nr2_median returns g0_anchor when g0_anchor >= 0 (trainer + encoder). */
#include "wex_neural_rans_g0.h"
#define X __declspec(dllexport)

int g0_anchor = -1;

static void lay(WexNr2Layout *l, const uint32_t *sc, const uint32_t *sy) {
    for (int m = 0; m < 3; m++) { l->scale_count[m] = sc[m]; l->symbol_count[m] = sy[m]; }
}

X void g0_set_anchor(int a) { g0_anchor = a; }

X void g0_count(const uint32_t *sc, const uint32_t *sy, const unsigned char *const *scales,
                const unsigned char *const *weights, uint64_t *wcounts, uint64_t *scounts) {
    WexNr2Layout l; lay(&l, sc, sy);
    wex_nr2_count(&l, scales, weights, wcounts, scounts);
}

X void g0_model_build(const uint64_t *wcounts, const uint64_t *scounts, unsigned char *blob) {
    wex_nr2_model_build(wcounts, scounts, blob);
}

/* returns 0 and the unpadded size in *written; -2 bad model, -1 encode failure */
X int g0_encode(const unsigned char *blob, const uint32_t *sc, const uint32_t *sy,
                const unsigned char *const *scales, const unsigned char *const *weights,
                uint32_t wchunk, uint32_t schunk, unsigned char *out, size_t cap, size_t *written) {
    WexNr2Model model; WexNr2Layout l; lay(&l, sc, sy);
    if (wex_nr2_model_open(blob, WEX_NR2_MODEL_BYTES, &model)) return -2;
    return wex_nr2_encode(&model, &l, scales, weights, wchunk, schunk, out, cap, written);
}

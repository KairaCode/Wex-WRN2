/* WNR2-g0 decoder side (lab only): the UNCHANGED KAIRA CPU decoder, src/colibri/c/wex_neural_rans.h
 * (compiled with -I src/colibri/c; separate translation unit from the scratch encoder copy). */
#include "wex_neural_rans.h"
#define X __declspec(dllexport)

/* 0 ok; -2 model invalid; -3 parse failed; -4 decode failed. median_out gets record bytes 24..26. */
X int g0_decode(const unsigned char *blob, const unsigned char *record, size_t total, const uint32_t *sc,
                const uint32_t *sy, unsigned char *const *scales_out, unsigned char *const *weights_out,
                unsigned char *median_out, uint32_t *model_id_out) {
    WexNr2Model model; WexNr2Layout l; WexNr2Record v;
    for (int m = 0; m < 3; m++) { l.scale_count[m] = sc[m]; l.symbol_count[m] = sy[m]; }
    if (wex_nr2_model_open(blob, WEX_NR2_MODEL_BYTES, &model)) return -2;
    *model_id_out = model.id;
    if (wex_nr2_parse(record, total, &l, model.id, &v)) return -3;
    memcpy(median_out, v.median, 3);
    return wex_nr2_decode(&v, &l, &model, scales_out, weights_out) ? -4 : 0;
}

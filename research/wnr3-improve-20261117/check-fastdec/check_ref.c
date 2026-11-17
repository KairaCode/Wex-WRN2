/* Independent cross-check: fast decoder default entry point (and selected kernels) vs the production CPU path
 * wex_nr3_decode (libwex_nr3_tools.so -> wex_decode_nr2_record), full raw slab memcmp + FNV-1a hash per record. */
#define _DEFAULT_SOURCE
#include "wex_fast_decode.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
int wex_nr3_decode(const unsigned char *blob, const WexEntry *e, const unsigned char *record, uint32_t layer, uint32_t expert, unsigned char *out);
static unsigned char *rd(const char *p, size_t *n) { FILE *f = fopen(p, "rb"); long s; unsigned char *d;
  if (!f) { perror(p); exit(2);} fseek(f,0,SEEK_END); s=ftell(f); fseek(f,0,SEEK_SET); d=malloc(s+1); if(fread(d,1,s,f)!=(size_t)s) exit(2); fclose(f); *n=s; return d; }
static uint64_t fnv(const unsigned char *p, size_t n) { uint64_t h = 1469598103934665603ull; while (n--) { h ^= *p++; h *= 1099511628211ull; } return h; }
int main(int argc, char **argv) {
  size_t mb, eb; unsigned char *blob = rd(argv[1], &mb), *ents = rd(argv[2], &eb);
  WexNr2Model model; static WexFastModel tab; int bad = 0, i;
  int kern = atoi(argv[3]), lanes = atoi(argv[4]);  /* kern -1 = default entry point */
  if (wex_nr2_model_open(blob, mb, &model)) return 2;
  wex_fast_model_init(&tab, &model);
  for (i = 5; i < argc; i++) {
    int L, E; const char *b = strrchr(argv[i], '/'); b = b ? b + 1 : argv[i];
    sscanf(b, "rec_%d_%d.bin", &L, &E);
    WexEntry e; memcpy(&e, ents + (size_t)(L*192+E)*sizeof(WexEntry), sizeof e);
    size_t rn; unsigned char *rec = rd(argv[i], &rn);
    uint64_t raw, al; wex_entry_raw_sizes(&e, &raw, &al);
    unsigned char *ref = malloc(raw), *fast = malloc(raw); memset(fast, 0x5a, raw);
    int s1 = wex_nr3_decode(blob, &e, rec, L, E, ref);
    WexNr2Layout lay; WexNr2Record v; wex_nr2_layout(&e, &lay);
    if (wex_nr3_parse(rec, e.total_bytes, &lay, model.id, &v)) { printf("%s parse fail\n", b); bad++; continue; }
    unsigned char *sc[3], *wt[3]; size_t off = 0; unsigned m;
    for (m = 0; m < 3; m++) { sc[m] = fast + off; off += lay.scale_count[m]; }
    for (m = 0; m < 3; m++) { wt[m] = fast + off; off += lay.symbol_count[m] / 2; }
    int s2;
    if (kern < 0) s2 = wex_fast_decode(&v, &lay, &model, sc, wt);
    else { WexFastOptions o = {1, kern, lanes, 1}; s2 = wex_fast_decode_with(&tab, &v, &lay, &model, sc, wt, &o); }
    int eq = off == raw && !memcmp(ref, fast, raw);
    printf("%s ref %d fast %d slab %llu laid %zu equal %d fnv_ref %016llx fnv_fast %016llx\n", b, s1, s2,
           (unsigned long long)raw, off, eq, (unsigned long long)fnv(ref, raw), (unsigned long long)fnv(fast, raw));
    bad += s1 || s2 || !eq; free(ref); free(fast); free(rec);
  }
  printf("records %d bad %d\n", argc - 5, bad); return bad != 0;
}

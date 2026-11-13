/* SPDX-License-Identifier: Apache-2.0
 * Derived from colibri c/tests/test_qwen38_int4_experts.c (https://github.com/JustVugg/colibri), Copyright 2026 Vincenzo Fornaro.
 * Modified by Ludovico Zanichelli, 2026. */
/* qwen38's routed experts as MXFP4 records (qwen38_wex.h: the AMD Quark checkpoint, from a
 * WEX1 container or from the Quark shards).
 *
 * Pinned with no model:
 *  - q38_weight_matmul on an MXFP4 weight is quant.h's matmul_mxfp4: equal to the dequantized
 *    reference (e2m1 times 2^(e-127), summed in double) to float rounding, on the release's two
 *    expert shapes and on a partial group, for one row and for a batch; every batch row is the
 *    bits of that row alone;
 *  - the record layout: w1 = gate, w2 = down, w3 = up, scales before nibbles, 2,611,200 bytes
 *    (2,613,248 aligned) for the release's geometry.
 * With tools/make_qwen38_tiny_mxfp4.py's fixture on argv (make qwen38-tiny-mxfp4-check):
 *  - Q38_MXFP4_DIR (the Quark shards) attaches; every expert loaded through the single LRU path
 *    and the parallel batch path holds the Quark tensors' bytes, read by name;
 *  - its decode is the Python reference (mxfp4_ref.f32) to the bit: an identity input turns the
 *    matmul into the dequantized matrix, every product exact;
 *  - COLI_WEX_PATH (the WEX1 container) gives the same record bytes as the Quark shards;
 *  - the engine's own FP8 experts of the snapshot, expanded to f32, are those values to the bit
 *    (the fixture writes the MXFP4 values as exact e4m3), so the FP8 run is the MXFP4 run's control;
 *  - a container for another geometry, or both sources at once, is refused (child process);
 *  - the WEX4 form of the WEX1 container (tests/wex4_fixture.h) gives the same records; a pinned
 *    COLI_WEX_ROOT refuses another root and the WEX1 form; COLI_WEX_VERIFY=1 stops the engine on
 *    a changed record byte, which loads without it (wex4_checks; also run under --nr3, whose
 *    multi-page records check a striped read's own hash against a differing mirror).
 * With a fixture holding experts-mxfp4.nr3.wex (make_qwen38_tiny_mxfp4.py --inter 512, then
 * tools/build_nr3.py container): the WNR3 records decode to the WEX1 container's bytes on the
 * single path, on the batch path (one helper call per wave when the helper has feature 32:
 * tests/fake_wex_cuda.dll next to the test unless COLI_WEX_CUDA_LIB says otherwise; =0 is the
 * CPU decoder) and when the helper fails a wave record; a record naming another expert stops
 * the engine (child process). */
#define _GNU_SOURCE
#define QWEN38_NO_MAIN
#include "../qwen38.c"
#include "../compat.h"   /* setenv/unsetenv: MinGW has neither */
#include "wex4_fixture.h"

static int fails;
static void ck(int ok,const char *what){ if(ok){printf("  ok   %s\n",what);return;} printf("  FAIL %s\n",what); fails++; }

static unsigned g_seed=20261111;
static unsigned rnd_u(void){ g_seed=g_seed*1103515245u+12345u; return g_seed>>8; }
static float rnd(void){ return (float)(rnd_u()&0xFFFF)/32768.f-1.f; }

static const double E2M1[16]={0,.5,1,1.5,2,3,4,6,-0.,-.5,-1,-1.5,-2,-3,-4,-6};
static double value_at(const uint8_t *q4,const uint8_t *e8,int I,int o,int i){
    uint8_t b=q4[(int64_t)o*((I+1)/2)+i/2];
    return E2M1[i&1?b>>4:b&15]*ldexp(1.0,(int)e8[(int64_t)o*((I+31)/32)+i/32]-127);
}

static void kernel_case(int O,int I,int S){
    int64_t rb=(I+1)/2,ng=(I+31)/32;
    uint8_t *q4=malloc((size_t)(O*rb)),*e8=malloc((size_t)(O*ng));
    float *x=malloc((size_t)S*I*sizeof(float)),*y=malloc((size_t)S*O*sizeof(float)),*one=malloc((size_t)O*sizeof(float));
    for(int64_t k=0;k<O*rb;k++)q4[k]=(uint8_t)rnd_u();
    for(int64_t k=0;k<O*ng;k++)e8[k]=(uint8_t)(120+rnd_u()%8);
    for(int64_t k=0;k<(int64_t)S*I;k++)x[k]=rnd()*3.f;
    Q38Weight w={0};
    w.data=q4;w.e8=e8;w.rows=O;w.cols=I;w.elements=(int64_t)O*I;w.scale_count=O*ng;w.kind=Q38_WEIGHT_MXFP4;
    q38_weight_matmul(y,x,&w,S,I,O);
    double tol=4.0*(sqrt((double)I)+8.0)*FLT_EPSILON;int ok=1;
    for(int s=0;s<S&&ok;s++)for(int o=0;o<O&&ok;o++){
        double a=0,mag=0;
        for(int i=0;i<I;i++){double t=value_at(q4,e8,I,o,i)*x[(int64_t)s*I+i];a+=t;mag+=fabs(t);}
        ok=fabs((double)y[(int64_t)s*O+o]-a)<=tol*mag+1e-30;
    }
    char what[160];
    snprintf(what,sizeof what,"[%d x %d], %d row(s): equals the dequantized reference to float rounding",O,I,S);
    ck(ok,what);
    int same=1;
    for(int s=0;s<S&&same;s++){
        q38_weight_matmul(one,x+(int64_t)s*I,&w,1,I,O);
        same=!memcmp(one,y+(int64_t)s*O,(size_t)O*sizeof(float));
    }
    snprintf(what,sizeof what,"[%d x %d]: each batch row is the bits of that row alone",O,I);
    if(S>1)ck(same,what);
    ck(q38_weight_bytes(&w)==(uint64_t)(O*rb+O*ng),"q38_weight_bytes counts nibbles and exponents");
    free(q4);free(e8);free(x);free(y);free(one);
}

/* the Quark tensor of one matrix, read by name */
static uint8_t *quark(Model *m,int layer,int expert,int k,int scale,int64_t *bytes){
    static const char *proj[3]={"gate_proj","up_proj","down_proj"};
    char suffix[192],name[320];
    snprintf(suffix,sizeof suffix,"mlp.experts.%d.%s.weight%s",expert,proj[k],scale?"_scale":"");
    q38_name(m,name,sizeof name,layer,suffix);
    st_tensor *t=st_find(&m->x4->S,name);
    if(!t)return NULL;
    uint8_t *out=malloc((size_t)t->nbytes);
    st_read_raw(&m->x4->S,name,out,0);*bytes=t->nbytes;
    return out;
}

/* identity rows in: the matrix out, every product exact (one nonzero term per sum) */
static float *decoded(const Q38Weight *w){
    int O=w->rows,I=w->cols;
    float *x=calloc((size_t)I*I,sizeof(float)),*y=malloc((size_t)I*O*sizeof(float)),*t=malloc((size_t)O*I*sizeof(float));
    for(int i=0;i<I;i++)x[(int64_t)i*I+i]=1.f;
    q38_weight_matmul(y,x,w,I,I,O);
    for(int i=0;i<I;i++)for(int o=0;o<O;o++)t[(int64_t)o*I+i]=y[(int64_t)i*O+o];
    free(x);free(y);
    return t;
}

static float *g_ref; static int64_t g_ref_n;
static const float *ref_at(Model *m,int layer,int expert,int k){
    int64_t H=m->c.hidden,I=m->c.inter,per=3*H*I,at=((int64_t)layer*m->c.experts+expert)*per+k*H*I;
    return at+H*I<=g_ref_n?g_ref+at:NULL;
}

/* a loaded slot holds the Quark bytes (when m reads the Quark shards) or `want` (a record
 * read elsewhere), and decodes to the Python reference to the bit */
static int slot_matches(Model *m,const Slot *slot,int layer,int expert,const uint8_t *want){
    const Q38Weight *w[3]={&slot->gate,&slot->up,&slot->down};
    int ok=slot->eid==expert;
    for(int k=0;k<3&&ok;k++){
        int O=w[k]->rows,I=w[k]->cols;
        ok=w[k]->kind==Q38_WEIGHT_MXFP4&&O==(k<2?m->c.inter:m->c.hidden)&&I==(k<2?m->c.hidden:m->c.inter);
        if(ok&&!want){
            int64_t nw=0,ns=0;
            uint8_t *nib=quark(m,layer,expert,k,0,&nw),*sc=quark(m,layer,expert,k,1,&ns);
            ok=nib&&sc&&nw==(int64_t)O*(I/2)&&ns==(int64_t)O*(I/32)&&
               !memcmp(w[k]->data,nib,(size_t)nw)&&!memcmp(w[k]->e8,sc,(size_t)ns);
            free(nib);free(sc);
        }
        const float *ref=ref_at(m,layer,expert,k);
        if(ok&&ref){float *t=decoded(w[k]);ok=!memcmp(t,ref,(size_t)O*I*sizeof(float));free(t);}
        else if(ok)ok=0;
    }
    if(ok&&want)ok=!memcmp((const uint8_t*)slot->gate.e8,want,(size_t)m->x4->mx_raw);
    return ok;
}

static void fixture_checks(const char *snap){
    char quark_dir[2100],wex_path[2100],ref_path[2100];
    snprintf(quark_dir,sizeof quark_dir,"%s/quark",snap);
    snprintf(wex_path,sizeof wex_path,"%s/experts-mxfp4.wex",snap);
    snprintf(ref_path,sizeof ref_path,"%s/mxfp4_ref.f32",snap);
    printf("the MXFP4 experts of %s\n",snap);
    FILE *f=fopen(ref_path,"rb");
    if(f){fseek(f,0,SEEK_END);g_ref_n=ftell(f)/4;fseek(f,0,SEEK_SET);g_ref=malloc((size_t)g_ref_n*4);
          if(fread(g_ref,4,(size_t)g_ref_n,f)!=(size_t)g_ref_n)g_ref_n=0;fclose(f);}
    ck(g_ref_n>0,"mxfp4_ref.f32 read");

    setenv("Q38_MXFP4_DIR",quark_dir,1);
    Model m;model_init(&m,snap,1,8);q38_expert_mx4_attach(&m);
    ck(m.x4&&m.x4->mx&&!m.x4->wex_n,"Q38_MXFP4_DIR attaches the Quark shards");
    if(!m.x4){q38_model_free(&m);return;}
    int L=m.c.layers,E=m.c.experts,ok=1;
    ck(m.x4->mx_raw==3LL*m.c.inter*m.c.hidden*17/32&&m.x4->record_bytes%4096==0,"record = nibbles + exponents, 4096-aligned slab");
    uint8_t **records=calloc((size_t)L*E,sizeof(uint8_t*));
    for(int layer=0;layer<L;layer++)for(int e=0;e<E;e++){
        Slot *slot=q38_expert_get(&m,layer,e);
        ok&=slot_matches(&m,slot,layer,e,NULL);
        records[layer*E+e]=malloc((size_t)m.x4->mx_raw);
        memcpy(records[layer*E+e],slot->gate.e8,(size_t)m.x4->mx_raw);   /* gate's exponents start the record */
        ok&=((uintptr_t)slot->gate.e8&4095)==0;
    }
    ck(ok,"single LRU path (cap 1): Quark bytes by name, 4096-aligned, decode = Python reference to the bit");
    ck(m.expert_weight_reads==(uint64_t)L*E&&m.miss==m.expert_weight_reads,"one record per miss");
    q38_model_free(&m);

    unsetenv("Q38_MXFP4_DIR");setenv("COLI_WEX_PATH",wex_path,1);
    for(int cap=1;cap<=E;cap+=E-1){
        Model b;model_init(&b,snap,cap,8);q38_expert_mx4_attach(&b);
        ck(b.x4&&b.x4->mx&&b.x4->wex_n==1,"COLI_WEX_PATH attaches the WEX1 container");
        if(!b.x4){q38_model_free(&b);continue;}
        int *all=malloc((size_t)E*sizeof(int));Slot **sel=malloc((size_t)E*sizeof(Slot*));
        for(int e=0;e<E;e++)all[e]=E-1-e;
        ok=1;int batched=1;
        for(int layer=0;layer<L;layer++){
            if(cap==E){
                batched&=q38_expert_get_batch(&b,layer,all,E,sel);
                for(int e=0;e<E&&batched;e++)ok&=slot_matches(&b,sel[e],layer,all[e],records[layer*E+all[e]]);
            }else for(int e=0;e<E;e++)ok&=slot_matches(&b,q38_expert_get(&b,layer,e),layer,e,records[layer*E+e]);
        }
        if(cap==E){
            ck(batched&&b.expert_parallel_batches==(uint64_t)L,"parallel batch path taken for every layer (WEX)");
            ck(ok,"WEX1 batch path: every record == the Quark shards' bytes, decode = Python reference");
        }else ck(ok,"WEX1 single path: every record == the Quark shards' bytes, decode = Python reference");
        free(all);free(sel);q38_model_free(&b);
    }
    /* a mirror copy shares the reads (COLI_WEX_WEIGHTS 1:3); one whose header differs is ignored */
    char mdir[2100],mpath[2200];snprintf(mdir,sizeof mdir,"%s/mirror",snap);snprintf(mpath,sizeof mpath,"%s/experts-mxfp4.wex",mdir);
#ifdef _WIN32
    _mkdir(mdir);
#else
    mkdir(mdir,0755);
#endif
    for(int pass=0;pass<2;pass++){
        FILE *in=fopen(wex_path,"rb"),*out=fopen(mpath,"wb");char buf[65536];size_t got;long at=0;
        while(in&&out&&(got=fread(buf,1,sizeof buf,in))>0){
            if(pass&&at==0)buf[1000]^=1;   /* the header page's padding: same records, another container */
            fwrite(buf,1,got,out);at+=(long)got;
        }
        if(in)fclose(in);
        if(out)fclose(out);
        setenv("COLI_WEX_EXTRA_MIRRORS",mdir,1);setenv("COLI_WEX_WEIGHTS","1,3",1);setenv("COLI_WEX_STRIPE","1",1);
        Model r;model_init(&r,snap,1,8);q38_expert_mx4_attach(&r);
        if(!pass){
            ck(r.x4&&r.x4->wex_n==2&&r.x4->wex_cut[0]==64&&r.x4->wex_cut[1]==256,"COLI_WEX_EXTRA_MIRRORS adds the copy, COLI_WEX_WEIGHTS 1,3 splits 25/75");
            ck(r.x4&&r.x4->wex_stripe,"COLI_WEX_STRIPE=1 with a mirror turns striping on");
            ok=r.x4!=NULL;
            for(int layer=0;ok&&layer<L;layer++)for(int e=0;e<E;e++)ok&=slot_matches(&r,q38_expert_get(&r,layer,e),layer,e,records[layer*E+e]);
            ck(ok,"primary + mirror reads: every record == the Quark shards' bytes");
        }else ck(r.x4&&r.x4->wex_n==1&&!r.x4->wex_stripe,"a mirror whose header page differs is ignored (and nothing stripes)");
        q38_model_free(&r);
    }
    remove(mpath);rmdir(mdir);
    unsetenv("COLI_WEX_EXTRA_MIRRORS");unsetenv("COLI_WEX_WEIGHTS");unsetenv("COLI_WEX_STRIPE");
    unsetenv("COLI_WEX_PATH");

    /* the snapshot's own FP8 experts, expanded to f32: the same values to the bit */
    setenv("Q38_NATIVE_FP8","0",1);setenv("Q38_NATIVE_BF16","0",1);
    Model fm;model_init(&fm,snap,1,8);q38_expert_mx4_attach(&fm);q38_expert_int4_attach(&fm,snap);
    ck(fm.x4==NULL,"without COLI_WEX_PATH / Q38_MXFP4_DIR the snapshot's experts stay");
    ok=1;
    for(int layer=0;layer<L;layer++)for(int e=0;e<E;e++){
        Slot *slot=q38_expert_get(&fm,layer,e);const Q38Weight *w[3]={&slot->gate,&slot->up,&slot->down};
        for(int k=0;k<3;k++){
            const float *ref=ref_at(&fm,layer,e,k);
            ok&=w[k]->kind==Q38_WEIGHT_F32&&ref&&!memcmp(w[k]->data,ref,(size_t)w[k]->rows*w[k]->cols*sizeof(float));
        }
    }
    ck(ok,"the snapshot's FP8 experts expanded to f32 are the MXFP4 values to the bit (the FP8 run is the control)");
    unsetenv("Q38_NATIVE_FP8");unsetenv("Q38_NATIVE_BF16");
    q38_model_free(&fm);
    for(int i=0;i<L*E;i++)free(records[i]);
    free(records);free(g_ref);
}

/* the WEX1 records of every expert (single path), the reference for the WNR3 decode */
static uint8_t **wex1_records(const char *snap,const char *path){
    setenv("COLI_WEX_PATH",path,1);
    Model m;model_init(&m,snap,1,8);q38_expert_mx4_attach(&m);
    int n=m.c.layers*m.c.experts;
    uint8_t **records=calloc((size_t)n,sizeof(uint8_t*));
    for(int layer=0;layer<m.c.layers;layer++)for(int e=0;e<m.c.experts;e++){
        Slot *slot=q38_expert_get(&m,layer,e);
        records[layer*m.c.experts+e]=malloc((size_t)m.x4->mx_raw);
        memcpy(records[layer*m.c.experts+e],slot->gate.e8,(size_t)m.x4->mx_raw);
    }
    q38_model_free(&m);unsetenv("COLI_WEX_PATH");
    return records;
}

/* COLI_WEX_STRIPE=1 end to end on the fixture's multi-page WEX1 records (single path: every miss
 * alone): with a byte-identical mirror every record is the reference; with a mirror whose records
 * differ in their first and last page (same header and table) none is, so each read took a part
 * from the mirror. */
static void stripe_records(const char *snap,const char *wex_path,uint8_t **want){
    char mdir[2100],mpath[2200];snprintf(mdir,sizeof mdir,"%s/mirror_stripe",snap);
    snprintf(mpath,sizeof mpath,"%s/experts-mxfp4.wex",mdir);
#ifdef _WIN32
    _mkdir(mdir);
#else
    mkdir(mdir,0755);
#endif
    FILE *in=fopen(wex_path,"rb");long size=0;unsigned char *bytes=NULL;
    if(in){fseek(in,0,SEEK_END);size=ftell(in);fseek(in,0,SEEK_SET);bytes=malloc((size_t)size);
           if(bytes&&fread(bytes,1,(size_t)size,in)!=(size_t)size){free(bytes);bytes=NULL;}fclose(in);}
    if(!bytes){ck(0,"stripe: read the WEX1 container");return;}
    setenv("COLI_WEX_PATH",wex_path,1);setenv("COLI_WEX_EXTRA_MIRRORS",mdir,1);setenv("COLI_WEX_STRIPE","1",1);
    for(int pass=0;pass<2;pass++){
        const WexHeader *h=(const WexHeader*)bytes;
        Model m;model_init(&m,snap,1,8);int L=m.c.layers,E=m.c.experts;q38_model_free(&m);
        if(pass)for(int i=0;i<L*E;i++){
            const WexEntry *e=(const WexEntry*)(bytes+h->table_offset)+i;
            bytes[e->file_offset]^=1;bytes[e->file_offset+e->total_bytes-WEX_ALIGN]^=1;   /* first and last page, both inside the payload */
        }
        FILE *out=fopen(mpath,"wb");if(out){fwrite(bytes,1,(size_t)size,out);fclose(out);}
        model_init(&m,snap,1,8);q38_expert_mx4_attach(&m);
        int pages=m.x4?(int)(m.x4->wex_entry[0].total_bytes/WEX_ALIGN):0,same=0;
        ck(m.x4&&m.x4->wex_n==2&&m.x4->wex_stripe&&pages>=2,"stripe: the mirror attaches, the records span pages");
        for(int layer=0;m.x4&&layer<L;layer++)for(int e=0;e<E;e++)
            same+=!memcmp(q38_expert_get(&m,layer,e)->gate.e8,want[layer*E+e],(size_t)m.x4->mx_raw);
        if(!pass)ck(same==L*E,"striped reads with a byte-identical mirror: every record == the WEX1 record");
        else ck(same==0,"striped reads with a mirror whose records differ: no record is whole (a part came from it)");
        if(pass&&m.x4){
            /* the tier warmstart's shape: q38_load_expert from an omp parallel loop is not alone, so
             * it reads whole from its hashed drive: whole exactly when that drive is the primary */
            int whole=0,primary=0,n=L*E;
            for(int i=0;i<n;i++)primary+=(int)((((uint32_t)i*2654435761u)>>16)&255)<m.x4->wex_cut[0];
            #pragma omp parallel for num_threads(2) reduction(+:whole)
            for(int i=0;i<n;i++){
                Slot tmp;memset(&tmp,0,sizeof tmp);
                q38_load_expert(&m,i/E,i%E,&tmp);
                whole+=!memcmp(tmp.gate.e8,want[i],(size_t)m.x4->mx_raw);
                free(tmp.int4_slab);
            }
            ck(primary>0&&primary<n&&whole==primary,"reads from a parallel region are not striped (whole iff the hashed drive is the primary)");
        }
        q38_model_free(&m);
    }
    unsetenv("COLI_WEX_PATH");unsetenv("COLI_WEX_EXTRA_MIRRORS");unsetenv("COLI_WEX_STRIPE");
    free(bytes);remove(mpath);rmdir(mdir);
}

static void nr3_checks(const char *self,const char *snap){
    char wex_path[2100],nr3_path[2100];
    snprintf(wex_path,sizeof wex_path,"%s/experts-mxfp4.wex",snap);
    snprintf(nr3_path,sizeof nr3_path,"%s/experts-mxfp4.nr3.wex",snap);
    FILE *probe=fopen(nr3_path,"rb");
    if(!probe){printf("(no %s: the WNR3 checks need make qwen38-tiny-mxfp4-nr3-generate)\n",nr3_path);return;}
    fclose(probe);
#ifdef _WIN32
    char helper[MAX_PATH+64];DWORD len=GetModuleFileNameA(NULL,helper,MAX_PATH);
    char *slash=len?strrchr(helper,'\\'):NULL;
    if(slash&&!getenv("COLI_WEX_CUDA_LIB")){strcpy(slash+1,"fake_wex_cuda.dll");setenv("COLI_WEX_CUDA_LIB",helper,1);}
#endif
    const char *lib=getenv("COLI_WEX_CUDA_LIB");
    printf("the WNR3 records of %s (helper: %s)\n",snap,lib?lib:"next to the test");
    uint8_t **want=wex1_records(snap,wex_path);
    stripe_records(snap,wex_path,want);
    setenv("COLI_WEX_PATH",nr3_path,1);
    int ok=1,L=0,E=0,codec5=0,fake=lib&&strstr(lib,"fake_wex_cuda")!=NULL;
    /* run 0: cap 1, the single path; 1: cap E, one batch per layer; 2: the same, the fake
     * helper failing record 1 of every wave (it is read and decoded again alone); 3: the batch
     * on the loader thread with Q38_ASYNC_MISS's done flags, each record compared the moment
     * its flag is set (a wave must set them only after the GPU decode) */
    for(int run=0;run<4;run++){
        if(run>=2&&!fake)break;
        if(run==2)setenv("FAKE_WEX_BATCH_FAIL","1",1);
        Model b;model_init(&b,snap,1,8);L=b.c.layers;E=b.c.experts;q38_model_free(&b);
        model_init(&b,snap,run?E:1,8);q38_expert_mx4_attach(&b);
        ck(b.x4&&b.x4->mx&&b.x4->nr3&&b.x4->wex_n==1,"COLI_WEX_PATH attaches the WNR3 container");
        if(!b.x4){q38_model_free(&b);continue;}
        codec5=0;
        for(int i=0;i<L*E;i++)codec5+=b.x4->wex_entry[i].flags==WEX_ENTRY_NEURAL_RANS3;
        uint64_t decoded0=coli_wex_cuda_decode_count();
        int *all=malloc((size_t)E*sizeof(int));Slot **sel=malloc((size_t)E*sizeof(Slot*));
        for(int e=0;e<E;e++)all[e]=E-1-e;
        ok=1;
        for(int layer=0;layer<L;layer++){
            if(run==3){
                Q38ExpertLoadJob *jobs=NULL;int n=0,*done=(int*)calloc((size_t)E,sizeof(int));
                ok&=done&&q38_expert_batch_reserve(&b,layer,all,E,sel,NULL,0,&jobs,&n)&&n==E;
                if(ok&&q38_loader_start(&b,layer,jobs,n,0,done)){
                    for(int j=0;j<n;j++){
                        q38_job_wait(done,j);
                        ok&=!memcmp(jobs[j].slot->gate.e8,want[layer*E+jobs[j].expert],(size_t)b.x4->mx_raw);
                    }
                    q38_loader_join(&b);
                }else ok=0;
                if(jobs)q38_expert_batch_publish(&b,layer,jobs,n);
                free(jobs);free(done);
            }else if(run){
                ok&=q38_expert_get_batch(&b,layer,all,E,sel);
                for(int e=0;e<E&&ok;e++)ok&=!memcmp(sel[e]->gate.e8,want[layer*E+all[e]],(size_t)b.x4->mx_raw);
            }else for(int e=0;e<E;e++)ok&=!memcmp(q38_expert_get(&b,layer,e)->gate.e8,want[layer*E+e],(size_t)b.x4->mx_raw);
        }
        uint64_t helper=coli_wex_cuda_decode_count()-decoded0;
        int wave=q38_mx4_wave(&b,E);
        if(run==0)ck(ok,"WNR3 single path: every decoded record == the WEX1 record");
        else if(run==1){
            ck(ok,"WNR3 batch path: every decoded record == the WEX1 record");
            if(wave)ck(helper==(uint64_t)codec5,"the batch path's waves decoded every WNR3 record in the helper (feature 32)");
            else printf("  (no helper with feature 32: the waves decode record by record%s)\n",helper?" in the helper":" on the CPU");
        }else if(run==2)ck(ok&&wave&&helper==(uint64_t)codec5,"a wave record the helper fails is read and decoded again: == the WEX1 record");
        else ck(ok&&wave&&helper==(uint64_t)codec5,"Q38_ASYNC_MISS's done flags on a WNR3 wave: a record staged at its flag is decoded == the WEX1 record");
        free(all);free(sel);q38_model_free(&b);
        unsetenv("FAKE_WEX_BATCH_FAIL");
    }
    ck(codec5==L*E,"the fixture's container holds WNR3 (codec 5) records only");
    unsetenv("COLI_WEX_PATH");
    for(int i=0;i<L*E;i++)free(want[i]);
    free(want);
    /* a record that names another expert stops the engine */
    char bad[2200];snprintf(bad,sizeof bad,"%s/owner_bad.nr3.wex",snap);
    FILE *in=fopen(nr3_path,"rb"),*out=fopen(bad,"wb");
    if(in&&out){
        fseek(in,0,SEEK_END);long size=ftell(in);fseek(in,0,SEEK_SET);
        unsigned char *bytes=malloc((size_t)size);
        if(bytes&&fread(bytes,1,(size_t)size,in)==(size_t)size){
            const WexHeader *h=(const WexHeader*)bytes;const WexEntry *e0=(const WexEntry*)(bytes+h->table_offset);
            bytes[e0->file_offset+28]^=1;   /* layer 0 expert 0's record now names expert 1 */
            fwrite(bytes,1,(size_t)size,out);
        }
        free(bytes);
    }
    if(in)fclose(in);
    if(out)fclose(out);
    char cmd[8000];
    snprintf(cmd,sizeof cmd,"\"\"%s\" --refuse-load \"%s\" \"%s\" > nul 2>&1\"",self,snap,bad);
#ifndef _WIN32
    snprintf(cmd,sizeof cmd,"'%s' --refuse-load '%s' '%s' > /dev/null 2>&1",self,snap,bad);
#endif
    ck(system(cmd)!=0,"a WNR3 record naming another expert stops the engine");
    remove(bad);
}

/* argv[0] --refuse-load <snap> <container>: attach, then load layer 0 expert 0 (must exit 1) */
static int refuse_load_child(const char *snap,const char *path){
    setenv("COLI_WEX_PATH",path,1);
    Model m;model_init(&m,snap,1,8);q38_expert_mx4_attach(&m);
    q38_expert_get(&m,0,0);
    return 0;
}

/* argv[0] --refuse <snap> <case>: attach in a child, which must exit 1 */
static int refuse_child(const char *snap,const char *what){
    Model m;model_init(&m,snap,1,8);
    if(!strcmp(what,"both")){
        char d[2100],p[2100];snprintf(d,sizeof d,"%s/quark",snap);snprintf(p,sizeof p,"%s/experts-mxfp4.wex",snap);
        setenv("Q38_MXFP4_DIR",d,1);setenv("COLI_WEX_PATH",p,1);
    }else setenv("COLI_WEX_PATH",what,1);
    q38_expert_mx4_attach(&m);
    return 0;   /* attached: the parent reports the failure */
}
static void refusals(const char *self,const char *snap){
    char cmd[8000],bad[2100];
    snprintf(bad,sizeof bad,"%s/mxfp4_ref.f32",snap);   /* not a container */
    const char *cases[2]={"both",bad};
    const char *names[2]={"both sources set at once is refused","a file that is not this model's WEX1 is refused"};
    for(int i=0;i<2;i++){
        snprintf(cmd,sizeof cmd,"\"\"%s\" --refuse \"%s\" \"%s\" > nul 2>&1\"",self,snap,cases[i]);   /* cmd strips the outer pair */
#ifndef _WIN32
        snprintf(cmd,sizeof cmd,"'%s' --refuse '%s' '%s' > /dev/null 2>&1",self,snap,cases[i]);
#endif
        int rc=system(cmd);
        ck(rc!=0,names[i]);
    }
}

/* argv[0] <mode> <snap> <container> in a child; its exit status (the environment is inherited) */
static int child(const char *self,const char *mode,const char *snap,const char *path){
    char cmd[8000];
    snprintf(cmd,sizeof cmd,"\"\"%s\" %s \"%s\" \"%s\" > nul 2>&1\"",self,mode,snap,path);
#ifndef _WIN32
    snprintf(cmd,sizeof cmd,"'%s' %s '%s' '%s' > /dev/null 2>&1",self,mode,snap,path);
#endif
    return system(cmd);
}
/* the slot holds that expert's record (bytes only: the reference decode is fixture_checks') */
static int same_record(const Model *m,const Slot *slot,int expert,const uint8_t *want){
    return slot->eid==expert&&!memcmp((const uint8_t*)slot->gate.e8,want,(size_t)m->x4->mx_raw);
}
static void put_file(const char *path,const unsigned char *data,size_t bytes){
    FILE *f=fopen(path,"wb");
    if(!f||fwrite(data,1,bytes,f)!=bytes)ck(0,path);
    if(f)fclose(f);
}

/* The WEX4 form of the fixture's WEX1 container (tests/wex4_fixture.h, as the in-place upgrade
 * lays it out): the same records on the single and batch paths; COLI_WEX_ROOT pins its root
 * (another root, or the WEX1 form under a pin, is refused: W06); COLI_WEX_VERIFY=1 hashes each
 * record at its first read and a changed byte stops the engine, while without it the same file
 * still loads (no per-record cost); with multi-page records a striped read is hashed apart, so a
 * mirror whose record tail differs is caught. */
static void wex4_checks(const char *self,const char *snap){
    char wex_path[2100],w4_path[2100],bad_path[2100],root[65],wrong[65];
    snprintf(wex_path,sizeof wex_path,"%s/experts-mxfp4.wex",snap);
    snprintf(w4_path,sizeof w4_path,"%s/experts-mxfp4.wex4",snap);
    snprintf(bad_path,sizeof bad_path,"%s/bad.wex4",snap);
    FILE *in=fopen(wex_path,"rb");long size=0;unsigned char *old=NULL;
    if(in){fseek(in,0,SEEK_END);size=ftell(in);fseek(in,0,SEEK_SET);old=malloc((size_t)size);
           if(old&&fread(old,1,(size_t)size,in)!=(size_t)size){free(old);old=NULL;}fclose(in);}
    if(!old){ck(0,"WEX4: read the WEX1 container");return;}
    printf("the WEX4 form of %s\n",wex_path);
    size_t bytes;unsigned char *w4=wex4_fixture_from_wex1(old,(size_t)size,&bytes,root);
    const WexEntry *e0=(const WexEntry*)(old+((const WexHeader*)old)->table_offset);   /* layer 0 expert 0 */
    put_file(w4_path,w4,bytes);
    uint8_t **want=wex1_records(snap,wex_path);
    Model a;model_init(&a,snap,1,8);int L=a.c.layers,E=a.c.experts,ok=1;q38_model_free(&a);
    setenv("COLI_WEX_PATH",w4_path,1);
    for(int verify=0;verify<2;verify++){
        if(verify){setenv("COLI_WEX_ROOT",root,1);setenv("COLI_WEX_VERIFY","1",1);}
        for(int cap=1;cap<=E;cap+=E-1){
            Model b;model_init(&b,snap,cap,8);q38_expert_mx4_attach(&b);
            ok=b.x4&&b.x4->wex_n==1&&!b.x4->wex_sha==!verify;
            ck(ok,verify?"pinned root + COLI_WEX_VERIFY=1: the WEX4 container attaches, first-read hashing on":
                         "COLI_WEX_PATH attaches the WEX4 container (no record hashing by default)");
            if(!ok){q38_model_free(&b);continue;}
            int all[64],batched=1;Slot *sel[64];
            for(int e=0;e<E&&e<64;e++)all[e]=E-1-e;
            for(int layer=0;layer<L;layer++){
                if(cap==E){
                    batched&=q38_expert_get_batch(&b,layer,all,E,sel);
                    for(int e=0;e<E&&batched;e++)ok&=same_record(&b,sel[e],all[e],want[layer*E+all[e]]);
                }else for(int e=0;e<E;e++)ok&=same_record(&b,q38_expert_get(&b,layer,e),e,want[layer*E+e]);
            }
            for(int i=0;verify&&i<L*E;i++)ok&=b.x4->wex_seen[i]==1;
            ck(ok&&batched,cap==E?"WEX4 batch path: every record == the WEX1 record":
                                  "WEX4 single path: every record == the WEX1 record");
            q38_model_free(&b);
        }
    }
    /* one byte of layer 0 expert 0's record (the index unchanged, so the same root) */
    w4[e0->file_offset+e0->total_bytes/2]^=1;
    put_file(bad_path,w4,bytes);
    w4[e0->file_offset+e0->total_bytes/2]^=1;
    ck(child(self,"--refuse-load",snap,bad_path)!=0,"COLI_WEX_VERIFY=1: a changed record byte stops the engine at its first read");
    unsetenv("COLI_WEX_VERIFY");
    ck(child(self,"--refuse-load",snap,bad_path)==0,"without COLI_WEX_VERIFY the same file loads (no per-record check: today's behaviour)");
    ck(child(self,"--refuse",snap,w4_path)==0,"the right root pinned: attaches in a child too");
    memcpy(wrong,root,sizeof wrong);wrong[0]=wrong[0]=='0'?'1':'0';
    setenv("COLI_WEX_ROOT",wrong,1);
    ck(child(self,"--refuse",snap,w4_path)!=0,"another root pinned: refused");
    setenv("COLI_WEX_ROOT",root,1);
    ck(child(self,"--refuse",snap,wex_path)!=0,"W06: under a pinned root the WEX1 form (no root) is refused");
    unsetenv("COLI_WEX_ROOT");
    ck(child(self,"--refuse",snap,wex_path)==0,"no pin: the WEX1 form still attaches");
    if(e0->total_bytes>=2*WEX_ALIGN){
        /* a mirror whose layer 0 expert 0 differs in its last page: a lone miss of it takes its tail
         * from the mirror (drive 0 hashed, cut 25/75), which the striped layout's own hash catches */
        char mdir[2100],mpath[2200];snprintf(mdir,sizeof mdir,"%s/mirror_wex4",snap);
        snprintf(mpath,sizeof mpath,"%s/experts-mxfp4.wex4",mdir);
#ifdef _WIN32
        _mkdir(mdir);
#else
        mkdir(mdir,0755);
#endif
        w4[e0->file_offset+e0->total_bytes-WEX_ALIGN]^=1;
        put_file(mpath,w4,bytes);
        w4[e0->file_offset+e0->total_bytes-WEX_ALIGN]^=1;
        setenv("COLI_WEX_EXTRA_MIRRORS",mdir,1);setenv("COLI_WEX_WEIGHTS","1,3",1);setenv("COLI_WEX_STRIPE","1",1);
        ck(child(self,"--refuse-load",snap,w4_path)==0,"striped, no COLI_WEX_VERIFY: the differing mirror tail is not seen");
        setenv("COLI_WEX_VERIFY","1",1);
        ck(child(self,"--refuse-load",snap,w4_path)!=0,"striped + COLI_WEX_VERIFY=1: a mirror's differing record tail stops the engine");
        unsetenv("COLI_WEX_VERIFY");unsetenv("COLI_WEX_EXTRA_MIRRORS");unsetenv("COLI_WEX_WEIGHTS");unsetenv("COLI_WEX_STRIPE");
        remove(mpath);rmdir(mdir);
    }else printf("  (one-page records: the striped hash check needs the --nr3 fixture)\n");
    unsetenv("COLI_WEX_PATH");
    for(int i=0;i<L*E;i++)free(want[i]);
    free(want);free(w4);free(old);remove(w4_path);remove(bad_path);
}

/* COLI_WEX_STRIPE: the cut follows the two drives' weights on a page, and a striped read takes its
 * first part from the first descriptor and the rest from the second (two files that differ). */
static void stripe_checks(void){
    int cuts[2]={64,256};   /* COLI_WEX_WEIGHTS 1,3 */
    ck(q38_wex_stripe_cut(cuts,0,1,2613248)==160*4096&&q38_wex_stripe_cut(cuts,1,0,2613248)==479*4096,
       "stripe cut of a release record: 25/75 and 75/25 of its 638 pages");
    ck(q38_wex_stripe_cut(cuts,0,1,4096)==0&&q38_wex_stripe_cut(cuts,0,1,8192+100)==0&&
       q38_wex_stripe_cut(cuts,0,1,8192)==4096,"a one-page or unaligned record is not striped; two pages split 1+1");
    enum{N=4*4096};const char *name[2]={"q38_stripe_a.bin","q38_stripe_b.bin"};
    uint8_t *buf=NULL;
    if(posix_memalign((void**)&buf,WEX_ALIGN,N)){ck(0,"stripe buffer");return;}
    for(int f=0;f<2;f++){
        for(int i=0;i<N;i++)buf[i]=(uint8_t)(i*7+f*101);
        FILE *o=fopen(name[f],"wb");if(o){fwrite(buf,1,N,o);fclose(o);}
    }
    int fa=q38_wex_open(name[0]),fb=q38_wex_open(name[1]),ok=fa>=0&&fb>=0;
    memset(buf,0,N);
    ok=ok&&!q38_wex_read2(fa,fb,buf,N-4096,4096,4096);   /* pages 1 from a, 2-3 from b */
    for(int i=0;ok&&i<N-4096;i++)ok=buf[i]==(uint8_t)((i+4096)*7+(i<4096?0:101));
    ck(ok,"striped read: the first part from the hashed drive, the rest from the second, both at the record's offset");
    /* a two-page file: a half past its end fails, and read2 names that half (the error names its drive) */
    {FILE *o=fopen("q38_stripe_c.bin","wb");if(o){fwrite(buf,1,2*4096,o);fclose(o);}}
    int fc=q38_wex_open("q38_stripe_c.bin");
    ck(fc>=0&&fa>=0&&q38_wex_read2(fa,fc,buf,N-4096,4096,4096)==2,"striped read: the second half past its file's end -> 2");
    ck(fc>=0&&fa>=0&&q38_wex_read2(fc,fa,buf,8192,4096,8192)==1,"striped read: the first half past its file's end -> 1");
    if(fc>=0)close(fc);
    remove("q38_stripe_c.bin");
    if(fa>=0)close(fa);
    if(fb>=0)close(fb);
    compat_aligned_free(buf);remove(name[0]);remove(name[1]);
}

/* argv[0] --real <quark dir> <container> <layer>: the release checkpoint's experts of one layer,
 * read by the engine's two readers without loading the model: every WEX1 record == the Quark
 * shards' bytes (tools/repack_q38_mxfp4_wex.py --layers <layer> writes such a container). */
static int real_layer(const char *dir,const char *path,int layer){
    Model q,w;memset(&q,0,sizeof q);
    q.c.layers=48;q.c.experts=512;q.c.hidden=2560;q.c.inter=640;
    snprintf(q.prefix,sizeof q.prefix,"model.language_model");
    q.range_begin=layer;q.range_end=layer+1;w=q;
    setenv("Q38_MXFP4_DIR",dir,1);q38_expert_mx4_attach(&q);unsetenv("Q38_MXFP4_DIR");
    setenv("COLI_WEX_PATH",path,1);q38_expert_mx4_attach(&w);unsetenv("COLI_WEX_PATH");
    Slot a,b;memset(&a,0,sizeof a);memset(&b,0,sizeof b);
    int same=0;double t0=now_s();
    for(int e=0;e<q.c.experts;e++){
        q38_mx4_load(&q,layer,e,&a,0,1);q38_mx4_load(&w,layer,e,&b,0,1);
        same+=!memcmp(a.gate.e8,b.gate.e8,(size_t)q.x4->mx_raw);
    }
    printf("layer %d: %d of %d WEX1 records == the Quark shards' bytes (%.1f s)\n",layer,same,q.c.experts,now_s()-t0);
    ck(same==q.c.experts,"release layer: WEX1 store reads == safetensors reads");
    return fails?1:0;
}

/* argv[0] --real-nr3 <WEX1> <WNR3> <layer>: the release geometry's records of one layer from a WNR3
 * container (tools/build_nr3.py container --layers <layer>) against the WEX1 one: the single path
 * (helper feature 16 or the CPU) and load waves of 10 (the top-k; one helper call each, feature
 * 32, when the helper has it); prints the decoded MB/s of both. */
static int real_nr3(const char *wex,const char *nr3,int layer){
    Model w,r;memset(&w,0,sizeof w);
    w.c.layers=48;w.c.experts=512;w.c.hidden=2560;w.c.inter=640;
    w.range_begin=layer;w.range_end=layer+1;r=w;
    setenv("COLI_WEX_PATH",wex,1);q38_expert_mx4_attach(&w);
    setenv("COLI_WEX_PATH",nr3,1);q38_expert_mx4_attach(&r);unsetenv("COLI_WEX_PATH");
    ck(w.x4&&!w.x4->nr3&&r.x4&&r.x4->nr3,"WEX1 and WNR3 containers attach");
    enum{K=10};
    Slot ref,one,wave[K];memset(&ref,0,sizeof ref);memset(&one,0,sizeof one);memset(wave,0,sizeof wave);
    size_t raw=(size_t)w.x4->mx_raw;int same=0,same_wave=0,E=w.c.experts;
    uint8_t *want=malloc(raw*(size_t)E);
    for(int e=0;e<E;e++){q38_mx4_load(&w,layer,e,&ref,0,1);memcpy(want+(size_t)e*raw,ref.gate.e8,raw);}
    double t0=now_s();
    for(int e=0;e<E;e++){q38_mx4_load(&r,layer,e,&one,0,1);same+=!memcmp(one.gate.e8,want+(size_t)e*raw,raw);}
    double t_one=now_s()-t0;
    int waves=q38_mx4_wave(&r,K);
    t0=now_s();
    for(int at=0;at<E;at+=K){
        int n=E-at<K?E-at:K,experts[K];Slot *slots[K];
        for(int i=0;i<n;i++){experts[i]=at+i;slots[i]=&wave[i];}
        if(waves){for(int i=0;i<n;i++)q38_mx4_load(&r,layer,at+i,&wave[i],1,0);q38_mx4_decode_wave(&r,layer,n,experts,slots);}
        else for(int i=0;i<n;i++)q38_mx4_load(&r,layer,at+i,&wave[i],0,0);
        for(int i=0;i<n;i++)same_wave+=!memcmp(wave[i].gate.e8,want+(size_t)(at+i)*raw,raw);
    }
    double t_wave=now_s()-t0;
    printf("layer %d: %d of %d WNR3 records == WEX1 on the single path (%.0f MB/s decoded incl. reads), "
           "%d of %d in waves of %d%s (%.0f MB/s)\n",layer,same,E,raw*(double)E/t_one/1e6,same_wave,E,K,
           waves?" through one helper call each":" (no feature 32: record by record)",raw*(double)E/t_wave/1e6);
    ck(same==E&&same_wave==E,"release layer: WNR3 records decode to the WEX1 records");
    free(want);free(ref.int4_slab);free(one.int4_slab);for(int i=0;i<K;i++)free(wave[i].int4_slab);
    return fails?1:0;
}

int main(int argc,char **argv){
    if(argc==4&&!strcmp(argv[1],"--refuse"))return refuse_child(argv[2],argv[3]);
    if(argc==5&&!strcmp(argv[1],"--real"))return real_layer(argv[2],argv[3],atoi(argv[4]));
    if(argc==5&&!strcmp(argv[1],"--real-nr3"))return real_nr3(argv[2],argv[3],atoi(argv[4]));
    if(argc==4&&!strcmp(argv[1],"--refuse-load"))return refuse_load_child(argv[2],argv[3]);
    if(argc==3&&!strcmp(argv[1],"--nr3")){nr3_checks(argv[0],argv[2]);wex4_checks(argv[0],argv[2]);return fails?1:0;}
    printf("the MXFP4 matmul (quant.h matmul_mxfp4 through q38_weight_matmul)\n");
    kernel_case(640,2560,1);      /* gate/up of the release */
    kernel_case(640,2560,5);
    kernel_case(2560,640,1);      /* down */
    kernel_case(2560,640,9);
    kernel_case(32,32,3);         /* the tiny fixture */
    kernel_case(32,8,3);          /* a partial group: the scalar path */
    Cfg rel={0};rel.hidden=2560;rel.inter=640;
    int64_t so[3],wo[3],raw=0,aligned=q38_mx4_layout(&rel,so,wo,&raw);
    ck(raw==2611200&&aligned==2613248&&so[0]==0&&so[1]==640*80&&so[2]==640*80+2560*20&&
       wo[0]==3*51200&&wo[1]==wo[0]+640*1280&&wo[2]==wo[1]+2560*320,
       "release record: scales w1 gate, w2 down, w3 up, then nibbles; 2,611,200 bytes, 2,613,248 aligned");
    stripe_checks();
    if(argc>1){fixture_checks(argv[1]);refusals(argv[0],argv[1]);wex4_checks(argv[0],argv[1]);}
    else printf("(no snapshot on argv: the record checks run from make qwen38-tiny-mxfp4-check)\n");
    if(fails){printf("test_qwen38_mxfp4: %d failure(s)\n",fails);return 1;}
    printf("test_qwen38_mxfp4: ok\n");
    return 0;
}

/* SPDX-License-Identifier: Apache-2.0
 * Copyright 2026 Ludovico Zanichelli */
#define main unused_merged_test_main
#include "test_deepseek_v4.c"
#undef main

int coli_v4_test_gpu_frequency_victim(const uint64_t *, const uint64_t *, int, uint64_t);
int coli_v4_test_gpu_seal(uint64_t *, int, int);
int coli_v4_test_gpu_pool_slot(const uint64_t *, const uint64_t *, const uint32_t *, const int *,
                               const int *, int *, int, uint32_t, int *);
void coli_v4_test_gpu_configure_host_pin(void);
int coli_v4_test_mirror_evict(const uint64_t *, const uint64_t *, const int *, const int *,
                              const int *, int, int, long long, long long, int *, int *, int *,
                              int *);

/* V41_GPU_MOE's room for the prefill bank: the ring and empty slots go first, then the
 * coldest residents (frequency, then clock) until the bytes are back; a pool entry stays;
 * the survivors keep their order packed to the front and the tail is cleared; a busy
 * entry refuses it all. */
static void test_bank_evict(void) {
    uint64_t frequency[8]={5,1,0,9,1,7,3,3}, clock[8]={1,2,3,4,8,6,7,8};
    int busy[8]={0}, pool[8]={0,0,0,1,0,0,0,0}, loaded[8]={1,1,0,1,1,1,1,1};
    int keys[8], resident, freed, gone[24];
    /* ring 6,7 (20) and the empty 2 go, then 1 (f1 c2) and 4 (f1 c8): 40 >= 35; not 0, 5 */
    assert(coli_v4_test_mirror_evict(frequency,clock,busy,pool,loaded,8,6,35,10,keys,&resident,&freed,gone)==4);
    int want[8]={0,3,5,-1,-1,-1,-1,-1};
    for (int i=0;i<8;i++) assert(keys[i]==want[i]);
    assert(resident==3 && freed==12);
    /* exactly the dropped entries' tensors, in entry order (2 had none) */
    int gone_want[12]={1,1,1,4,4,4,6,6,6,7,7,7};
    for (int i=0;i<12;i++) assert(gone[i]==gone_want[i]);
    /* nothing more needed: the ring and the empty slot only */
    assert(coli_v4_test_mirror_evict(frequency,clock,busy,pool,loaded,8,6,0,10,keys,&resident,&freed,gone)==2);
    int ring_only[8]={0,1,3,4,5,-1,-1,-1};
    for (int i=0;i<8;i++) assert(keys[i]==ring_only[i]);
    assert(resident==5 && freed==6);
    for (int i=0;i<6;i++) assert(gone[i]==(i<3?6:7));
    /* everything asked: all but the pool entry */
    assert(coli_v4_test_mirror_evict(frequency,clock,busy,pool,loaded,8,0,1LL<<40,10,keys,&resident,&freed,gone)==6);
    assert(keys[0]==3 && keys[1]==-1 && keys[7]==-1 && resident==0 && freed==18);
    int all_want[6]={0,1,4,5,6,7};   /* never the pool entry 3 */
    for (int i=0;i<18;i++) assert(gone[i]==all_want[i/3]);
    /* an entry being loaded: refused, the cache unchanged */
    busy[5]=1;
    assert(coli_v4_test_mirror_evict(frequency,clock,busy,pool,loaded,8,6,35,10,keys,&resident,&freed,gone)==-1);
    for (int i=0;i<8;i++) assert(keys[i]==i);
    assert(resident==6 && freed==0);
}

/* WEX direct prefetch pool: grows over the coldest idle resident mirror, then
 * reuses its own least recently used entry; never busy, never one handed out in
 * the current epoch, never an entry without tensors. */
static void test_prefetch_pool(void) {
    setenv("V4_GPU_HOT_CACHE","1",1); setenv("COLI_WEX_GPU_DIRECT_POOL","2",1);
    uint64_t frequency[8]={9,1,3,2,8,7,6,5}, clock[8]={1,2,3,4,5,6,10,20};
    uint32_t epoch[8]={4,5,4,4,4,4,4,4};
    int busy[8]={0,0,0,1,0,0,0,0}, loaded[8]={1,1,0,1,1,1,1,1}, pool[8]={0}, count=0;
    /* 1 is in use, 3 busy, 2 empty: the coldest idle ones join the pool */
    assert(coli_v4_test_gpu_pool_slot(frequency,clock,epoch,busy,loaded,pool,8,5,&count)==7);
    assert(pool[7] && count==1);
    assert(coli_v4_test_gpu_pool_slot(frequency,clock,epoch,busy,loaded,pool,8,5,&count)==6);
    assert(pool[6] && count==2);
    /* full: least recently used pool entry, unless handed out this epoch */
    assert(coli_v4_test_gpu_pool_slot(frequency,clock,epoch,busy,loaded,pool,8,5,&count)==6);
    epoch[6]=5;
    assert(coli_v4_test_gpu_pool_slot(frequency,clock,epoch,busy,loaded,pool,8,5,&count)==7);
    epoch[7]=5;
    assert(coli_v4_test_gpu_pool_slot(frequency,clock,epoch,busy,loaded,pool,8,5,&count)==-1);
    /* an entry emptied by a failed load leaves the pool, which regrows */
    epoch[7]=4; loaded[7]=0;
    assert(coli_v4_test_gpu_pool_slot(frequency,clock,epoch,busy,loaded,pool,8,5,&count)==-1);
    assert(!pool[7] && count==1);
    assert(coli_v4_test_gpu_pool_slot(frequency,clock,epoch,busy,loaded,pool,8,5,&count)==5);
    assert(pool[5] && count==2);
    /* the host path's victim never comes from the pool: 0 is not pooled */
    assert(!pool[0] && !pool[1] && !pool[4]);
    unsetenv("COLI_WEX_GPU_DIRECT_POOL"); unsetenv("V4_GPU_HOT_CACHE");
}

int main(void) {
    setenv("COLI_V4_ELASTIC_RAM","1",1); setenv("DSV4_CUDA_PIN_HOST","1",1);
    coli_v4_test_gpu_configure_host_pin();
    assert(!strcmp(getenv("DSV4_CUDA_PIN_HOST"),"0"));
    setenv("COLI_V4_ELASTIC_RAM","0",1); setenv("DSV4_CUDA_PIN_HOST","1",1);
    coli_v4_test_gpu_configure_host_pin();
    assert(!strcmp(getenv("DSV4_CUDA_PIN_HOST"),"1"));
    unsetenv("DSV4_CUDA_PIN_HOST");
    uint64_t frequency[64], clock[64];
    for (int i=0;i<64;i++) { frequency[i]=100; clock[i]=i+1; }
    frequency[0]=50000; frequency[12]=1; clock[0]=0;
    /* An old, heavily used expert survives a newer, rarely used one. */
    assert(coli_v4_test_gpu_frequency_victim(frequency,clock,64,100)==12);
    frequency[12]=100; frequency[4]=2; frequency[5]=2;
    assert(coli_v4_test_gpu_frequency_victim(frequency,clock,64,100)==4);
    /* Never recycle a view just attached by the current forward. */
    frequency[63]=0; clock[63]=100;
    assert(coli_v4_test_gpu_frequency_victim(frequency,clock,64,100)==4);
    for (int i=0;i<8;i++) { frequency[i]=100; clock[i]=i+1; }
    frequency[6]=1;
    /* A nearly full eight-entry cache must retain the current layer's
     * newly attached views even if one has a very low frequency. */
    assert(coli_v4_test_gpu_frequency_victim(frequency,clock,8,10)==0);
    test_prefetch_pool();
    test_bank_evict();
    /* tier seal: residents keep their prewarm score unless live frequency zeroes it;
     * then all-zero residents fall back to recency (the oldest is the victim) */
    uint64_t sealed[4]={7,9,3,5}, sealed_clock[4]={4,1,3,2};
    assert(coli_v4_test_gpu_seal(sealed,4,0)==4 && sealed[0]==7 && sealed[2]==3);
    assert(coli_v4_test_gpu_seal(sealed,4,1)==4);
    for (int i=0;i<4;i++) assert(sealed[i]==0);
    assert(coli_v4_test_gpu_frequency_victim(sealed,sealed_clock,4,100)==1);

    char directory[]="colibri-v4-gpu-priority-XXXXXX", path[512], error[256];
    assert(mkdtemp(directory)); snprintf(path,sizeof(path),"%s/model.safetensors",directory);
    assert(!write_fixture_experts(path,8));
    setenv("COLI_V4_AUTOPIN","0",1);setenv("USAGE_SAVE","0",1);
    setenv("COLI_V4_PREWARM","0",1);setenv("COLI_V4_ROWS16","0",1);
    ColiDeepSeekV4ExpertStoreOptions options={directory,1,8,8*51,-1,UINT64_MAX,0,0};
    ColiExpertStore *store=NULL;
    assert(!coli_deepseek_v4_expert_store_open(&options,&store,error,sizeof(error)));
    ColiExpertKey key={0,3};
    assert(coli_v4_expert_store_frequency(store,key)==0);
    coli_v4_expert_store_note_routed(store,key);
    coli_v4_expert_store_note_routed(store,key);
    assert(coli_v4_expert_store_frequency(store,key)==2);
    ColiV4TestCreditSnapshot snap = {0};
    assert(!coli_v4_test_credit_snapshot(store,key,&snap));
    assert(snap.usage==2 && snap.heat==2 && snap.hit==1 && snap.layer_requests==2);
    ColiExpertStoreStats stats;store->ops->stats(store,&stats);
    assert(stats.requests==0 && stats.misses==0 && stats.bytes_read==0);
    store->ops->destroy(store);
    /* COLI_V4_LIVE_FREQUENCY=1: the promotion score ignores the seeded history.
     * Write a 3-selection history, then reopen with it seeded. */
    setenv("USAGE_SAVE","1",1);
    assert(!coli_deepseek_v4_expert_store_open(&options,&store,error,sizeof(error)));
    for (int i=0;i<3;i++) coli_v4_expert_store_note_routed(store,key);
    store->ops->destroy(store);
    setenv("USAGE_SAVE","0",1); unsetenv("COLI_V4_AUTOPIN"); setenv("COLI_V4_LIVE_FREQUENCY","1",1);
    assert(!coli_deepseek_v4_expert_store_open(&options,&store,error,sizeof(error)));
    assert(!coli_v4_test_credit_snapshot(store,key,&snap) && snap.history==3);
    assert(coli_v4_expert_store_frequency(store,key)==0);
    assert(coli_v4_expert_store_frequency_pending(store,key)==1);   /* WEX direct: credited after */
    coli_v4_expert_store_note_routed(store,key);
    assert(coli_v4_expert_store_frequency(store,key)==1);
    assert(!coli_v4_test_credit_snapshot(store,key,&snap) && snap.history==4);
    /* a credited host lookup (lookup_hot) counts too */
    ColiExpertView view={0};
    assert(!store->ops->lookup(store,key,&view));
    store->ops->release(store,&view);
    assert(coli_v4_expert_store_frequency(store,key)==2);
    store->ops->destroy(store);
    unsetenv("COLI_V4_LIVE_FREQUENCY"); setenv("COLI_V4_AUTOPIN","0",1);
    /* live off: the pending score is the history's, unchanged */
    assert(!coli_deepseek_v4_expert_store_open(&options,&store,error,sizeof(error)));
    assert(coli_v4_expert_store_frequency_pending(store,key)==coli_v4_expert_store_frequency(store,key));
    store->ops->destroy(store);
    char usage[600]; snprintf(usage,sizeof(usage),"%s/.coli_usage",directory);
    assert(!unlink(usage));
    assert(!scratch_remove(directory,path));
    puts("v4 GPU priority: frequency beats recency, live views protected, prefetch pool bounded, bank room evicts coldest packed, CUDA history counted without RAM reads: ok");
    return 0;
}

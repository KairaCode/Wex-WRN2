/* SPDX-License-Identifier: Apache-2.0
 * Copyright 2026 Ludovico Zanichelli */
/* Windows-only bounded GPU staging probe; does not start/deploy a model.
 * bench_wex_cuda RAW1.wex WEX2.wex QUANTIZED_RAW1.wex WEX3.wex [iterations]
 */
#define NOMINMAX
#include <windows.h>
#ifndef WEX_CUDA_HEADER
#define WEX_CUDA_HEADER "../wex_cuda.cuh"
#endif
#include WEX_CUDA_HEADER
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <algorithm>

#define CHECK(x) do { if (!(x)) { std::fprintf(stderr,"FAIL line %d: %s\n",__LINE__,#x); std::exit(1); } } while(0)
#define CUDA(x) do { cudaError_t status=(x); if(status!=cudaSuccess) { std::fprintf(stderr,"CUDA line %d: %s\n",__LINE__,cudaGetErrorString(status));std::exit(1); } } while(0)

struct Container { HANDLE file; WexHeader header; WexEntry entries[2]; };
static double seconds() {
    LARGE_INTEGER now, frequency; QueryPerformanceCounter(&now); QueryPerformanceFrequency(&frequency);
    return double(now.QuadPart)/double(frequency.QuadPart);
}
static Container open_container(const char *path) {
    Container out={};
    FILE *f=std::fopen(path,"rb"); CHECK(f);
    CHECK(!_fseeki64(f,0,SEEK_END)); auto bytes=_ftelli64(f); CHECK(bytes>=0);
    CHECK(!_fseeki64(f,0,SEEK_SET)); CHECK(std::fread(&out.header,sizeof(out.header),1,f)==1);
    CHECK(!wex_validate_header(&out.header,(uint64_t)bytes,1,2,NULL,NULL));
    CHECK(!_fseeki64(f,(__int64)out.header.table_offset,SEEK_SET));
    CHECK(std::fread(out.entries,sizeof(out.entries),1,f)==1); CHECK(!std::fclose(f));
    CHECK(!wex_validate_table(&out.header,out.entries,(uint64_t)bytes));
    out.file=CreateFileA(path,GENERIC_READ,FILE_SHARE_READ,NULL,OPEN_EXISTING,
                         FILE_FLAG_NO_BUFFERING|FILE_FLAG_RANDOM_ACCESS,NULL);
    CHECK(out.file!=INVALID_HANDLE_VALUE);
    return out;
}
static void read_record(const Container &c,int expert,unsigned char *dst) {
    LARGE_INTEGER position; position.QuadPart=(LONGLONG)c.entries[expert].file_offset;
    CHECK(WEX_IS_ALIGNED(dst,WEX_ALIGN)); CHECK(SetFilePointerEx(c.file,position,NULL,FILE_BEGIN));
    DWORD bytes=0; CHECK(ReadFile(c.file,dst,c.entries[expert].total_bytes,&bytes,NULL));
    CHECK(bytes==c.entries[expert].total_bytes);
}
static int prepare(const WexEntry *entry, const unsigned char *record, size_t capacity,
                   uint32_t *offsets, size_t tiles, uint32_t *parameter) {
    return entry->flags == WEX_ENTRY_NEURAL_RANS ?
        wex_cuda_prepare_neural_rans(entry,record,capacity,offsets,tiles,parameter) :
        wex_cuda_prepare_palette3(entry,record,capacity,offsets,tiles,parameter);
}
static cudaError_t launch(const WexEntry *entry, const unsigned char *host_record, uint32_t parameter,
                         const unsigned char *record, const uint32_t *offsets, unsigned char *raw,
                         uint32_t *lut, uint32_t *error, cudaStream_t stream) {
    if (entry->flags == WEX_ENTRY_NEURAL_RANS) {
        WexNeuralRans v={};
        v.classes=wex_read_u32_le(host_record+12u); v.streams=wex_read_u32_le(host_record+16u);
        v.scale_encoded=wex_read_u32_le(host_record+4u); v.metadata_bytes=parameter;
        v.chunk_symbols=wex_read_u32_le(host_record+20u);
        v.scale_only=wex_read_u32_le(host_record+24u);
        return wex_cuda_launch_neural_rans(entry,&v,record,offsets,raw,lut,error,stream);
    }
    return wex_cuda_launch_palette3(entry,parameter,record,offsets,raw,stream);
}

int main(int argc,char **argv) {
    if(argc<5 || argc>6) {std::fprintf(stderr,"Usage: %s RAW1 WEX2 QUANTIZED_RAW1 WEX3 [iterations]\n",argv[0]);return 1;}
    int iterations=argc==6?std::atoi(argv[5]):20; CHECK(iterations>0 && iterations<=10000);
    Container original=open_container(argv[1]), v2=open_container(argv[2]);
    Container reference=open_container(argv[3]), v3=open_container(argv[4]);
    CHECK(original.header.version==1 && reference.header.version==1 && v3.header.version==3);
    size_t capacity=(size_t)original.header.aligned_expert_bytes;
    CHECK(reference.header.aligned_expert_bytes==capacity && v3.header.aligned_expert_bytes==capacity);
    size_t tiles=(v3.entries[0].scale_bytes+255u)/256u;
    for(int e=0;e<2;e++) CHECK((v3.entries[e].flags==WEX_ENTRY_PALETTE3 || v3.entries[e].flags==WEX_ENTRY_NEURAL_RANS) &&
                              v3.entries[e].scale_bytes==v3.entries[0].scale_bytes);
    unsigned char *host_raw,*host_record,*device_raw,*device_raw_base,*device_record;
    uint32_t *host_offsets,*device_offsets,*host_error,*device_error,*device_lut;
    CUDA(cudaSetDevice(0));
    cudaDeviceProp properties; CUDA(cudaGetDeviceProperties(&properties,0));
    size_t free_before,total_device; CUDA(cudaMemGetInfo(&free_before,&total_device));
    CUDA(cudaMallocHost((void**)&host_raw,capacity));
    CUDA(cudaMallocHost((void**)&host_record,capacity));
    CUDA(cudaMallocHost((void**)&host_offsets,tiles*sizeof(uint32_t)));
    CUDA(cudaMalloc((void**)&device_raw_base,capacity+32u)); device_raw=device_raw_base+16u;
    CUDA(cudaMalloc((void**)&device_record,capacity));
    CUDA(cudaMalloc((void**)&device_offsets,tiles*sizeof(uint32_t)));
    CUDA(cudaMallocHost((void**)&host_error,4u)); CUDA(cudaMalloc((void**)&device_error,4u));
    CUDA(cudaMemset(device_error,0,4u));
    size_t lut_bytes=0;
    for(int e=0;e<2;e++) if(v3.entries[e].flags==WEX_ENTRY_NEURAL_RANS) {
        read_record(v3,e,host_record);
        WexNeuralRans view;
        CHECK(!wex_nr_parse(host_record,v3.entries[e].total_bytes,v3.entries[e].weight_bytes,&view));
        size_t bytes=(size_t)view.classes*(view.scale_only?1u:17u)*WEX_CUDA_NR_LUT_SLOTS*4u;
        lut_bytes=std::max(lut_bytes,bytes);
    }
    CUDA(cudaMalloc((void**)&device_lut,lut_bytes?lut_bytes:4u));
    cudaStream_t stream; CUDA(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));
    cudaEvent_t begin,end; CUDA(cudaEventCreate(&begin)); CUDA(cudaEventCreate(&end));
    std::vector<unsigned char> expected(capacity),actual(capacity);
    std::vector<float> kernel_ms;
    for(int e=0;e<2;e++) {
        read_record(reference,e,host_raw);
        std::memcpy(expected.data(),host_raw,capacity);
        if(v3.entries[e].flags==WEX_ENTRY_NEURAL_RANS) {
            read_record(original,e,host_record);
            CHECK(!std::memcmp(expected.data(),host_record,v3.entries[e].scale_bytes+v3.entries[e].weight_bytes));
        }
        read_record(v3,e,host_record);
        uint32_t scale_stream=0;
        CHECK(!prepare(&v3.entries[e],host_record,capacity,host_offsets,tiles,&scale_stream));
        // Invalid input must be refused before any launch.
        host_record[0]^=1;
        CHECK(prepare(&v3.entries[e],host_record,capacity,host_offsets,tiles,&scale_stream));
        host_record[0]^=1;
        CUDA(cudaMemcpyAsync(device_record,host_record,v3.entries[e].total_bytes,cudaMemcpyHostToDevice,stream));
        CUDA(cudaMemcpyAsync(device_offsets,host_offsets,tiles*sizeof(uint32_t),cudaMemcpyHostToDevice,stream));
        CUDA(cudaMemsetAsync(device_raw_base,0xa5,capacity+32u,stream));
        int warmups=v3.entries[e].flags==WEX_ENTRY_NEURAL_RANS?256:20;
        for(int repeat=0;repeat<warmups;repeat++) {
            CUDA(cudaEventRecord(begin,stream));
            CUDA(launch(&v3.entries[e],host_record,scale_stream,device_record,device_offsets,device_raw,device_lut,device_error,stream));
            CUDA(cudaEventRecord(end,stream)); CUDA(cudaEventSynchronize(end));
            float ms; CUDA(cudaEventElapsedTime(&ms,begin,end));
            if(repeat>=warmups-(warmups==256?32:15)) kernel_ms.push_back(ms);
        }
        CUDA(cudaMemcpyAsync(actual.data(),device_raw,capacity,cudaMemcpyDeviceToHost,stream));
        CUDA(cudaMemcpyAsync(host_error,device_error,4u,cudaMemcpyDeviceToHost,stream));
        CUDA(cudaStreamSynchronize(stream));
        CHECK(!*host_error);
        size_t raw_bytes=(size_t)v3.entries[e].scale_bytes+v3.entries[e].weight_bytes;
        CHECK(!std::memcmp(expected.data(),actual.data(),raw_bytes));
        unsigned char guards[32];
        CUDA(cudaMemcpy(guards,device_raw_base,16u,cudaMemcpyDeviceToHost));
        CUDA(cudaMemcpy(guards+16,device_raw+capacity,16u,cudaMemcpyDeviceToHost));
        for(unsigned int g=0;g<32;g++) CHECK(guards[g]==0xa5);
        if(v3.entries[e].flags==WEX_ENTRY_NEURAL_RANS) {
            WexNeuralRans view;
            CHECK(!wex_nr_parse(host_record,v3.entries[e].total_bytes,v3.entries[e].weight_bytes,&view));
            CUDA(cudaMemsetAsync(device_record+view.metadata_bytes+view.scale_encoded,0,4u,stream));
            CUDA(launch(&v3.entries[e],host_record,scale_stream,device_record,device_offsets,device_raw,device_lut,device_error,stream));
            CUDA(cudaMemcpyAsync(host_error,device_error,4u,cudaMemcpyDeviceToHost,stream));
            CUDA(cudaStreamSynchronize(stream)); CHECK(*host_error);
        }
    }
    std::sort(kernel_ms.begin(),kernel_ms.end());
    std::printf("{\"gpu\":\"%s\",\"gpu_free_before_bytes\":%llu,\"device_workspace_bytes\":%llu,"
                "\"kernel_median_us\":%.3f,\"exact_quantized_reference\":true,\"exact_original_reference\":%s,\"iterations_per_record\":%d,\"trials\":[",
                properties.name,(unsigned long long)free_before,(unsigned long long)(capacity*2+32u+tiles*4+4u+(lut_bytes?lut_bytes:4u)),
                kernel_ms[kernel_ms.size()/2]*1000,v3.entries[0].flags==WEX_ENTRY_NEURAL_RANS?"true":"false",iterations);
    const int order[]={0,1,1,0};
    for(int trial=0;trial<4;trial++) {
        int gpu=order[trial];
        double start=seconds();
        for(int repeat=0;repeat<iterations;repeat++) for(int e=0;e<2;e++) {
            if(gpu) {
                uint32_t scale_stream=0;
                read_record(v3,e,host_record);
                CHECK(!prepare(&v3.entries[e],host_record,capacity,host_offsets,tiles,&scale_stream));
                CUDA(cudaMemcpyAsync(device_record,host_record,v3.entries[e].total_bytes,cudaMemcpyHostToDevice,stream));
                CUDA(cudaMemcpyAsync(device_offsets,host_offsets,tiles*sizeof(uint32_t),cudaMemcpyHostToDevice,stream));
                CUDA(launch(&v3.entries[e],host_record,scale_stream,device_record,device_offsets,device_raw,device_lut,device_error,stream));
                CUDA(cudaMemcpyAsync(host_error,device_error,4u,cudaMemcpyDeviceToHost,stream));
            } else {
                read_record(v2,e,host_raw+wex_record_read_offset(&v2.entries[e]));
                CHECK(!wex_decode_record(&v2.entries[e],host_raw,capacity));
                CUDA(cudaMemcpyAsync(device_raw,host_raw,capacity,cudaMemcpyHostToDevice,stream));
            }
            CUDA(cudaStreamSynchronize(stream));
            if(gpu) CHECK(!*host_error);
        }
        double us=(seconds()-start)*1e6/(iterations*2);
        std::printf("%s{\"path\":\"%s\",\"load_to_vram_us\":%.3f}",trial?",":"",gpu?"gpu_wex3":"cpu_wex2",us);
        // Compare outside timing after each trial, including baseline original.
        read_record(gpu?reference:original,1,host_raw);
        CUDA(cudaMemcpyAsync(actual.data(),device_raw,capacity,cudaMemcpyDeviceToHost,stream));
        CUDA(cudaStreamSynchronize(stream));
        CHECK(!std::memcmp(host_raw,actual.data(),v3.entries[1].scale_bytes+v3.entries[1].weight_bytes));
    }
    std::puts("],\"scope\":\"ABBA synchronous unbuffered read + preparation + pinned H2D + reconstruction; no inference or compute/decode overlap\"}");
    CUDA(cudaEventDestroy(begin));CUDA(cudaEventDestroy(end));CUDA(cudaStreamDestroy(stream));
    CUDA(cudaFree(device_raw_base));CUDA(cudaFree(device_record));CUDA(cudaFree(device_offsets));
    CUDA(cudaFree(device_error));CUDA(cudaFreeHost(host_error));
    CUDA(cudaFree(device_lut));
    CUDA(cudaFreeHost(host_raw));CUDA(cudaFreeHost(host_record));CUDA(cudaFreeHost(host_offsets));
    CHECK(CloseHandle(original.file));CHECK(CloseHandle(v2.file));CHECK(CloseHandle(reference.file));CHECK(CloseHandle(v3.file));
    return 0;
}

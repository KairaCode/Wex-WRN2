#define _GNU_SOURCE
#include <linux/perf_event.h>
#include <sys/syscall.h>
#include <unistd.h>
#include <stdio.h>
#include <string.h>
#include <stdint.h>
int main(void){ struct perf_event_attr a; memset(&a,0,sizeof a); a.size=sizeof a; a.type=PERF_TYPE_HARDWARE; a.config=PERF_COUNT_HW_CPU_CYCLES; a.exclude_kernel=1;
 int fd=syscall(SYS_perf_event_open,&a,0,-1,-1,0); printf("fd=%d\n",fd); if(fd<0){perror("open");return 1;}
 volatile double x=0; for(int i=0;i<10000000;i++) x+=i; uint64_t c; read(fd,&c,8); printf("cycles %llu\n",(unsigned long long)c); return 0;}

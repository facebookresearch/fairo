/**
 * Note: Parts taken from
 * https://wiki.linuxfoundation.org/realtime/documentation/howto/applications/application_base
 *
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#ifndef REAL_TIME_H
#define REAL_TIME_H

#include <limits.h>
#include <malloc.h>
#include <pthread.h>
#include <sched.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/mman.h>
#include <sys/resource.h>

#include <fcntl.h>
#include <unistd.h>

int write_cpu_dma_latency(int max_latency_microseconds) {
  int fd = open("/dev/cpu_dma_latency", O_WRONLY);
  if (fd < 0) {
    printf("failed to open /dev/cpu_dma_latency\n");
    return 1;
  }

  if (write(fd, &max_latency_microseconds, sizeof(max_latency_microseconds)) !=
      sizeof(max_latency_microseconds)) {
    printf("failed to write to /dev/cpu_dma_latency\n");
    return 2;
  }

  return 0;
}

int create_real_time_thread(void *(*start_routine)(void *), void *arg = NULL) {
  struct sched_param param;
  pthread_attr_t attr;
  pthread_t thread;
  int ret;

  int stack_size_mb = 20;
  size_t stack_size = 1024 * 1024 * stack_size_mb;
  printf("Using %dMB as stack size\n", stack_size_mb);

  int sched_priority = 80;
  printf("Using %d as real-time thread priority\n", sched_priority);

  int cpu_dma_latency = 0;
  printf("Using %d as cpu_dma_latency\n", cpu_dma_latency);

  /* Disable sbrk */
  ret = mallopt(M_TRIM_THRESHOLD, 0);
  if (ret != 1) {
    printf("failed to disable sbrk\n");
    goto out;
  }
  printf("Disabled sbrk...\n");

  /* Disable mmap */
  ret = mallopt(M_MMAP_MAX, 1);
  if (ret != 1) {
    printf("failed to disable mmap\n");
    goto out;
  }
  printf("Disabled mmap...\n");

  write_cpu_dma_latency(cpu_dma_latency);

  /* Lock memory */
  if (mlockall(MCL_CURRENT | MCL_FUTURE)) {
    printf("mlockall failed: %m\n");
    goto out;
  }
  printf("Locked memory...\n");

  /* Initialize pthread attributes (default values) */
  ret = pthread_attr_init(&attr);
  if (ret) {
    printf("init pthread attributes failed\n");
    goto out;
  }
  printf("Initialized pthread...\n");

  /* Set a specific stack size  */
  ret = pthread_attr_setstacksize(&attr, stack_size);
  if (ret) {
    printf("pthread setstacksize failed with error code %d\n", ret);
    goto out;
  }
  printf("Finished setting stacksize...\n");

  /* Set scheduler policy and priority of pthread */
  ret = pthread_attr_setschedpolicy(&attr, SCHED_FIFO);
  if (ret) {
    printf("pthread setschedpolicy failed\n");
    goto out;
  }
  param.sched_priority = sched_priority;
  ret = pthread_attr_setschedparam(&attr, &param);
  if (ret) {
    printf("pthread setschedparam failed\n");
    goto out;
  }
  printf("Finished setting scheduling policy & priority...\n");

  /* Use scheduling parameters of attr */
  ret = pthread_attr_setinheritsched(&attr, PTHREAD_EXPLICIT_SCHED);
  if (ret) {
    printf("pthread setinheritsched failed\n");
    goto out;
  }

  printf("Creating thread...\n");
  /* Create a pthread with specified attributes */
  ret = pthread_create(&thread, &attr, start_routine, arg);
  if (ret) {
    printf("create pthread failed\n");
    goto out;
  }
  printf("Started realtime thread.\n");

  /* Join the thread and wait until it is done */
  ret = pthread_join(thread, NULL);
  if (ret)
    printf("join pthread failed: %m\n");

out:
  if (ret) {
    printf("starting thread in non-realtime.\n");
    start_routine(arg);
  }
  return ret;
}

#endif
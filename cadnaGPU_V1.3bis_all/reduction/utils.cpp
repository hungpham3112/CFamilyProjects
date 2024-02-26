unsigned int nextPow2(unsigned int x) {
  --x;
  x |= x >> 1;
  x |= x >> 2;
  x |= x >> 4;
  x |= x >> 8;
  x |= x >> 16;
  return ++x;
}

void getNumBlocksAndThreads(int n, int maxBlocks, int maxThreads, int &blocks,
                            int &threads) {
  // get device capability, to avoid block/grid size exceed the upper bound
  cudaDeviceProp prop;
  int device;
  cudaGetDevice(&device);
  cudaGetDeviceProperties(&prop, device);

  threads = (n < maxThreads) ? nextPow2(n) : maxThreads;
  blocks = (n + threads - 1) / threads;

  if (threads * blocks > (unsigned int)prop.maxGridSize[0] *
                             (unsigned int)prop.maxThreadsPerBlock) {
    printf("threads: %d, blocks: %d, propmaxgrid: %d, propmaxthread: %d",
           threads, blocks, prop.maxGridSize[0], prop.maxThreadsPerBlock);
    printf("n is too large, please choose a smaller number!\n");
  }

  if (blocks > prop.maxGridSize[0]) {
    printf(
        "Grid size <%d> exceeds the device capability <%d>, set block size as "
        "%d (original %d)\n",
        blocks, prop.maxGridSize[0], threads * 2, threads);

    blocks /= 2;
    threads *= 2;
  }
}

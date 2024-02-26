#include "reduction.cu"
#include <iostream>
#include <fstream>
#include <random>
#include <bitset>
#include <cstring>

template <class T>
void generate_random_array(T* array, const int& size, const int& seed)
{
    std::mt19937 engine(seed);
    std::normal_distribution<T> generator(0, 1);
    for (int i = 0; i < size; i++)
        array[i] = generator(engine);
}

int get_sizes(int kernel, int size, int& gsize, int& lsize)
{
    int blocks;
    if (kernel < 3)
    {
        gsize = (size + lsize - 1) / lsize * lsize;
        return gsize / lsize;
    }

    gsize = (size + 2 * lsize - 1) / (2 * lsize) *  lsize;
    return gsize /  lsize;
}

template <class T>
T reduce_cpu(T* a, int size)
{
    T sum = a[0], c = 0;
    for (int i = 1; i < size; i++){
        T y = a[i] - c;
        T t = sum + y;
        c = (t - sum) - y;
        sum = t;
    }
    return sum;
}




template <class T>
T benchmarkReduce(int  n,
                  int  numThreads,
                  int  numBlocks,
                  int  whichKernel,
                  int  testIterations,
                  bool cpuFinalReduction,
                  int  cpuFinalThreshold,
                  T *h_odata,
                  T *d_idata,
                  T *d_odata)
{
    T gpu_result = 0;
    bool needReadBack = true;

    for (int i = 0; i < testIterations; ++i)
    {
        gpu_result = 0;

        cudaDeviceSynchronize();

        // execute the kernel
        reduce<T>(n, numThreads, numBlocks, whichKernel, d_idata, d_odata);

        if (cpuFinalReduction)
        {
            // sum partial sums from each block on CPU
            // copy result from device to host
            cudaMemcpy(h_odata, d_odata, numBlocks*sizeof(T), cudaMemcpyDeviceToHost);

            for (int i=0; i<numBlocks; i++)
            {
                gpu_result += h_odata[i];
                printf("%f\n", h_odata[i]);
                
            }

            needReadBack = false;
        }
        else
        {
            // sum partial block sums on GPU
            int s=numBlocks;
            int kernel = whichKernel;

            while (s > cpuFinalThreshold)
            {
                int threads = numThreads, blocks = 0, totalThreads = 0;
                blocks = get_sizes(whichKernel, n, totalThreads, threads);

                reduce<T>(s, threads, blocks, kernel, d_odata, d_odata);

                if (kernel < 3)
                {
                    s = (s + threads - 1) / threads;
                }
                else
                {
                    s = (s + (threads*2-1)) / (threads*2);
                }
            }

            if (s > 1)
            {
                // copy result from device to host
                cudaMemcpy(h_odata, d_odata, s * sizeof(T), cudaMemcpyDeviceToHost);

                for (int i=0; i < s; i++)
                {
                    gpu_result += h_odata[i];
                }

                needReadBack = false;
            }
        }

        cudaDeviceSynchronize();
    }

    if (needReadBack)
    {
        // copy final sum from device to host
        cudaMemcpy(&gpu_result, d_odata, sizeof(T), cudaMemcpyDeviceToHost);
    }

    return gpu_result;
}


int run(int argc, char** argv)
{
    int kernel = 0;
    int size = 1 << 10;
    if (argc > 1) kernel = atoi(argv[1]);
    if (kernel > 6) {
        std::cout << "invalid kernel id (expected 0-6, got " << kernel << ")\n";
        return 1;
    }
    if (argc > 2) size = atoi(argv[2]);
    int seed = std::random_device()();


    const int size_c = size;
    float a_f[size_c]; 
    if (argc > 4){
        std::ifstream ifile(argv[4], std::ios::binary);
        ifile.read((char*)a_f, sizeof(float) * size_c);
        ifile.close();
    }
    else generate_random_array(a_f, size_c, seed);
    double a_d[size_c]; for (int i = 0; i < size_c; i++) a_d[i] = a_f[i];


    int local_size = 1024, global_size;
    if (argc > 3) local_size = (size_t)atoi(argv[3]);
    std::cout << "local size: " << local_size << std::endl;
    if (local_size == 1 && kernel < 3)
    {
        std::cout << "kernel " << kernel << " cannot have " << local_size << " as local size\n";
        return -1;
    }
    int num_blocks = get_sizes(kernel, size, global_size, local_size);
    if (kernel == 6 and argc > 5) {
        num_blocks = atoi(argv[5]);
        global_size = num_blocks * local_size;
    }
    int fp64_num_blocks = num_blocks;
    if (argc > 5) fp64_num_blocks = atoi(argv[5]);
    std::cout << "global: " << global_size << "\n";

    float* input_f, *output_f;
    double* input_d, *output_d;

    cudaMalloc(&input_f, sizeof(float) * size_c);
    cudaMalloc(&output_f, sizeof(float) * num_blocks);
    cudaMalloc(&input_d, sizeof(double) * size_c);
    cudaMalloc(&output_d, sizeof(double) * num_blocks);
    cudaMemcpy(input_f, a_f, sizeof(float) * size_c, cudaMemcpyHostToDevice);
    cudaMemcpy(input_d, a_d, sizeof(double) * size_c, cudaMemcpyHostToDevice);

    float reduce_f_r = benchmarkReduce<float>(size_c, local_size, num_blocks, kernel, 1, 
                        false, 1, a_f, input_f, output_f);
    double reduce_d_r = benchmarkReduce<double>(size_c, local_size, fp64_num_blocks, kernel, 1, 
                        true, 1, a_d, input_d, output_d);
    float reduce_f_cpu = reduce_cpu<float>(a_f, size_c);

    int reduce_f_r_toint = *(int*)(&reduce_f_r), reduce_f_cpu_toint = *(int*)(&reduce_f_cpu);
    uint64_t reduce_d_r_toint = *(uint64_t*)(&reduce_d_r);
    std::bitset<32> f_r(reduce_f_r_toint), f_cpu(reduce_f_cpu_toint);
    std::bitset<64> d_r(reduce_d_r_toint);


    std::cout << reduce_f_r << std::endl;
    std::cout << reduce_f_cpu << std::endl;
    std::cout << reduce_d_r << std::endl;

    std::string bit_string_f = f_r.to_string(), bit_string_cpu = f_cpu.to_string(), bit_string_d = d_r.to_string();
    auto exponent_f = (int)std::bitset<8>(bit_string_f.substr(1, 8)).to_ulong() - 127;
    auto mantissa_f = bit_string_f.substr(9, 23);

    auto exponent_d = (int)std::bitset<11>(bit_string_d.substr(1, 11)).to_ulong() - 1023;
    auto mantissa_d = bit_string_d.substr(12, 52);

    std::cout << "(single) exponent: " << exponent_f << "\tmantissa: " << mantissa_f << std::endl;
    std::cout << "(double) exponent: " << exponent_d << "\tmantissa: " << mantissa_d << std::endl;
    cudaFree(input_f);
    cudaFree(input_d);
    cudaFree(output_f);
    cudaFree(output_d);
    return 0;
}

int main(int argc, char**argv)
{
    std::cout << "reduction-cu [kernel (0-6), default 0] [inputSize, default 16777216] [localSize, default 256] [dataFile, default null] [kernel6NumBlocks, default null]\n";
    return run(argc, argv);
}

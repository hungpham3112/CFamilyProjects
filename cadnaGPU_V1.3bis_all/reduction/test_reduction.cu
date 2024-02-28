#include <stdio.h>
#include <cadna.h>
#include <cadna_gpu.cu>
#include <iostream>

int main(int argc, char *argv[])
{
    cadna_init(-1);
    
    double_st n = static_cast<double_st>(1.2312);
    // float_st m = static_cast<float_st>(0);
    float_st m = static_cast<float_st>(2.3);

    // std::cout << n << std::endl;
    std::cout << static_cast<double_st>(m) - n << std::endl;
    cadna_end();
    return 0;
}


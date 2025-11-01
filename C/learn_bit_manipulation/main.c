#include <errno.h>
#include <stdint.h>
#include <stdio.h>

// int set_alive_payload(uint8_t payload[8], uint8_t count, uint64_t start_bit, uint32_t length, int lower_bound,
//                       int higher_bound)
// {
//     if (count < lower_bound && count > higher_bound)
//     {
//         return 1;
//     }
//
//     11
//
//         uint8_t mask = (count << (11 % 8));
//     payload[1] |= mask
//
//         printf("%b", mask);
//     // payload |= mask;
// }
//
// int hex2dec(int)
// {
// }

int add_bitwise(int a, int b)
{
    while (b != 0)
    {
        int sum = a ^ b;
        int carry = (a & b) << 1;
        a = sum;
        b = carry;
    }
    return a;
}

int64_t multiply_by_2_pow_n(int64_t num, int n)
{
    return num << n;
}

int take_low_nibble(int64_t data)
{
    return data & 0x0F;
}

int is_on(int64_t num, uint8_t bit)
{
    return num == (num | (1LL << bit));
}

int turn_on(int64_t num, uint8_t bit)
{
    return num | (1LL << bit);
}

int turn_off(int64_t num, uint8_t bit)
{
    return num & ~(1LL << bit);
}

int flip_bit(int64_t num, uint8_t bit)
{
    return num ^ (1LL << bit);
}

int main()
{
    int a = 0x20;
    int turned_on_a = turn_on(a, 0);
    int turned_off_a = turn_off(a, 5);
    printf("%d\n", is_on(turned_on_a, 0));
    printf("%d\n", is_on(turned_off_a, 3));
    printf("%b\n", turned_on_a);
    printf("%b\n", turned_off_a);
    int flipped_bit_a = flip_bit(a, 2);
    printf("%b\n", flipped_bit_a);
    flipped_bit_a = flip_bit(a, 4);
    printf("%b\n", flipped_bit_a);
    printf("%d\n", is_on(flipped_bit_a, 4));

    int64_t num1 = 5;
    int64_t num2 = 3;
    printf("%d\n", add_bitwise(num1, num2));
    int data = 0x3000E;
    printf("%X", take_low_nibble(data));

    printf("%d\n", -12 >> 1);

    return 0;
}

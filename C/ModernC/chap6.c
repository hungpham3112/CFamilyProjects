#include <stdbool.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

char db_username[16];

size_t my_strlen(char str[])
{
    size_t count;
    for (count = 0; str[count] != '\0'; ++count)
    {
    }

    return count;
}

char *my_strncpy(char *dst, const char *src, size_t len)
{
    for (size_t i = 0; i < len; ++i)
    {
        dst[i] = src[i];
    }
    return dst;
}

char *my_strcpy(char *dst, const char *src)
{
    size_t i;
    for (i = 0; src[i] != '\0'; ++i)
    {
        dst[i] = src[i];
    }
    dst[i] = '\0';
    return dst;
}

bool store_username(char *username)
{
    size_t name_len = my_strlen(username);
    if (name_len >= 16)
    {
        my_strncpy(db_username, username, 15);
        db_username[15] = '\0';
        return 0;
    }
    my_strcpy(db_username, username);

    return true;
}

struct Point
{
    int x;
    int y;
};

void swap_double_arr(double arr[])
{
    double tmp = arr[0];
    arr[0] = arr[1];
    arr[1] = tmp;
}

void swap_coordinate(struct Point *p)
{
    int tmp = p->x;
    p->x = p->y;
    p->y = tmp;
}

struct BadOrder
{
    char a;
    int b;
    char c;
};

struct GoodOrder
{
    char a;
    char b;
    int c;
};

struct __attribute__((packed)) BadPacked
{
    char a;
    int b;
    char c;
};

struct __attribute__((packed)) GoodPacked
{
    char a;
    char b;
    int c;
};

int main(int argc, char *argv[])
{
    { // learn struct and array
        double arr[] = {[0] = 1, [1] = 2};
        printf("arr[0]: %lf, arr[1]: %lf\n", arr[0], arr[1]);
        swap_double_arr(arr);
        printf("arr[0]: %lf, arr[1]: %lf\n", arr[0], arr[1]);

        struct Point p = {.x = 1, .y = 2};
        printf("Point.x: %d, Point.y: %d\n", p.x, p.y);
        swap_coordinate(&p);
        printf("Point.x: %d, Point.y: %d\n", p.x, p.y);
    }

    { // practice string
        memset(db_username, 'X', 16);
        printf("Debug dirty: %.16s\n", db_username);
        char name[14] = "tututututututu";

        printf("name len: %zu\n", strlen(name));

        store_username(name);

        printf("db_username: %s, len: %zu\n", db_username, strlen(db_username));
        printf("db_username[14]: %c\n", db_username[14]);
        printf("db_username[15]: %c\n", db_username[15]);
    }

    { // learn data alignment in struct
        printf("=== Bad Order ===\n");
        printf("Size: %zu bytes\n", sizeof(struct BadOrder));
        printf("Offset a: %zu\n", offsetof(struct BadOrder, a));
        printf("Offset b: %zu (Padding: %zu bytes)\n", offsetof(struct BadOrder, b),
               offsetof(struct BadOrder, b) - sizeof(char));
        printf("Offset c: %zu\n", offsetof(struct BadOrder, c));

        printf("\n=== Good Order ===\n");
        printf("Size: %zu bytes (Not 6!)\n", sizeof(struct GoodOrder));
        printf("Offset a: %zu\n", offsetof(struct GoodOrder, a));
        printf("Offset b: %zu\n", offsetof(struct GoodOrder, b));
        printf("Offset c: %zu (Padding before int: %zu bytes)\n", offsetof(struct GoodOrder, c),
               offsetof(struct GoodOrder, c) - (offsetof(struct GoodOrder, b) + 1));

        printf("\n=== Forced Packed ===\n");
        printf("Size: %zu bytes (Performance Killer)\n", sizeof(struct BadPacked));

        printf("\n=== Forced Packed ===\n");
        printf("Size: %zu bytes (Performance Killer)\n", sizeof(struct GoodPacked));
    }
    // {
    //     // use after free
    //     int *p = (int *)malloc(sizeof(int));
    //     *p = 10;
    //     free(p);
    //     *p = 10;
    // }
    // {
    //     // Double free
    //     int *p = (int *)malloc(sizeof(int));
    //     *p = 10;
    //     free(p);
    //     free(p);
    // }
    {
        // Dangling pointer
        int *p = (int *)malloc(sizeof(int));
        *p = 100;
        printf("1. address p: %p | value %d\n", (void *)p, *p);

        free(p);
        // p = NULL; // them dong nay se tao ra segfault vi *p = 666 dang truy cap vao NULL
        printf("2. Đã free p1.\n");

        // 3. Cấp phát p2 ngay lập tức
        // Hệ điều hành thấy vùng nhớ cũ vừa trống, thường sẽ cấp lại NGAY CHỖ ĐÓ cho p2
        int *p2 = (int *)malloc(sizeof(int));
        *p2 = 999;
        printf("3. Địa chỉ p2: %p | Giá trị: %d\n", (void *)p2, *p2);

        // 4. THẢM HỌA: Dùng p1 (dangling) để phá hoại
        printf("4. Ghi đè vào *p1 (Use After Free)...\n");
        *p = 666; // <--- Đây là dòng bạn "nghịch"

        // 5. Kiểm tra lại p2
        printf("5. Giá trị của p2 bây giờ là: %d\n", *p2);
    }

    return 0;
}

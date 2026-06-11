#include <stdio.h>

typedef enum
{
    BIG = 123412341234,
    BIGGER,
    BIGGEST,
} BigNumbers;

typedef enum
{
    HTTP_BAD_REQUEST = 400,
    HTTP_UNAUTHORIZED = 401,
    HTTP_NOT_FOUND = 404,
    HTTP_I_AM_A_TEAPOT = 418,
    HTTP_INTERNAL_SERVER_ERROR = 500
} HttpErrorCode;

int main()
{
    printf("The size of BigNumbers is %zu bytes", sizeof(BigNumbers));
    printf("The size of HttpErrorCode is %zu bytes", sizeof(HttpErrorCode));

    return 0;
}

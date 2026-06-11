#include "http.h"
#include <stdio.h>

char *http_to_str(http_error_code_t code)
{
    switch (code)
    {
    case 400:
        return "400 Bad Request";
    case 401:
        return "401 Unauthorized";
    case 404:
        return "404 Not Found";
    case 418:
        return "418 I AM A TEAPOT!";
    case 500:
        return "500 Internal Server Error";
    default:
        return "Unknown HTTP status code";
    }
}

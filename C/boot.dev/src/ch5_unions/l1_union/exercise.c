#include "exercise.h"
#include <stdio.h>

void format_object(snek_object_t obj, char *buffer)
{
    switch (obj.kind)
    {
    case INTEGER:
        sprintf(buffer, "int:%d", obj.data.v_int);
        break;
    case STRING:
        sprintf(buffer, "string:%s", obj.data.v_string);
        break;
    }
}

// don't touch below this line

snek_object_t new_integer(int i)
{
    return (snek_object_t){
        .kind = INTEGER,
        .data = {.v_int = i},
    };
}

snek_object_t new_string(char *str)
{
    // NOTE: We will learn how to copy this data later.
    return (snek_object_t){
        .kind = STRING,
        .data = {.v_string = str},
    };
}

#include "exercise.h"
#include "munit.h"
#include <stdlib.h>

munit_case(RUN, test_allocate, {
    int *pointer = NULL;
    allocate_int(&pointer, 10);

    assert_ptr_not_null(pointer, "Should allocate pointer");
    assert_int(*pointer, ==, 10, "Should assign value to pointer");

    free(pointer);
});

munit_case(SUBMIT, test_does_not_overwrite, {
    int value = 5;
    int *pointer = &value;

    allocate_int(&pointer, 20);

    assert_int(value, ==, 5, "Should not overwrite original value");

    assert_ptr_not_null(pointer, "Should allocate pointer");
    assert_int(*pointer, ==, 20, "Should assign value to pointer");

    free(pointer);
});

int main()
{
    MunitTest tests[] = {
        munit_test("/create", test_allocate),
        munit_test("/overwrite", test_does_not_overwrite),
        munit_null_test,
    };
    MunitSuite suite = munit_suite("allocate_list", tests);

    return munit_suite_main(&suite, NULL, 0, NULL);
}

#include "exercise.h"
#include "munit.h"

munit_case(RUN, test_allocate_scalar_array_size, {
    int size = 5;
    int multiplier = 2;
    int *result = allocate_scalar_array(size, multiplier);
    munit_assert_not_null(result, "Function should return a non-null pointer");
    free(result);
});

munit_case(RUN, test_allocate_scalar_array_values, {
    int size = 5;
    int multiplier = 2;
    int *result = allocate_scalar_array(size, multiplier);
    int expected[5];
    expected[0] = 0;
    expected[1] = 2;
    expected[2] = 4;
    expected[3] = 6;
    expected[4] = 8;
    for (int i = 0; i < size; i++)
    {
        munit_assert_int(result[i], ==, expected[i], "Element does not match expected value");
    }
    free(result);
});

munit_case(SUBMIT, test_allocate_scalar_array_zero_multiplier, {
    int size = 3;
    int multiplier = 0;
    int *result = allocate_scalar_array(size, multiplier);
    for (int i = 0; i < size; i++)
    {
        munit_assert_int(result[i], ==, 0, "All elements should be 0 with multiplier 0");
    }
    free(result);
});

munit_case(SUBMIT, test_allocate_too_much, {
    int size = (64 * 1024 * 1024) / sizeof(int); // 64 MiB
    int multiplier = 1;
    int *result = allocate_scalar_array(size, multiplier);
    // It was originally intended for a large allocation to fail and return NULL
    // After a change to Emscripten compiler settings, the allocation may succeed
    // In that case, we check values to make sure it isn't garbage
    if (result != NULL)
    {
        munit_assert_int(result[0], ==, 0, "First element should be 0");
        munit_assert_int(result[size - 1], ==, size - 1, "Last element should match expected value");
        free(result);
    }
    munit_assert_int(1, ==, 1, "Function should handle large allocations without crashing");
});

int main()
{
    MunitTest tests[] = {
        munit_test("/test_allocate_scalar_array_size", test_allocate_scalar_array_size),
        munit_test("/test_allocate_scalar_array_values", test_allocate_scalar_array_values),
        munit_test("/test_allocate_scalar_array_zero_multiplier", test_allocate_scalar_array_zero_multiplier),
        munit_test("/test_allocate_too_much", test_allocate_too_much),
        munit_null_test,
    };

    MunitSuite suite = munit_suite("allocate_scalar_array", tests);

    return munit_suite_main(&suite, NULL, 0, NULL);
}

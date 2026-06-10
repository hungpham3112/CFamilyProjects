#include "color.h"
#include "munit.h"

munit_case(RUN, test_color_enum1, {
    assert_int(RED, ==, 0, "RED is defined as 0");
    assert_int(GREEN, ==, 1, "GREEN is defined as 1");
    assert_int(BLUE, ==, 2, "BLUE is defined as 2");
});

munit_case(SUBMIT, test_color_enum2, {
    assert_int(RED, !=, 4, "RED is not defined as 4");
    assert_int(GREEN, !=, 2, "GREEN is not defined as 2");
    assert_int(BLUE, !=, 0, "BLUE is not defined as 0");
});

int main()
{
    MunitTest tests[] = {
        munit_test("/are_defined", test_color_enum1),
        munit_test("/are_defined_correctly", test_color_enum2),
        munit_null_test,
    };

    MunitSuite suite = munit_suite("colors", tests);

    return munit_suite_main(&suite, NULL, 0, NULL);
}

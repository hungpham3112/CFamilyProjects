#include "color.h"
#include "munit.h"

munit_case(RUN, test_colors_defined, {
    assert_int(RED, ==, 55, "RED is defined as 55 (nvim green!)");
    assert_int(GREEN, ==, 176, "GREEN is defined as 176 (nvim green!)");
    assert_int(BLUE, ==, 38, "BLUE is defined as 38 (nvim green!)");
});

munit_case(SUBMIT, test_colors_defined_correctly, {
    assert_int(RED, !=, 0, "RED is not defined as 0 (vsc*de blue!)");
    assert_int(GREEN, !=, 120, "GREEN is not defined as 120 (vsc*de blue!)");
    assert_int(BLUE, !=, 215, "BLUE is not defined as 215 (vsc*de blue!)");
});

int main()
{
    MunitTest tests[] = {
        munit_test("/defined", test_colors_defined),
        munit_test("/defined_vscode", test_colors_defined_correctly),
        munit_null_test,
    };

    MunitSuite suite = munit_suite("colors", tests);

    return munit_suite_main(&suite, NULL, 0, NULL);
}

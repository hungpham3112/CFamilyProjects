#include "exercise.h"
#include "munit.h"
#include <stdio.h>

munit_case(RUN, test_formats_int1, {
    char buffer[100];
    snek_object_t i = new_integer(5);
    format_object(i, buffer);

    assert_string_equal("int:5", buffer, "formats INTEGER");
});

munit_case(RUN, test_formats_string1, {
    char buffer[100];
    snek_object_t s = new_string("Hello!");
    format_object(s, buffer);

    assert_string_equal("string:Hello!", buffer, "formats STRING");
});

munit_case(SUBMIT, test_formats_int2, {
    char buffer[100];
    snek_object_t i = new_integer(2014);
    format_object(i, buffer);

    assert_string_equal("int:2014", buffer, "formats INTEGER");
});

munit_case(SUBMIT, test_formats_string2, {
    char buffer[100];
    snek_object_t s = new_string("nvim btw");
    format_object(s, buffer);

    assert_string_equal("string:nvim btw", buffer, "formats STRING");
});

int main()
{
    MunitTest tests[] = {
        munit_test("/integer", test_formats_int1),
        munit_test("/string", test_formats_string1),
        munit_test("/integer_nvim", test_formats_int2),
        munit_test("/string_nvim", test_formats_string2),
        munit_null_test,
    };

    MunitSuite suite = munit_suite("format", tests);

    return munit_suite_main(&suite, NULL, 0, NULL);
}

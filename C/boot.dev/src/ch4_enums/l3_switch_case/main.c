#include "http.h"
#include "munit.h"

munit_case(RUN, test_switch_enum, {
    assert_string_equal(http_to_str(HTTP_BAD_REQUEST), "400 Bad Request", "");
    assert_string_equal(http_to_str(HTTP_UNAUTHORIZED), "401 Unauthorized", "");
    assert_string_equal(http_to_str(HTTP_NOT_FOUND), "404 Not Found", "");
    assert_string_equal(http_to_str(HTTP_TEAPOT), "418 I AM A TEAPOT!", "");
    assert_string_equal(http_to_str(HTTP_INTERNAL_SERVER_ERROR), "500 Internal Server Error", "");
});

munit_case(SUBMIT, test_switch_enum_default,
           { assert_string_equal(http_to_str((http_error_code_t)999), "Unknown HTTP status code", ""); });

int main()
{
    MunitTest tests[] = {
        munit_test("/switch_enum", test_switch_enum),
        munit_test("/switch_enum_default", test_switch_enum_default),
        munit_null_test,
    };

    MunitSuite suite = munit_suite("http", tests);

    return munit_suite_main(&suite, NULL, 0, NULL);
}

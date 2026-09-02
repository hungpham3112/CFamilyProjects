#include "exercise.h"
#include "munit.h"
#include <stdlib.h>
#include <string.h>

munit_case(RUN, test_create_token_pointer_array_single,
           {
               token_t token = {"hello", 1, 1};
               token_t **result = create_token_pointer_array(&token, 1);

               munit_assert_not_null(result, "Result array should not be null");
               munit_assert_not_null(result[0], "First token pointer should not be null");
               munit_assert_string_equal(result[0]->literal, "hello", "Literal should match");
               munit_assert_int(result[0]->line, ==, 1, "Line number should match");
               munit_assert_int(result[0]->column, ==, 1, "Column number should match");
               munit_assert_ptr_not_equal(result[0], &token, "Token pointer should not point to original token");

               free(result[0]);
               free(result);
           })

    munit_case(RUN, test_create_token_pointer_array_multiple,
               {
                   token_t tokens[3] = {
                       {"foo", 1, 1},
                       {"bar", 2, 5},
                       {"baz", 3, 10},
                   };
                   token_t **result = create_token_pointer_array(tokens, 3);

                   munit_assert_not_null(result, "Result array should not be null");
                   for (int i = 0; i < 3; i++)
                   {
                       munit_assert_not_null(result[i], "Token pointer should not be null");
                       munit_assert_string_equal(result[i]->literal, tokens[i].literal, "Literal should match");
                       munit_assert_int(result[i]->line, ==, tokens[i].line, "Line number should match");
                       munit_assert_int(result[i]->column, ==, tokens[i].column, "Column number should match");
                       munit_assert_ptr_not_equal(result[i], &tokens[i],
                                                  "Token pointer should not point to original token");
                   }

                   for (int i = 0; i < 3; i++)
                   {
                       free(result[i]);
                   }
                   free(result);
               })

        munit_case(SUBMIT, test_create_token_pointer_array_memory_allocation,
                   {
                       token_t tokens[2] = {
                           {"test1", 1, 1},
                           {"test2", 2, 2},
                       };
                       token_t **result = create_token_pointer_array(tokens, 2);

                       munit_assert_not_null(result, "Result array should not be null");
                       munit_assert_not_null(result[0], "First token pointer should not be null");
                       munit_assert_not_null(result[1], "Second token pointer should not be null");
                       munit_assert_ptr_not_equal(result[0], result[1], "Token pointers should be different");
                       munit_assert_ptr_not_equal(result[0], &tokens[0],
                                                  "First token pointer should not point to original token");
                       munit_assert_ptr_not_equal(result[1], &tokens[1],
                                                  "Second token pointer should not point to original token");

                       free(result[0]);
                       free(result[1]);
                       free(result);
                   })

            int main()
{
    MunitTest tests[] = {
        munit_test("/test_create_token_pointer_array_single", test_create_token_pointer_array_single),
        munit_test("/test_create_token_pointer_array_multiple", test_create_token_pointer_array_multiple),
        munit_test("/test_create_token_pointer_array_memory_allocation",
                   test_create_token_pointer_array_memory_allocation),
        munit_null_test,
    };

    MunitSuite suite = munit_suite("create_token_pointer_array", tests);

    return munit_suite_main(&suite, NULL, 0, NULL);
}

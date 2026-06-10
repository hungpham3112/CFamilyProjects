#ifndef LOCAL_BOOTDEV_STYLE_MUNIT_H
#define LOCAL_BOOTDEV_STYLE_MUNIT_H

#include "../third_party/munit/munit.h"

#define RUN RUN
#define SUBMIT SUBMIT

#define munit_case(kind, name, body)                                        \
  static MunitResult name(const MunitParameter params[], void *user_data) { \
    (void)params;                                                           \
    (void)user_data;                                                        \
    (void)#kind;                                                            \
    body                                                                    \
    return MUNIT_OK;                                                        \
  }                                                                         \
  typedef int name##_munit_case_requires_semicolon

#define assert_int(left, op, right, msg)       \
  do {                                         \
    (void)(msg);                               \
    munit_assert_int((left), op, (right));     \
  } while (0)

#define assert_uint(left, op, right, msg)      \
  do {                                         \
    (void)(msg);                               \
    munit_assert_uint((left), op, (right));    \
  } while (0)

#define assert_long(left, op, right, msg)      \
  do {                                         \
    (void)(msg);                               \
    munit_assert_long((left), op, (right));    \
  } while (0)

#define assert_size(left, op, right, msg)      \
  do {                                         \
    (void)(msg);                               \
    munit_assert_size((left), op, (right));    \
  } while (0)

#define assert_true(expr, msg)                 \
  do {                                         \
    (void)(msg);                               \
    munit_assert_true(expr);                   \
  } while (0)

#define assert_false(expr, msg)                \
  do {                                         \
    (void)(msg);                               \
    munit_assert_false(expr);                  \
  } while (0)

#define assert_null(ptr, msg)                  \
  do {                                         \
    (void)(msg);                               \
    munit_assert_null(ptr);                    \
  } while (0)

#define assert_not_null(ptr, msg)              \
  do {                                         \
    (void)(msg);                               \
    munit_assert_not_null(ptr);                \
  } while (0)

#define assert_string_equal(left, right, msg)    \
  do {                                           \
    (void)(msg);                                 \
    munit_assert_string_equal((left), (right));  \
  } while (0)

#define munit_test(path, fn) \
  { (char *)(path), fn, NULL, NULL, MUNIT_TEST_OPTION_NONE, NULL }

#define munit_null_test \
  { NULL, NULL, NULL, NULL, MUNIT_TEST_OPTION_NONE, NULL }

#define munit_suite(name, tests_array) \
  { "/" name, tests_array, NULL, 1, MUNIT_SUITE_OPTION_NONE }

#endif

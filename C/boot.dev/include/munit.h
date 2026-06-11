#ifndef LOCAL_BOOTDEV_STYLE_MUNIT_H
#define LOCAL_BOOTDEV_STYLE_MUNIT_H

#include <stddef.h>
#include <stdint.h>
#include <string.h>

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

/*
 * Macro dispatchers.
 *
 * Support both upstream-style:
 *   munit_assert_int(a, ==, b)
 *
 * and Boot.dev-style:
 *   munit_assert_int(a, ==, b, "message")
 */

#define BOOTDEV_SELECT_3_4(_1, _2, _3, _4, NAME, ...) NAME
#define BOOTDEV_SELECT_2_3(_1, _2, _3, NAME, ...) NAME
#define BOOTDEV_SELECT_1_2(_1, _2, NAME, ...) NAME

#define BOOTDEV_ASSERT_OP(left, op, right)                                  \
  do {                                                                      \
    if (!((left) op (right))) {                                              \
      munit_errorf("assertion failed: %s %s %s", #left, #op, #right);        \
    }                                                                       \
  } while (0)

#define BOOTDEV_ASSERT_OP_MSG(left, op, right, msg)                         \
  do {                                                                      \
    (void)(msg);                                                            \
    if (!((left) op (right))) {                                              \
      munit_errorf("assertion failed: %s %s %s", #left, #op, #right);        \
    }                                                                       \
  } while (0)

#define BOOTDEV_ASSERT_TYPED(left, op, right, type)                         \
  do {                                                                      \
    type bootdev_left_value = (type)(left);                                  \
    type bootdev_right_value = (type)(right);                                \
    if (!(bootdev_left_value op bootdev_right_value)) {                      \
      munit_errorf("assertion failed: %s %s %s", #left, #op, #right);        \
    }                                                                       \
  } while (0)

#define BOOTDEV_ASSERT_TYPED_MSG(left, op, right, msg, type)                 \
  do {                                                                      \
    (void)(msg);                                                            \
    type bootdev_left_value = (type)(left);                                  \
    type bootdev_right_value = (type)(right);                                \
    if (!(bootdev_left_value op bootdev_right_value)) {                      \
      munit_errorf("assertion failed: %s %s %s", #left, #op, #right);        \
    }                                                                       \
  } while (0)

#define BOOTDEV_ASSERT_STRING_EQUAL(left, right)                             \
  do {                                                                      \
    const char *bootdev_left = (left);                                       \
    const char *bootdev_right = (right);                                     \
    if ((bootdev_left == NULL) != (bootdev_right == NULL)) {                 \
      munit_errorf("string equality failed: %s != %s", #left, #right);       \
    }                                                                       \
    if (bootdev_left != NULL && strcmp(bootdev_left, bootdev_right) != 0) {   \
      munit_errorf("string equality failed: %s != %s", #left, #right);       \
    }                                                                       \
  } while (0)

#define BOOTDEV_ASSERT_STRING_EQUAL_MSG(left, right, msg)                    \
  do {                                                                      \
    (void)(msg);                                                            \
    BOOTDEV_ASSERT_STRING_EQUAL((left), (right));                            \
  } while (0)

#define BOOTDEV_ASSERT_TRUE_1(expr)                                          \
  do {                                                                      \
    if (!(expr)) {                                                           \
      munit_errorf("assertion failed: %s", #expr);                           \
    }                                                                       \
  } while (0)

#define BOOTDEV_ASSERT_TRUE_2(expr, msg)                                     \
  do {                                                                      \
    (void)(msg);                                                            \
    if (!(expr)) {                                                           \
      munit_errorf("assertion failed: %s", #expr);                           \
    }                                                                       \
  } while (0)

#define BOOTDEV_ASSERT_FALSE_1(expr)                                         \
  do {                                                                      \
    if (expr) {                                                              \
      munit_errorf("assertion failed: !(%s)", #expr);                        \
    }                                                                       \
  } while (0)

#define BOOTDEV_ASSERT_FALSE_2(expr, msg)                                    \
  do {                                                                      \
    (void)(msg);                                                            \
    if (expr) {                                                              \
      munit_errorf("assertion failed: !(%s)", #expr);                        \
    }                                                                       \
  } while (0)

#define BOOTDEV_ASSERT_NULL_1(ptr)                                           \
  do {                                                                      \
    if ((ptr) != NULL) {                                                     \
      munit_errorf("assertion failed: %s == NULL", #ptr);                    \
    }                                                                       \
  } while (0)

#define BOOTDEV_ASSERT_NULL_2(ptr, msg)                                      \
  do {                                                                      \
    (void)(msg);                                                            \
    if ((ptr) != NULL) {                                                     \
      munit_errorf("assertion failed: %s == NULL", #ptr);                    \
    }                                                                       \
  } while (0)

#define BOOTDEV_ASSERT_NOT_NULL_1(ptr)                                       \
  do {                                                                      \
    if ((ptr) == NULL) {                                                     \
      munit_errorf("assertion failed: %s != NULL", #ptr);                    \
    }                                                                       \
  } while (0)

#define BOOTDEV_ASSERT_NOT_NULL_2(ptr, msg)                                  \
  do {                                                                      \
    (void)(msg);                                                            \
    if ((ptr) == NULL) {                                                     \
      munit_errorf("assertion failed: %s != NULL", #ptr);                    \
    }                                                                       \
  } while (0)

/*
 * Replace upstream assert macros with Boot.dev-compatible versions.
 */

#undef munit_assert_int
#define BOOTDEV_ASSERT_INT_3(left, op, right) \
  BOOTDEV_ASSERT_TYPED((left), op, (right), int)
#define BOOTDEV_ASSERT_INT_4(left, op, right, msg) \
  BOOTDEV_ASSERT_TYPED_MSG((left), op, (right), (msg), int)
#define munit_assert_int(...) \
  BOOTDEV_SELECT_3_4(__VA_ARGS__, BOOTDEV_ASSERT_INT_4, BOOTDEV_ASSERT_INT_3)(__VA_ARGS__)

#undef munit_assert_uint
#define BOOTDEV_ASSERT_UINT_3(left, op, right) \
  BOOTDEV_ASSERT_TYPED((left), op, (right), unsigned int)
#define BOOTDEV_ASSERT_UINT_4(left, op, right, msg) \
  BOOTDEV_ASSERT_TYPED_MSG((left), op, (right), (msg), unsigned int)
#define munit_assert_uint(...) \
  BOOTDEV_SELECT_3_4(__VA_ARGS__, BOOTDEV_ASSERT_UINT_4, BOOTDEV_ASSERT_UINT_3)(__VA_ARGS__)

#undef munit_assert_long
#define BOOTDEV_ASSERT_LONG_3(left, op, right) \
  BOOTDEV_ASSERT_TYPED((left), op, (right), long)
#define BOOTDEV_ASSERT_LONG_4(left, op, right, msg) \
  BOOTDEV_ASSERT_TYPED_MSG((left), op, (right), (msg), long)
#define munit_assert_long(...) \
  BOOTDEV_SELECT_3_4(__VA_ARGS__, BOOTDEV_ASSERT_LONG_4, BOOTDEV_ASSERT_LONG_3)(__VA_ARGS__)

#undef munit_assert_size
#define BOOTDEV_ASSERT_SIZE_3(left, op, right) \
  BOOTDEV_ASSERT_TYPED((left), op, (right), size_t)
#define BOOTDEV_ASSERT_SIZE_4(left, op, right, msg) \
  BOOTDEV_ASSERT_TYPED_MSG((left), op, (right), (msg), size_t)
#define munit_assert_size(...) \
  BOOTDEV_SELECT_3_4(__VA_ARGS__, BOOTDEV_ASSERT_SIZE_4, BOOTDEV_ASSERT_SIZE_3)(__VA_ARGS__)

#undef munit_assert_uint8
#define BOOTDEV_ASSERT_UINT8_3(left, op, right) \
  BOOTDEV_ASSERT_TYPED((left), op, (right), uint8_t)
#define BOOTDEV_ASSERT_UINT8_4(left, op, right, msg) \
  BOOTDEV_ASSERT_TYPED_MSG((left), op, (right), (msg), uint8_t)
#define munit_assert_uint8(...) \
  BOOTDEV_SELECT_3_4(__VA_ARGS__, BOOTDEV_ASSERT_UINT8_4, BOOTDEV_ASSERT_UINT8_3)(__VA_ARGS__)

#undef munit_assert_uint16
#define BOOTDEV_ASSERT_UINT16_3(left, op, right) \
  BOOTDEV_ASSERT_TYPED((left), op, (right), uint16_t)
#define BOOTDEV_ASSERT_UINT16_4(left, op, right, msg) \
  BOOTDEV_ASSERT_TYPED_MSG((left), op, (right), (msg), uint16_t)
#define munit_assert_uint16(...) \
  BOOTDEV_SELECT_3_4(__VA_ARGS__, BOOTDEV_ASSERT_UINT16_4, BOOTDEV_ASSERT_UINT16_3)(__VA_ARGS__)

#undef munit_assert_uint32
#define BOOTDEV_ASSERT_UINT32_3(left, op, right) \
  BOOTDEV_ASSERT_TYPED((left), op, (right), uint32_t)
#define BOOTDEV_ASSERT_UINT32_4(left, op, right, msg) \
  BOOTDEV_ASSERT_TYPED_MSG((left), op, (right), (msg), uint32_t)
#define munit_assert_uint32(...) \
  BOOTDEV_SELECT_3_4(__VA_ARGS__, BOOTDEV_ASSERT_UINT32_4, BOOTDEV_ASSERT_UINT32_3)(__VA_ARGS__)

#undef munit_assert_uint64
#define BOOTDEV_ASSERT_UINT64_3(left, op, right) \
  BOOTDEV_ASSERT_TYPED((left), op, (right), uint64_t)
#define BOOTDEV_ASSERT_UINT64_4(left, op, right, msg) \
  BOOTDEV_ASSERT_TYPED_MSG((left), op, (right), (msg), uint64_t)
#define munit_assert_uint64(...) \
  BOOTDEV_SELECT_3_4(__VA_ARGS__, BOOTDEV_ASSERT_UINT64_4, BOOTDEV_ASSERT_UINT64_3)(__VA_ARGS__)

#undef munit_assert_true
#define munit_assert_true(...) \
  BOOTDEV_SELECT_1_2(__VA_ARGS__, BOOTDEV_ASSERT_TRUE_2, BOOTDEV_ASSERT_TRUE_1)(__VA_ARGS__)

#undef munit_assert_false
#define munit_assert_false(...) \
  BOOTDEV_SELECT_1_2(__VA_ARGS__, BOOTDEV_ASSERT_FALSE_2, BOOTDEV_ASSERT_FALSE_1)(__VA_ARGS__)

#undef munit_assert_null
#define munit_assert_null(...) \
  BOOTDEV_SELECT_1_2(__VA_ARGS__, BOOTDEV_ASSERT_NULL_2, BOOTDEV_ASSERT_NULL_1)(__VA_ARGS__)

#undef munit_assert_not_null
#define munit_assert_not_null(...) \
  BOOTDEV_SELECT_1_2(__VA_ARGS__, BOOTDEV_ASSERT_NOT_NULL_2, BOOTDEV_ASSERT_NOT_NULL_1)(__VA_ARGS__)

#undef munit_assert_string_equal
#define munit_assert_string_equal(...) \
  BOOTDEV_SELECT_2_3(__VA_ARGS__, BOOTDEV_ASSERT_STRING_EQUAL_MSG, BOOTDEV_ASSERT_STRING_EQUAL)(__VA_ARGS__)

/*
 * Short Boot.dev aliases.
 */

#define assert_int(left, op, right, msg) \
  munit_assert_int((left), op, (right), (msg))

#define assert_uint(left, op, right, msg) \
  munit_assert_uint((left), op, (right), (msg))

#define assert_long(left, op, right, msg) \
  munit_assert_long((left), op, (right), (msg))

#define assert_size(left, op, right, msg) \
  munit_assert_size((left), op, (right), (msg))

#define assert_true(expr, msg) \
  munit_assert_true((expr), (msg))

#define assert_false(expr, msg) \
  munit_assert_false((expr), (msg))

#define assert_null(ptr, msg) \
  munit_assert_null((ptr), (msg))

#define assert_not_null(ptr, msg) \
  munit_assert_not_null((ptr), (msg))

#define assert_string_equal(left, right, msg) \
  munit_assert_string_equal((left), (right), (msg))

#define munit_test(path, fn) \
  { (char *)(path), fn, NULL, NULL, MUNIT_TEST_OPTION_NONE, NULL }

#define munit_null_test \
  { NULL, NULL, NULL, NULL, MUNIT_TEST_OPTION_NONE, NULL }

#define munit_suite(name, tests_array) \
  { "/" name, tests_array, NULL, 1, MUNIT_SUITE_OPTION_NONE }

#endif

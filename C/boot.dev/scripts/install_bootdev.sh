#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

BOOTDEV_SRC="$ROOT/scripts/bootdev.c"
BOOTDEV_BIN="$ROOT/build/bootdev"

if [ ! -f "$BOOTDEV_SRC" ]; then
  echo "error: missing $BOOTDEV_SRC" >&2
  exit 2
fi

mkdir -p "$ROOT/build" "$ROOT/include" "$ROOT/third_party/munit"
touch "$ROOT/.bootdevroot"

if [ ! -f "$ROOT/third_party/munit/munit.c" ] || [ ! -f "$ROOT/third_party/munit/munit.h" ]; then
  if command -v curl >/dev/null 2>&1; then
    curl -fsSL -o "$ROOT/third_party/munit/munit.h" https://raw.githubusercontent.com/nemequ/munit/master/munit.h
    curl -fsSL -o "$ROOT/third_party/munit/munit.c" https://raw.githubusercontent.com/nemequ/munit/master/munit.c
  elif command -v wget >/dev/null 2>&1; then
    wget -qO "$ROOT/third_party/munit/munit.h" https://raw.githubusercontent.com/nemequ/munit/master/munit.h
    wget -qO "$ROOT/third_party/munit/munit.c" https://raw.githubusercontent.com/nemequ/munit/master/munit.c
  else
    echo "error: need curl or wget to download munit" >&2
    exit 2
  fi
fi

cat > "$ROOT/include/munit.h" <<'MUNIT_WRAPPER'
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
MUNIT_WRAPPER

cc -std=c17 -Wall -Wextra -O2 "$BOOTDEV_SRC" -o "$BOOTDEV_BIN"
sudo install -m 0755 "$BOOTDEV_BIN" /usr/local/bin/bootdev

echo "installed: /usr/local/bin/bootdev"
echo "root:      $ROOT"
echo "Run bootdev new <project> to start"

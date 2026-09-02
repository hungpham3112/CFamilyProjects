# Repository Instructions

## Project

Local C exercise runner for Boot.dev-style lessons.

Root marker: `.bootdevroot`.

Lesson layout:

```text
src/<chapter>/<lesson>/
  main.c       test harness
  exercise.h   public declarations
  exercise.c   implementation, when needed
```

## Commands

Run current exercise from repository root or lesson directory:

```bash
bootdev r
```

Equivalent long form:

```bash
bootdev run
```

Run a specific exercise from repository root:

```bash
bootdev run src/<chapter>/<lesson>
```

`bootdev` writes `compile_commands.json` at repository root for clangd.

## Build Contract

`bootdev` compiles with C17, `-Wall`, `-Wextra`, debug symbols, and include paths for:

```text
<lesson_dir>
include/
third_party/munit/
```

It also links `third_party/munit/munit.c` automatically.

Do not manually add MUnit sources to lesson directories.

## MUnit Wrapper

Use repository wrapper:

```c
#include "munit.h"
```

This resolves to root `include/munit.h`, which wraps `third_party/munit/munit.h` and provides Boot.dev-style helpers such as:

```c
munit_case(RUN, name, { ... });
munit_case(SUBMIT, name, { ... });
assert_int(left, ==, right, "message");
assert_not_null(ptr, "message");
```

If a Boot.dev lesson uses a missing assertion alias, add compatibility in root `include/munit.h`. Do not edit lesson tests to work around missing aliases unless lesson itself is wrong.

## Debugging

Before changing C code, reproduce from lesson directory with:

```bash
bootdev r
```

Read first compiler or linker error. If error mentions missing assertion or MUnit helper, inspect `include/munit.h` before changing lesson implementation.

For pointer lessons, keep ownership clear. If a function allocates memory through an output parameter, assign through pointed-to pointer, then write allocated value. Caller frees returned allocation.

## Security And Safety

Do not include local absolute paths, usernames, hostnames, API keys, tokens, passwords, SSH keys, or other secrets in committed files.

Do not commit generated binaries, object files, local caches, or build outputs unless explicitly required.

Do not run destructive commands such as `rm -rf`, `git reset --hard`, or `git clean -fd` unless user explicitly requests them and risk is clear.

Do not revert or overwrite unrelated user changes. Worktree may contain edits from other sessions.

## Editing Rules

Keep fixes minimal and lesson-focused.

Do not modify unrelated chapters or generated build artifacts.

Prefer fixing shared wrappers or build helpers when error is caused by repository infrastructure, not lesson code.

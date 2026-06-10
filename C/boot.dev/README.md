# bootdev runner

Local C exercise runner for Boot.dev-style lessons.

## Install

From the repository root:

```bash
./scripts/install-bootdev
```

This installs:

```text
/usr/local/bin/bootdev
```

Check:

```bash
which bootdev
bootdev --help
```

Expected:

```text
/usr/local/bin/bootdev
```

## Basic usage

Create a new exercise:

```bash
bootdev new hello
```

Run an exercise:

```bash
bootdev run hello
```

Shortcut:

```bash
bootdev hello
```

Run the current exercise:

```bash
bootdev run
```

or:

```bash
bootdev
```

Select the current exercise:

```bash
bootdev select hello
```

Show the current exercise:

```bash
bootdev current
```

List exercises:

```bash
bootdev list
```

Move forward/backward:

```bash
bootdev next
bootdev prev
```

Open the current exercise in editor:

```bash
bootdev edit
```

## Environment config

Show config:

```bash
bootdev env
```

Set compiler:

```bash
bootdev env CC /usr/bin/clang
```

Set editor:

```bash
bootdev env EDITOR /usr/bin/vim
```

Unset:

```bash
bootdev env unset CC
bootdev env unset EDITOR
```

Shell environment overrides persistent config:

```bash
CC=/usr/bin/gcc bootdev run hello
```

## Uninstall

```bash
sudo rm -f /usr/local/bin/bootdev
```


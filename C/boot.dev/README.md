# lab v2

Root is your chosen repo directory, for example:

```text
~/CFamilyProjects/C/boot.dev
```

Exercises live directly under that root:

```text
boot.dev/
├── .clabroot
├── include/munit.h
├── vendor/munit/munit.c
├── build/
├── src/lab.c
├── c_basics/
│   ├── README.md
│   ├── color.h
│   └── main.c
└── hello/
    ├── README.md
    └── main.c
```

No required `courses/` directory.

## Install

```bash
cd ~/CFamilyProjects/C/boot.dev
mkdir -p src
cp /path/to/lab.c src/lab.c
bash /path/to/install_lab.sh
```

## Usage

```bash
lab new hello    # creates ./hello
lab hello        # runs ./hello
lab c_basics     # runs ./c_basics
lab use hello    # sets current project
lab              # runs current project
lab ls           # lists projects
lab where        # shows current project
```

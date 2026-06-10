// bootdev.c - standalone C exercise runner.
// Compile:
//   cc -std=c17 -Wall -Wextra -O2 bootdev.c -o bootdev
//
// Install:
//   sudo install -m 0755 bootdev /usr/local/bin/bootdev
//
// Root detection:
//   - If CLAB_ROOT is set: use it.
//   - Otherwise walk upward from $PWD until .cbootdevroot is found.
//   - Fallback: walk upward until include/munit.h and third_party/munit/munit.c exist.
//
// Project contract:
//   <root>/<project>/
//   ├── main.c      // test harness
//   ├── *.h
//   └── *.c         // optional implementation files

#define _XOPEN_SOURCE 700
#define _POSIX_C_SOURCE 200809L

#include <sys/stat.h>
#include <sys/types.h>
#include <sys/wait.h>

#include <dirent.h>
#include <errno.h>
#include <limits.h>
#include <stdarg.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#ifndef PATH_MAX
#define PATH_MAX 4096
#endif

typedef struct
{
    char **items;
    size_t len;
    size_t cap;
} StrVec;

static void die(const char *fmt, ...)
{
    va_list ap;
    va_start(ap, fmt);
    fprintf(stderr, "error: ");
    vfprintf(stderr, fmt, ap);
    fprintf(stderr, "\n");
    va_end(ap);
    exit(2);
}

static char *xstrdup(const char *s)
{
    char *p = strdup(s);
    if (!p)
        die("out of memory");
    return p;
}

static void *xmalloc(size_t n)
{
    void *p = malloc(n);
    if (!p)
        die("out of memory");
    return p;
}

static void vec_push(StrVec *v, const char *s)
{
    if (v->len == v->cap)
    {
        size_t nc = v->cap ? v->cap * 2 : 16;
        char **ni = realloc(v->items, nc * sizeof(char *));
        if (!ni)
            die("out of memory");
        v->items = ni;
        v->cap = nc;
    }
    v->items[v->len++] = xstrdup(s);
}

static void vec_free(StrVec *v)
{
    for (size_t i = 0; i < v->len; i++)
        free(v->items[i]);
    free(v->items);
    v->items = NULL;
    v->len = 0;
    v->cap = 0;
}

static int cmp_strptr(const void *a, const void *b)
{
    const char *sa = *(const char *const *)a;
    const char *sb = *(const char *const *)b;
    return strcmp(sa, sb);
}

static void vec_sort(StrVec *v)
{
    qsort(v->items, v->len, sizeof(char *), cmp_strptr);
}

static char g_cc[PATH_MAX] = {0};
static char g_editor[PATH_MAX] = {0};

static const char *get_cc(void)
{
    const char *cc = getenv("CC");
    if (cc && cc[0])
        return cc;
    if (g_cc[0])
        return g_cc;
    return "cc";
}

static const char *get_editor(void)
{
    const char *ed = getenv("EDITOR");
    if (ed && ed[0])
        return ed;
    if (g_editor[0])
        return g_editor;
    return "nvim";
}

static bool path_exists(const char *path)
{
    struct stat st;
    return stat(path, &st) == 0;
}

static bool is_dir(const char *path)
{
    struct stat st;
    return stat(path, &st) == 0 && S_ISDIR(st.st_mode);
}

static bool is_file(const char *path)
{
    struct stat st;
    return stat(path, &st) == 0 && S_ISREG(st.st_mode);
}

static void path_join(char out[PATH_MAX], const char *a, const char *b)
{
    if (snprintf(out, PATH_MAX, "%s/%s", a, b) >= PATH_MAX)
    {
        die("path too long");
    }
}

static void parent_dir(char path[PATH_MAX])
{
    size_t n = strlen(path);
    while (n > 1 && path[n - 1] == '/')
        path[--n] = '\0';

    char *slash = strrchr(path, '/');
    if (!slash)
    {
        strcpy(path, ".");
    }
    else if (slash == path)
    {
        path[1] = '\0';
    }
    else
    {
        *slash = '\0';
    }
}

static bool has_suffix(const char *s, const char *suffix)
{
    size_t a = strlen(s), b = strlen(suffix);
    return a >= b && strcmp(s + a - b, suffix) == 0;
}

static bool starts_with_path(const char *path, const char *prefix)
{
    size_t n = strlen(prefix);
    if (strncmp(path, prefix, n) != 0)
        return false;
    return path[n] == '\0' || path[n] == '/';
}

static void relpath(char out[PATH_MAX], const char *root, const char *path)
{
    if (!starts_with_path(path, root))
        die("path is outside root: %s", path);
    const char *p = path + strlen(root);
    if (*p == '/')
        p++;
    if (snprintf(out, PATH_MAX, "%s", p) >= PATH_MAX)
        die("path too long");
}

static bool root_marker_ok(const char *root)
{
    char marker[PATH_MAX], inc[PATH_MAX], munit_c[PATH_MAX];

    path_join(marker, root, ".bootdevroot");
    if (is_file(marker))
        return true;

    path_join(marker, root, ".cbootdevroot");
    if (is_file(marker))
        return true;

    path_join(inc, root, "include/munit.h");
    path_join(munit_c, root, "third_party/munit/munit.c");
    return is_file(inc) && is_file(munit_c);
}

static void find_root(char root[PATH_MAX])
{
    const char *env_root = getenv("BOOTDEV_ROOT");
    if (!(env_root && env_root[0]))
        env_root = getenv("CLAB_ROOT");

    if (env_root && env_root[0])
    {
        char resolved[PATH_MAX];
        if (!realpath(env_root, resolved))
            die("CLAB_ROOT does not exist: %s", env_root);
        if (!root_marker_ok(resolved))
            die("CLAB_ROOT is not a bootdev root: %s", resolved);
        strcpy(root, resolved);
        return;
    }

    char cur[PATH_MAX];
    if (!getcwd(cur, sizeof(cur)))
        die("getcwd failed: %s", strerror(errno));

    while (true)
    {
        if (root_marker_ok(cur))
        {
            strcpy(root, cur);
            return;
        }

        if (strcmp(cur, "/") == 0)
            break;
        parent_dir(cur);
    }

    die("cannot find bootdev root. Run inside the repo or set CLAB_ROOT=/path/to/repo");
}

static bool is_project(const char *path)
{
    char main_c[PATH_MAX];
    path_join(main_c, path, "main.c");
    return is_dir(path) && is_file(main_c);
}

static bool ignored_dir_name(const char *name)
{
    return strcmp(name, ".git") == 0 || strcmp(name, ".bootdev") == 0 || strcmp(name, ".bootdev") == 0 ||
           strcmp(name, "build") == 0 || strcmp(name, "third_party") == 0 || strcmp(name, "include") == 0 ||
           strcmp(name, "src") == 0 || strcmp(name, "bin") == 0;
}

static bool find_project_from_cwd(const char *root, char project[PATH_MAX])
{
    char cur[PATH_MAX];
    if (!getcwd(cur, sizeof(cur)))
        die("getcwd failed: %s", strerror(errno));

    if (!starts_with_path(cur, root))
        return false;

    while (true)
    {
        if (is_project(cur))
        {
            strcpy(project, cur);
            return true;
        }

        if (strcmp(cur, root) == 0)
            break;
        parent_dir(cur);
    }

    return false;
}

static void ensure_dir(const char *path)
{
    if (mkdir(path, 0775) == 0)
        return;
    if (errno == EEXIST && is_dir(path))
        return;
    die("mkdir failed for %s: %s", path, strerror(errno));
}

static void ensure_parent_dirs(const char *path)
{
    char tmp[PATH_MAX];
    snprintf(tmp, sizeof(tmp), "%s", path);

    for (char *p = tmp + 1; *p; p++)
    {
        if (*p == '/')
        {
            *p = '\0';
            ensure_dir(tmp);
            *p = '/';
        }
    }
}

static void state_file_path(char out[PATH_MAX], const char *root)
{
    path_join(out, root, ".bootdev/current");
}

static void bootdev_env_path(char out[PATH_MAX], const char *root)
{
    path_join(out, root, ".bootdev/env");
}

static void trim_newline(char *s)
{
    s[strcspn(s, "\r\n")] = '\0';
}

static void load_persistent_env(const char *root)
{
    char path[PATH_MAX];
    bootdev_env_path(path, root);

    FILE *f = fopen(path, "r");
    if (!f)
        return;

    char line[PATH_MAX * 2];

    while (fgets(line, sizeof(line), f))
    {
        trim_newline(line);

        char *eq = strchr(line, '=');
        if (!eq)
            continue;

        *eq = '\0';
        const char *key = line;
        const char *value = eq + 1;

        if (strcmp(key, "CC") == 0)
        {
            snprintf(g_cc, sizeof(g_cc), "%s", value);
        }
        else if (strcmp(key, "EDITOR") == 0)
        {
            snprintf(g_editor, sizeof(g_editor), "%s", value);
        }
    }

    fclose(f);
}

static void save_persistent_env(const char *root)
{
    char dir[PATH_MAX], path[PATH_MAX];
    path_join(dir, root, ".bootdev");
    ensure_dir(dir);
    bootdev_env_path(path, root);

    FILE *f = fopen(path, "w");
    if (!f)
        die("cannot write env file: %s", path);

    if (g_cc[0])
        fprintf(f, "CC=%s\n", g_cc);
    if (g_editor[0])
        fprintf(f, "EDITOR=%s\n", g_editor);

    fclose(f);
}

static void cmd_env(const char *root, int argc, char **argv)
{
    load_persistent_env(root);

    if (argc == 2 || (argc == 3 && strcmp(argv[2], "list") == 0))
    {
        printf("CC=%s\n", get_cc());
        printf("EDITOR=%s\n", get_editor());
        return;
    }

    if (argc == 5 && strcmp(argv[2], "set") == 0)
    {
        const char *key = argv[3];
        const char *value = argv[4];

        if (strcmp(key, "CC") == 0)
        {
            snprintf(g_cc, sizeof(g_cc), "%s", value);
        }
        else if (strcmp(key, "EDITOR") == 0)
        {
            snprintf(g_editor, sizeof(g_editor), "%s", value);
        }
        else
        {
            die("unsupported env key: %s. supported: CC, EDITOR", key);
        }

        save_persistent_env(root);
        printf("%s=%s\n", key, value);
        return;
    }

    if (argc == 4 && strcmp(argv[2], "unset") == 0)
    {
        const char *key = argv[3];

        if (strcmp(key, "CC") == 0)
        {
            g_cc[0] = '\0';
        }
        else if (strcmp(key, "EDITOR") == 0)
        {
            g_editor[0] = '\0';
        }
        else
        {
            die("unsupported env key: %s. supported: CC, EDITOR", key);
        }

        save_persistent_env(root);
        printf("unset %s\n", key);
        return;
    }

    if (argc == 4)
    {
        // Shorthand:
        //   bootdev env CC /usr/bin/clang
        //   bootdev env EDITOR /usr/bin/vim
        const char *key = argv[2];
        const char *value = argv[3];

        if (strcmp(key, "CC") == 0)
        {
            snprintf(g_cc, sizeof(g_cc), "%s", value);
        }
        else if (strcmp(key, "EDITOR") == 0)
        {
            snprintf(g_editor, sizeof(g_editor), "%s", value);
        }
        else
        {
            die("unsupported env key: %s. supported: CC, EDITOR", key);
        }

        save_persistent_env(root);
        printf("%s=%s\n", key, value);
        return;
    }

    die("usage: bootdev env [list] | bootdev env set <CC|EDITOR> <value> | bootdev env unset <CC|EDITOR>");
}

static void set_current_project(const char *root, const char *project)
{
    char bootdev_dir[PATH_MAX], state_path[PATH_MAX], rel[PATH_MAX];

    path_join(bootdev_dir, root, ".bootdev");
    ensure_dir(bootdev_dir);

    relpath(rel, root, project);
    state_file_path(state_path, root);

    FILE *f = fopen(state_path, "w");
    if (!f)
        die("cannot write %s: %s", state_path, strerror(errno));
    fprintf(f, "%s\n", rel);
    fclose(f);
}

static void get_state_project(const char *root, char project[PATH_MAX])
{
    char state_path[PATH_MAX];
    state_file_path(state_path, root);

    FILE *f = fopen(state_path, "r");
    if (!f)
        die("no current project. Run: bootdev select <project> or bootdev <project>");

    char rel[PATH_MAX];
    if (!fgets(rel, sizeof(rel), f))
    {
        fclose(f);
        die("state file is empty: %s", state_path);
    }
    fclose(f);

    rel[strcspn(rel, "\r\n")] = '\0';

    char abs_path[PATH_MAX];
    path_join(abs_path, root, rel);

    if (!is_project(abs_path))
        die("saved project is invalid: %s", rel);
    strcpy(project, abs_path);
}

static void current_project(const char *root, char project[PATH_MAX])
{
    if (find_project_from_cwd(root, project))
        return;
    get_state_project(root, project);
}

static void resolve_project_arg(const char *root, const char *arg, char project[PATH_MAX])
{
    char tmp[PATH_MAX];

    if (arg[0] == '/')
    {
        snprintf(tmp, sizeof(tmp), "%s", arg);
    }
    else
    {
        path_join(tmp, root, arg);
    }

    char resolved[PATH_MAX];
    if (!realpath(tmp, resolved))
    {
        die("project not found: %s. Create it with: bootdev new %s", arg, arg);
    }

    if (!starts_with_path(resolved, root))
        die("path escapes root: %s", arg);
    if (!is_project(resolved))
        die("not a project: %s", arg);

    strcpy(project, resolved);
}

static void collect_projects_recursive(const char *root, const char *dir, StrVec *out)
{
    (void)root;

    if (is_project(dir))
    {
        vec_push(out, dir);
        return;
    }

    DIR *d = opendir(dir);
    if (!d)
        return;

    struct dirent *ent;
    while ((ent = readdir(d)) != NULL)
    {
        if (strcmp(ent->d_name, ".") == 0 || strcmp(ent->d_name, "..") == 0)
            continue;
        if (ignored_dir_name(ent->d_name))
            continue;

        char child[PATH_MAX];
        path_join(child, dir, ent->d_name);

        if (is_dir(child))
            collect_projects_recursive(root, child, out);
    }

    closedir(d);
}

static void all_projects(const char *root, StrVec *out)
{
    collect_projects_recursive(root, root, out);
    vec_sort(out);
}

static void list_projects(const char *root)
{
    StrVec projects = {0};
    all_projects(root, &projects);

    char cur[PATH_MAX] = {0};
    bool has_cur = false;
    if (find_project_from_cwd(root, cur))
    {
        has_cur = true;
    }
    else
    {
        char state_path[PATH_MAX];
        state_file_path(state_path, root);
        if (is_file(state_path))
        {
            get_state_project(root, cur);
            has_cur = true;
        }
    }

    for (size_t i = 0; i < projects.len; i++)
    {
        char rel[PATH_MAX];
        relpath(rel, root, projects.items[i]);
        printf("%c %s\n", has_cur && strcmp(cur, projects.items[i]) == 0 ? '*' : ' ', rel);
    }

    vec_free(&projects);
}

static void sanitize_name(char *out, size_t out_size, const char *s)
{
    size_t j = 0;
    for (size_t i = 0; s[i] && j + 1 < out_size; i++)
    {
        char c = s[i];
        bool ok = (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z') || (c >= '0' && c <= '9') || c == '.' || c == '-' ||
                  c == '_';
        out[j++] = ok ? c : '_';
    }
    out[j] = '\0';
}

static void collect_c_sources_flat(const char *project, StrVec *sources)
{
    DIR *d = opendir(project);
    if (!d)
        die("cannot open project dir: %s", project);

    struct dirent *ent;
    while ((ent = readdir(d)) != NULL)
    {
        if (strcmp(ent->d_name, ".") == 0 || strcmp(ent->d_name, "..") == 0)
            continue;
        if (!has_suffix(ent->d_name, ".c"))
            continue;

        char p[PATH_MAX];
        path_join(p, project, ent->d_name);
        if (is_file(p))
            vec_push(sources, p);
    }

    closedir(d);
    vec_sort(sources);
}

static void json_write_escaped(FILE *f, const char *s)
{
    fputc('"', f);

    for (const unsigned char *p = (const unsigned char *)s; *p; p++)
    {
        switch (*p)
        {
        case '\\':
            fputs("\\\\", f);
            break;
        case '"':
            fputs("\\\"", f);
            break;
        case '\n':
            fputs("\\n", f);
            break;
        case '\r':
            fputs("\\r", f);
            break;
        case '\t':
            fputs("\\t", f);
            break;
        default:
            if (*p < 0x20)
            {
                fprintf(f, "\\u%04x", *p);
            }
            else
            {
                fputc(*p, f);
            }
            break;
        }
    }

    fputc('"', f);
}

static void write_compile_commands(const char *root, const char *project, const char *include_dir,
                                   const char *third_party_dir, const StrVec *sources)
{
    char db_path[PATH_MAX];
    path_join(db_path, root, "compile_commands.json");

    FILE *f = fopen(db_path, "w");
    if (!f)
        die("cannot write compile_commands.json: %s", strerror(errno));

    fprintf(f, "[\n");

    for (size_t i = 0; i < sources->len; i++)
    {
        fprintf(f, "  {\n");

        fprintf(f, "    \"directory\": ");
        json_write_escaped(f, root);
        fprintf(f, ",\n");

        fprintf(f, "    \"file\": ");
        json_write_escaped(f, sources->items[i]);
        fprintf(f, ",\n");

        fprintf(f, "    \"arguments\": [\n");

        const char *args[] = {
            get_cc(),    "-std=c17", "-Wall",         "-Wextra",         "-g", "-I", project, "-I",
            include_dir, "-I",       third_party_dir, sources->items[i], NULL,
        };

        for (size_t j = 0; args[j] != NULL; j++)
        {
            fprintf(f, "      ");
            json_write_escaped(f, args[j]);
            fprintf(f, "%s\n", args[j + 1] ? "," : "");
        }

        fprintf(f, "    ]\n");
        fprintf(f, "  }%s\n", (i + 1 < sources->len) ? "," : "");
    }

    fprintf(f, "]\n");
    fclose(f);
}

static int run_argv(char *const argv[])
{
    pid_t pid = fork();
    if (pid < 0)
        die("fork failed: %s", strerror(errno));

    if (pid == 0)
    {
        execvp(argv[0], argv);
        fprintf(stderr, "error: exec failed for %s: %s\n", argv[0], strerror(errno));
        _exit(127);
    }

    int status = 0;
    if (waitpid(pid, &status, 0) < 0)
        die("waitpid failed: %s", strerror(errno));

    if (WIFEXITED(status))
        return WEXITSTATUS(status);
    if (WIFSIGNALED(status))
        return 128 + WTERMSIG(status);
    return 1;
}

static int build_project(const char *root, const char *project, char output[PATH_MAX])
{
    char build_dir[PATH_MAX], rel[PATH_MAX], safe[PATH_MAX];
    path_join(build_dir, root, "build");
    ensure_dir(build_dir);

    relpath(rel, root, project);
    sanitize_name(safe, sizeof(safe), rel);
    path_join(output, build_dir, safe);

    StrVec sources = {0};
    collect_c_sources_flat(project, &sources);

    if (sources.len == 0)
    {
        vec_free(&sources);
        die("no .c files found in project: %s", rel);
    }

    char include_dir[PATH_MAX], third_party_dir[PATH_MAX], munit_c[PATH_MAX];
    path_join(include_dir, root, "include");
    path_join(third_party_dir, root, "third_party/munit");
    path_join(munit_c, root, "third_party/munit/munit.c");

    write_compile_commands(root, project, include_dir, third_party_dir, &sources);

    size_t cap = 32 + sources.len;
    size_t argc = 0;
    char **argv = xmalloc(cap * sizeof(char *));

#define ADD_ARG(s)                                                                                                     \
    do                                                                                                                 \
    {                                                                                                                  \
        if (argc + 1 >= cap)                                                                                           \
            die("argv overflow");                                                                                      \
        argv[argc++] = (char *)(s);                                                                                    \
    } while (0)

    ADD_ARG((char *)get_cc());
    ADD_ARG("-std=c17");
    ADD_ARG("-Wall");
    ADD_ARG("-Wextra");
    ADD_ARG("-g");
    ADD_ARG("-I");
    ADD_ARG((char *)project);
    ADD_ARG("-I");
    ADD_ARG(include_dir);
    ADD_ARG("-I");
    ADD_ARG(third_party_dir);

    for (size_t i = 0; i < sources.len; i++)
        ADD_ARG(sources.items[i]);

    ADD_ARG(munit_c);
    ADD_ARG("-o");
    ADD_ARG(output);
    ADD_ARG(NULL);

#undef ADD_ARG

    printf("BUILD %s\n", rel);
    int code = run_argv(argv);

    free(argv);
    vec_free(&sources);
    return code;
}

static int run_project(const char *root, const char *project)
{
    char output[PATH_MAX], rel[PATH_MAX];
    relpath(rel, root, project);

    int build_code = build_project(root, project, output);

    if (build_code != 0)
    {
        printf("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");
        printf("BUILD FAIL  %s\n", rel);
        printf("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");
        return build_code;
    }

    printf("RUN %s\n", rel);

    char *argv[] = {output, NULL};
    int code = run_argv(argv);

    printf("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");
    printf("%s  %s\n", code == 0 ? "PASS" : "FAIL", rel);
    printf("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    if (code == 0)
    {
        printf("\nNext:\n");
        printf("  bootdev n             run current project; move next only when it passes\n");
        printf("  bootdev <project>     run another project, e.g. bootdev hello\n");
    }
    else
    {
        printf("\nStill on this project. Fix the code, then run:\n");
        printf("  bootdev\n");
    }

    return code;
}

static long find_index(StrVec *v, const char *s)
{
    for (size_t i = 0; i < v->len; i++)
    {
        if (strcmp(v->items[i], s) == 0)
            return (long)i;
    }
    return -1;
}

static int cmd_next(const char *root)
{
    char project[PATH_MAX];
    current_project(root, project);

    int code = run_project(root, project);
    if (code != 0)
        return code;

    StrVec projects = {0};
    all_projects(root, &projects);

    long idx = find_index(&projects, project);
    if (idx < 0)
        die("current project not found in project tree");
    if ((size_t)idx + 1 >= projects.len)
        die("already at last project");

    set_current_project(root, projects.items[idx + 1]);

    char rel[PATH_MAX];
    relpath(rel, root, projects.items[idx + 1]);
    printf("\nmoved to: %s\n", rel);

    vec_free(&projects);
    return 0;
}

static void cmd_prev(const char *root)
{
    char project[PATH_MAX];
    current_project(root, project);

    StrVec projects = {0};
    all_projects(root, &projects);

    long idx = find_index(&projects, project);
    if (idx < 0)
        die("current project not found in project tree");
    if (idx == 0)
        die("already at first project");

    set_current_project(root, projects.items[idx - 1]);

    char rel[PATH_MAX];
    relpath(rel, root, projects.items[idx - 1]);
    printf("current project: %s\n", rel);

    vec_free(&projects);
}

static void cmd_where(const char *root)
{
    char project[PATH_MAX], rel[PATH_MAX];
    current_project(root, project);
    relpath(rel, root, project);
    printf("%s\n", rel);
}

static void cmd_use(const char *root, const char *arg)
{
    char project[PATH_MAX], rel[PATH_MAX];
    resolve_project_arg(root, arg, project);
    set_current_project(root, project);
    relpath(rel, root, project);
    printf("current project: %s\n", rel);
}

static void cmd_edit(const char *root)
{
    char project[PATH_MAX];
    current_project(root, project);

    char *argv[] = {(char *)get_editor(), project, NULL};
    int code = run_argv(argv);
    exit(code);
}

static void write_file_if_missing(const char *path, const char *content)
{
    if (path_exists(path))
        return;

    FILE *f = fopen(path, "w");
    if (!f)
        die("cannot create %s: %s", path, strerror(errno));
    fputs(content, f);
    fclose(f);
}

static void cmd_new(const char *root, const char *arg)
{
    char project[PATH_MAX];

    if (arg[0] == '/')
    {
        snprintf(project, sizeof(project), "%s", arg);
    }
    else
    {
        path_join(project, root, arg);
    }

    if (!starts_with_path(project, root))
        die("path escapes root: %s", arg);

    ensure_parent_dirs(project);
    ensure_dir(project);

    char main_c[PATH_MAX];
    path_join(main_c, project, "main.c");

    write_file_if_missing(main_c, "#include \"munit.h\"\n\n"
                                  "munit_case(RUN, test_placeholder, {\n"
                                  "  assert_int(1, ==, 1, \"placeholder\");\n"
                                  "});\n\n"
                                  "int main(void) {\n"
                                  "  MunitTest tests[] = {\n"
                                  "      munit_test(\"/placeholder\", test_placeholder),\n"
                                  "      munit_null_test,\n"
                                  "  };\n\n"
                                  "  MunitSuite suite = munit_suite(\"exercise\", tests);\n"
                                  "  return munit_suite_main(&suite, NULL, 0, NULL);\n"
                                  "}\n");

    char resolved[PATH_MAX], rel[PATH_MAX];
    if (!realpath(project, resolved))
        die("realpath failed: %s", project);
    relpath(rel, root, resolved);

    set_current_project(root, resolved);
    printf("created: %s\n", rel);
    printf("current project: %s\n", rel);
}

static void usage(void)
{
    printf("usage:\n"
           "  bootdev run [project]       build and run current or named project\n"
           "  bootdev <project>           shortcut for: bootdev run <project>\n"
           "  bootdev new <project>       create root/<project>\n"
           "  bootdev select <project>    set current project\n"
           "  bootdev current             show current project\n"
           "  bootdev next                run current; move next only if pass\n"
           "  bootdev prev                move previous\n"
           "  bootdev edit                open current project in $EDITOR\n"
           "  bootdev list                list projects\n"
           "  bootdev env                 show persistent env\n"
           "  bootdev env CC <path>       persist compiler path\n"
           "  bootdev env EDITOR <path>   persist editor path\n"
           "  bootdev env unset <key>     remove persisted key\n"
           "\n"
           "aliases:\n"
           "  r=run, n=next, p=prev, e=edit, ls=list, where=current\n"
           "\n"
           "environment override precedence:\n"
           "  shell env > .bootdev/env > default\n"
           "\n"
           "environment:\n"
           "  BOOTDEV_ROOT                optional explicit root\n"
           "  CC                          compiler override for this shell\n"
           "  EDITOR                      editor override for this shell\n");
}

int main(int argc, char **argv)
{
    char root[PATH_MAX];
    find_root(root);
    load_persistent_env(root);

    if (argc == 1)
    {
        char project[PATH_MAX];
        current_project(root, project);
        return run_project(root, project);
    }

    const char *cmd = argv[1];

    if (strcmp(cmd, "h") == 0 || strcmp(cmd, "help") == 0 || strcmp(cmd, "-h") == 0 || strcmp(cmd, "--help") == 0)
    {
        usage();
        return 0;
    }

    if (strcmp(cmd, "r") == 0 || strcmp(cmd, "run") == 0)
    {
        char project[PATH_MAX];

        if (argc == 3)
        {
            resolve_project_arg(root, argv[2], project);
            set_current_project(root, project);
        }
        else if (argc == 2)
        {
            current_project(root, project);
        }
        else
        {
            die("usage: bootdev run [project]");
        }

        return run_project(root, project);
    }

    if (strcmp(cmd, "n") == 0 || strcmp(cmd, "next") == 0)
        return cmd_next(root);

    if (strcmp(cmd, "p") == 0 || strcmp(cmd, "prev") == 0 || strcmp(cmd, "previous") == 0)
    {
        cmd_prev(root);
        return 0;
    }

    if (strcmp(cmd, "ls") == 0 || strcmp(cmd, "list") == 0)
    {
        list_projects(root);
        return 0;
    }

    if ((strcmp(cmd, "current") == 0 && argc == 2) || strcmp(cmd, "where") == 0 || strcmp(cmd, "w") == 0)
    {
        cmd_where(root);
        return 0;
    }

    if (strcmp(cmd, "e") == 0 || strcmp(cmd, "edit") == 0)
    {
        cmd_edit(root);
        return 0;
    }

    if (strcmp(cmd, "select") == 0 || strcmp(cmd, "default") == 0 || (strcmp(cmd, "current") == 0 && argc == 3))
    {
        if (argc != 3)
            die("usage: bootdev select <project>");
        cmd_use(root, argv[2]);
        return 0;
    }

    if (strcmp(cmd, "env") == 0 || strcmp(cmd, "config") == 0)
    {
        cmd_env(root, argc, argv);
        return 0;
    }

    if (strcmp(cmd, "new") == 0)
    {
        if (argc != 3)
            die("usage: bootdev new <project>");
        cmd_new(root, argv[2]);
        return 0;
    }

    if (argc == 2)
    {
        char project[PATH_MAX];
        resolve_project_arg(root, cmd, project);
        set_current_project(root, project);
        return run_project(root, project);
    }

    die("unknown command: %s", cmd);
}

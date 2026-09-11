/* Run the existing Python environment in the app's process, preserving the
 * bundle identity and delivery of macOS document-open events to Qt. */
#include <dlfcn.h>
#include <mach-o/dyld.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/stat.h>
#include "runtime_config.h"

int main(int argc, char **argv) {
    char executable[4096], resolved[4096], resources[4096];
    uint32_t size = sizeof(executable);
    if (_NSGetExecutablePath(executable, &size) != 0 || !realpath(executable, resolved)) {
        perror("Cannot locate application"); return 1;
    }
    char *slash = strrchr(resolved, '/');
    if (!slash) return 1;
    *slash = '\0';
    snprintf(resources, sizeof(resources), "%s/../Resources", resolved);
    if (chdir(resources) != 0) { perror("Cannot open application resources"); return 1; }
    const char *old_path = getenv("PATH");
    char path[16384];
    snprintf(path, sizeof(path), "/opt/homebrew/bin:/usr/local/bin:%s", old_path ? old_path : "/usr/bin:/bin");
    setenv("PATH", path, 1);
    int smoke_test = 0;
    for (int i = 1; i < argc; i++) if (strcmp(argv[i], "--smoke-test") == 0) smoke_test = 1;
    const char *home = getenv("HOME");
    if (home && !smoke_test) {
        char log_dir[4096], log_file[4096];
        snprintf(log_dir, sizeof(log_dir), "%s/Library/Logs/Mp4ToTranscript", home);
        mkdir(log_dir, 0700);
        snprintf(log_file, sizeof(log_file), "%s/launch.log", log_dir);
        freopen(log_file, "a", stdout);
        freopen(log_file, "a", stderr);
    }
    /* Do not inherit another Python installation from a terminal session. */
    unsetenv("PYTHONHOME");
    unsetenv("PYTHONPATH");
    void *python = dlopen(PYTHON_LIBRARY, RTLD_NOW | RTLD_GLOBAL);
    if (!python) { fprintf(stderr, "Python runtime unavailable: %s\n", dlerror()); return 1; }
    int (*run_python)(int, char **) = dlsym(python, "Py_BytesMain");
    if (!run_python) { fprintf(stderr, "Cannot load Py_BytesMain\n"); return 1; }
    char **arguments = calloc((size_t)argc + 4, sizeof(char *));
    if (!arguments) return 1;
    arguments[0] = PYTHON_EXECUTABLE;
    arguments[1] = "-m";
    arguments[2] = "mp4_to_transcript.desktop";
    int count = 3;
    for (int i = 1; i < argc; i++) {
        if (strncmp(argv[i], "-psn_", 5) != 0) arguments[count++] = argv[i];
    }
    int result = run_python(count, arguments);
    free(arguments);
    return result;
}

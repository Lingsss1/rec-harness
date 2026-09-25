/* Supply only the current process' live maps inside a chroot without mounting
 * host /proc. The trusted launcher must not exec after chroot. This grants no
 * directory descriptor, process memory, /proc/PID/root or other process access.
 */
#define _GNU_SOURCE
#include <dlfcn.h>
#include <errno.h>
#include <fcntl.h>
#include <pthread.h>
#include <stdio.h>
#include <stdarg.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/syscall.h>
#include <unistd.h>
static int maps_fd = -1;
static pthread_mutex_t maps_lock = PTHREAD_MUTEX_INITIALIZER;
__attribute__((constructor)) static void initialize(void) {
    maps_fd = syscall(SYS_openat, AT_FDCWD, "/proc/self/maps", O_RDONLY | O_CLOEXEC, 0);
}
int rec_harness_maps_fd(void) { return maps_fd; }
static int snapshot(void) {
    if (maps_fd < 0) { errno = ENOENT; return -1; }
    int result = syscall(SYS_memfd_create, "own-process-maps", MFD_CLOEXEC);
    if (result < 0) return -1;
    pthread_mutex_lock(&maps_lock);
    if (lseek(maps_fd, 0, SEEK_SET) < 0) { close(result); result = -1; }
    else {
        char buffer[16384]; ssize_t count;
        while ((count = read(maps_fd, buffer, sizeof(buffer))) > 0) {
            ssize_t offset = 0;
            while (offset < count) {
                ssize_t n = write(result, buffer + offset, count - offset);
                if (n <= 0) { count = -1; break; }
                offset += n;
            }
            if (count < 0) break;
        }
        if (count < 0) { close(result); result = -1; }
        else lseek(result, 0, SEEK_SET);
    }
    pthread_mutex_unlock(&maps_lock);
    return result;
}
FILE *fopen(const char *path, const char *mode) {
    FILE *(*original)(const char *,const char *) = dlsym(RTLD_NEXT,"fopen");
    if (!strcmp(path,"/proc/self/maps") && mode[0] == 'r' && !strchr(mode,'+')) {
        int fd = snapshot();
        return fd < 0 ? NULL : fdopen(fd,"r");
    }
    return original(path,mode);
}

int open(const char *path, int flags, ...) {
    if (!strcmp(path,"/proc/self/maps") && (flags & O_ACCMODE) == O_RDONLY) return snapshot();
    mode_t mode = 0;
    if (flags & O_CREAT) { va_list args; va_start(args,flags); mode = va_arg(args,int); va_end(args); }
    int (*original)(const char *,int,...) = dlsym(RTLD_NEXT,"open");
    return original(path,flags,mode);
}
int open64(const char *path, int flags, ...) {
    if (!strcmp(path,"/proc/self/maps") && (flags & O_ACCMODE) == O_RDONLY) return snapshot();
    mode_t mode = 0;
    if (flags & O_CREAT) { va_list args; va_start(args,flags); mode = va_arg(args,int); va_end(args); }
    int (*original)(const char *,int,...) = dlsym(RTLD_NEXT,"open64");
    return original(path,flags,mode);
}

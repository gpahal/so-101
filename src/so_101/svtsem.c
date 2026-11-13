// Applies SVT-AV1's own fix for its macOS teardown crash to the SVT-AV1 that PyAV bundles.
//
// The crash: when an encoder closes, a race in SVT-AV1 can leave its ref_buffer_available_semaphore one
// count below where it started. Apple's libdispatch traps (SIGTRAP, exit 133, "BUG IN CLIENT OF
// LIBDISPATCH: Semaphore object deallocated while in use") when a semaphore is freed below its initial
// count; Linux and Windows don't check, so only macOS crashes. Reported as
// https://github.com/AOMediaCodec/SVT-AV1/issues/2961 and fixed by
// https://github.com/AOMediaCodec/SVT-AV1/commit/049193fc0e16ba7f1afe8baa51292157b681cb2a (SVT-AV1
// 4.3.0), which on Apple creates each semaphore at 0 and signals it up to its initial count, so the
// dispose check compares against 0 and a lost count no longer matters.
//
// PyAV's wheels bundle SVT-AV1 3.0.0 (av 15.1.0, the newest LeRobot 0.7.0 allows), which predates the
// fix, so this file does the same at runtime. Loaded with DYLD_INSERT_LIBRARIES, its __interpose entry
// replaces dispatch_semaphore_create for every image in the process, but it changes only creations
// with a count above 0 whose caller is libSvtAv1Enc: one semaphore per encoder. Calls made from this
// image itself are never interposed, so the inner dispatch_semaphore_create is the real one.
//
// so_101/svtshim.py builds it with the Mac's clang, loads it, and says when it's no longer needed.

#include <dispatch/dispatch.h>
#include <dlfcn.h>
#include <stdatomic.h>
#include <string.h>

static atomic_long rewritten;  // how many semaphores this has created for SVT-AV1, for the tests

static int from_svt(const void *caller) {
    Dl_info info;
    return dladdr(caller, &info) && info.dli_fname && strstr(info.dli_fname, "libSvtAv1Enc");
}

dispatch_semaphore_t so101_semaphore_create(intptr_t value) {
    if (value > 0 && from_svt(__builtin_return_address(0))) {
        dispatch_semaphore_t semaphore = dispatch_semaphore_create(0);
        for (intptr_t i = 0; semaphore && i < value; i++) dispatch_semaphore_signal(semaphore);
        atomic_fetch_add(&rewritten, 1);
        return semaphore;
    }
    return dispatch_semaphore_create(value);
}

long so101_svt_semaphores(void) { return atomic_load(&rewritten); }

__attribute__((used, section("__DATA,__interpose"))) static const struct {
    const void *replacement;
    const void *original;
} interpose_semaphore_create = {(const void *)so101_semaphore_create, (const void *)dispatch_semaphore_create};

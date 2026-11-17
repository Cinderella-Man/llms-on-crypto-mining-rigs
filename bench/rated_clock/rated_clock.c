/*
 * rated_clock: asks NVIDIA's driver about its rated-clock control (RATED_TDP) on one card, by talking to the
 * driver's control device directly, the way NVIDIA's own libraries do. README.md says what each mode does, the
 * risk, and the order to run them in.
 *
 *   rated_clock status <card>                              read-only: what each client asks for, who wins
 *   rated_clock lock-boost <card> <seconds> [os|global]    lock at the turbo-boost point, hold, then reset
 *   rated_clock reset <card> [os|global]                   send the default action
 *
 * <card> is an index in PCI bus order (the order of nvidia-smi -i) or a bus id such as 00000000:01:00.0.
 * Every mode frees what it allocated. A failed call prints the driver's status code and the tool exits 1 without
 * repeating the call. A failed read stops lock-boost before it sends anything; once the lock was sent, the only call
 * that follows a failure is the reset (and the reads that show it). reset sends the default action even after a
 * failed read, since it is the undo.
 * SPDX-License-Identifier: MIT
 */
#define _GNU_SOURCE
#include <errno.h>
#include <fcntl.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/ioctl.h>
#include <time.h>
#include <unistd.h>

#include "nvidia/nvrm_abi.h"
#include "abi_check.h"

#define NV_OK_STATUS 0x00000000u          /* NV_OK */
#define NV_ERR_NOT_SUPPORTED_STATUS 0x56u /* NV_ERR_NOT_SUPPORTED */

static const struct { NV_STATUS code; const char *name; const char *text; } nv_status_table[] = {
#define NV_STATUS_CODE(name, code, string) { code, #name, string },
#include "nvidia/nvstatuscodes.h"
#undef NV_STATUS_CODE
};

static const char *const client_name[] = { "RM", "WAR_BUG_1785342", "GLOBAL", "OS", "PROFILE" };
static const char *const action_name[] = { "DEFAULT", "FORCE_EXCEED", "FORCE_LIMIT", "FORCE_LOCK", "FORCE_FLOOR" };
static const char *const vpstate_name[] = { "RATED_TDP", "TURBO_BOOST" };
#define NCLIENTS NV2080_CTRL_PERF_RATED_TDP_CLIENT_NUM_CLIENTS
#define NAME(table, v) ((unsigned)(v) < sizeof(table) / sizeof(table[0]) ? table[v] : "unknown")
#define NV_IOCTL(nr, size) _IOC(_IOC_READ | _IOC_WRITE, NV_IOCTL_MAGIC, (nr), (size))

static int ctl_fd = -1, dev_fd = -1;
static NvHandle h_client, h_device, h_subdevice;
static volatile sig_atomic_t stop_hold;

static void on_signal(int sig) { (void)sig; stop_hold = 1; }

static void stamp(void) {
    struct timespec ts;
    struct tm tm;
    clock_gettime(CLOCK_REALTIME, &ts);
    localtime_r(&ts.tv_sec, &tm);
    printf("%02d:%02d:%02d.%03ld ", tm.tm_hour, tm.tm_min, tm.tm_sec, ts.tv_nsec / 1000000);
}

/* Prints a failed call: the ioctl's errno when the ioctl itself failed, else the driver's status. Returns -1. */
static int fail(const char *what, int err, NV_STATUS status) {
    stamp();
    if (err) {
        printf("%s: ioctl failed: %s (errno %d)\n", what, strerror(err), err);
        return -1;
    }
    const char *name = "unknown", *text = "not in the 580.178.04 status list";
    for (size_t i = 0; i < sizeof(nv_status_table) / sizeof(nv_status_table[0]); i++)
        if (nv_status_table[i].code == status) {
            name = nv_status_table[i].name;
            text = nv_status_table[i].text;
            break;
        }
    printf("%s: driver status 0x%08x %s (%s)\n", what, status, name, text);
    return -1;
}

static int rm_alloc(NvHandle root, NvHandle parent, NvU32 cls, void *params, NvU32 size, NvHandle *out,
                    const char *what) {
    NVOS21_PARAMETERS a = { .hRoot = root, .hObjectParent = parent, .hClass = cls,
                            .pAllocParms = params, .paramsSize = size };
    int rc = ioctl(ctl_fd, NV_IOCTL(NV_ESC_RM_ALLOC, sizeof(a)), &a);
    if (rc != 0 || a.status != NV_OK_STATUS) return fail(what, rc ? errno : 0, a.status);
    *out = a.hObjectNew;
    return 0;
}

/* Returns 0, -1 after printing a failure, or 1 for NV_ERR_NOT_SUPPORTED when quiet_unsupported is set. */
static int rm_control(NvHandle object, NvU32 cmd, void *params, NvU32 size, const char *what, int quiet_unsupported) {
    NVOS54_PARAMETERS c = { .hClient = h_client, .hObject = object, .cmd = cmd, .params = params, .paramsSize = size };
    int rc = ioctl(ctl_fd, NV_IOCTL(NV_ESC_RM_CONTROL, sizeof(c)), &c);
    if (rc == 0 && c.status == NV_ERR_NOT_SUPPORTED_STATUS && quiet_unsupported) return 1;
    if (rc != 0 || c.status != NV_OK_STATUS) return fail(what, rc ? errno : 0, c.status);
    return 0;
}

static int rm_free(NvHandle parent, NvHandle *object, const char *what) {
    NVOS00_PARAMETERS f = { .hRoot = h_client, .hObjectParent = parent, .hObjectOld = *object };
    int rc = ioctl(ctl_fd, NV_IOCTL(NV_ESC_RM_FREE, sizeof(f)), &f);
    *object = 0;
    if (rc != 0 || f.status != NV_OK_STATUS) return fail(what, rc ? errno : 0, f.status);
    return 0;
}

static int cmp_card(const void *x, const void *y) {
    const nv_pci_info_t *a = &((const nv_ioctl_card_info_t *)x)->pci_info;
    const nv_pci_info_t *b = &((const nv_ioctl_card_info_t *)y)->pci_info;
    if (a->domain != b->domain) return a->domain < b->domain ? -1 : 1;
    if (a->bus != b->bus) return a->bus < b->bus ? -1 : 1;
    return (int)a->slot - (int)b->slot;
}

/* Refuses unless the driver's version string equals the headers' (the driver's own strict comparison). */
static int check_version(void) {
    nv_ioctl_rm_api_version_t v;
    memset(&v, 0, sizeof(v));
    v.cmd = NV_RM_API_VERSION_CMD_STRICT;
    snprintf(v.versionString, sizeof(v.versionString), "%s", RC_HEADERS_VERSION);
    if (ioctl(ctl_fd, NV_IOCTL(NV_ESC_CHECK_VERSION_STR, sizeof(v)), &v) == 0) {
        printf("driver %s: matches the headers\n", RC_HEADERS_VERSION);
        return 0;
    }
    if (v.reply == NV_RM_API_VERSION_REPLY_RECOGNIZED) {
        v.versionString[sizeof(v.versionString) - 1] = '\0';
        printf("refusing: the driver is %s, the headers are %s\n", v.versionString, RC_HEADERS_VERSION);
    } else {
        printf("refusing: the driver did not answer the version check (%s)\n", strerror(errno));
    }
    return -1;
}

static int open_card(const char *arg) {
    nv_ioctl_card_info_t cards[32];
    memset(cards, 0, sizeof(cards));
    if (ioctl(ctl_fd, NV_IOCTL(NV_ESC_CARD_INFO, sizeof(cards)), cards) != 0) return fail("card list", errno, 0);
    int n = 0;
    for (int i = 0; i < 32; i++)
        if (cards[i].valid) cards[n++] = cards[i];
    qsort(cards, n, sizeof(cards[0]), cmp_card);

    int pick = -1;
    unsigned dom = 0, bus, slot, fn;
    if (strchr(arg, ':')) {
        int fields = sscanf(arg, "%x:%x:%x.%x", &dom, &bus, &slot, &fn);
        if (fields != 4) {
            dom = 0;
            fields = sscanf(arg, "%x:%x.%x", &bus, &slot, &fn) + 1;
        }
        if (fields != 4) {
            printf("cannot read the bus id %s\n", arg);
            return -1;
        }
        for (int i = 0; i < n; i++)
            if (cards[i].pci_info.domain == dom && cards[i].pci_info.bus == bus && cards[i].pci_info.slot == slot) pick = i;
    } else {
        char *end;
        long idx = strtol(arg, &end, 10);
        if (*arg && !*end && idx >= 0 && idx < n) pick = (int)idx;
    }
    if (pick < 0) {
        printf("no card %s among the driver's %d cards\n", arg, n);
        return -1;
    }
    const nv_ioctl_card_info_t *c = &cards[pick];
    printf("card %d of %d: bus id %08x:%02x:%02x.0, device %04x:%04x, /dev/nvidia%u\n", pick, n, c->pci_info.domain,
           c->pci_info.bus, c->pci_info.slot, c->pci_info.vendor_id, c->pci_info.device_id, c->minor_number);

    char path[32];
    snprintf(path, sizeof(path), "/dev/nvidia%u", c->minor_number);
    dev_fd = open(path, O_RDWR | O_CLOEXEC); /* holding it keeps the card set up and lets this process use it */
    if (dev_fd < 0) return fail(path, errno, 0);

    NV0000_CTRL_GPU_GET_ID_INFO_V2_PARAMS id = { .gpuId = c->gpu_id };
    NV0080_ALLOC_PARAMETERS dev = { 0 };
    NV2080_ALLOC_PARAMETERS sub = { .subDeviceId = 0 };
    if (rm_alloc(0, 0, NV01_ROOT, NULL, 0, &h_client, "alloc root client") != 0) return -1;
    if (rm_control(h_client, NV0000_CTRL_CMD_GPU_GET_ID_INFO_V2, &id, sizeof(id), "gpu id info", 0) != 0) return -1;
    dev.deviceId = id.deviceInstance;
    dev.hClientShare = h_client;
    if (rm_alloc(h_client, h_client, NV01_DEVICE_0, &dev, sizeof(dev), &h_device, "alloc device") != 0) return -1;
    return rm_alloc(h_client, h_device, NV20_SUBDEVICE_0, &sub, sizeof(sub), &h_subdevice, "alloc subdevice");
}

/*
 * Reads each client's request (five gets, each made once) and then the status call, prints every field and which
 * client wins the arbitration. Returns -1 if any read failed; NV_ERR_NOT_SUPPORTED on the status call is not a
 * failure, since the 580.178.04 headers define no command for it.
 */
static int read_state(const char *when) {
    NV2080_CTRL_PERF_RATED_TDP_CONTROL_PARAMS req[NCLIENTS];
    char what[64];
    int failed = 0;
    stamp();
    printf("%s\n", when);
    for (int i = 0; i < NCLIENTS; i++) {
        memset(&req[i], 0, sizeof(req[i]));
        req[i].client = (NV2080_CTRL_PERF_RATED_TDP_CLIENT)i;
        snprintf(what, sizeof(what), "  get_control client %s", client_name[i]);
        if (rm_control(h_subdevice, NV2080_CTRL_CMD_PERF_RATED_TDP_GET_CONTROL, &req[i], sizeof(req[i]), what, 0)) {
            failed = 1;
            continue;
        }
        printf("  get_control: client %-15s input %-12s (%u)  vPstateType %-11s (%u)\n", client_name[i],
               NAME(action_name, req[i].input), req[i].input, NAME(vpstate_name, req[i].vPstateType), req[i].vPstateType);
    }
    int w = 0;
    while (w < NCLIENTS && req[w].input == NV2080_CTRL_PERF_RATED_TDP_ACTION_DEFAULT) w++;
    if (failed) {
        printf("  holder: unknown, a get failed\n");
    } else if (w == NCLIENTS) {
        printf("  holder: none; every client asks DEFAULT, which allows boost above the rated point\n");
    } else {
        int rated = req[w].input == NV2080_CTRL_PERF_RATED_TDP_ACTION_FORCE_LIMIT ||
                    (req[w].input == NV2080_CTRL_PERF_RATED_TDP_ACTION_FORCE_LOCK &&
                     req[w].vPstateType == NV2080_CTRL_PERF_VPSTATE_RATED_TDP);
        printf("  holder: client %s (the first, in priority order, whose action is not DEFAULT): %s at %s%s\n",
               client_name[w], NAME(action_name, req[w].input), NAME(vpstate_name, req[w].vPstateType),
               rated ? "; this holds the card at the rated point" : "");
    }

    NV2080_CTRL_PERF_RATED_TDP_STATUS_PARAMS st;
    memset(&st, 0, sizeof(st));
    int rc = rm_control(h_subdevice, RC_CMD_PERF_RATED_TDP_GET_STATUS, &st, sizeof(st), "  status call 0x2080206d", 1);
    if (rc == 1) {
        printf("  status: call 0x2080206d not supported by this driver (NV_ERR_NOT_SUPPORTED); the gets above stand\n");
        return failed ? -1 : 0;
    }
    if (rc != 0) return -1;
    printf("  status: rm.clientActiveMask 0x%08x  rm.bRegkeyLimitRatedTdp %u  output %s (%u)  outputVPstate %s (%u)\n",
           st.rm.clientActiveMask, st.rm.bRegkeyLimitRatedTdp, NAME(action_name, st.output), st.output,
           NAME(vpstate_name, st.outputVPstate), st.outputVPstate);
    for (int i = 0; i < NCLIENTS; i++)
        printf("  status: inputs[%-15s] %-12s (%u)  vPstateTypes[%-15s] %-11s (%u)\n", client_name[i],
               NAME(action_name, st.inputs[i]), st.inputs[i], client_name[i], NAME(vpstate_name, st.vPstateTypes[i]),
               st.vPstateTypes[i]);
    return failed ? -1 : 0;
}

static int set_control(NV2080_CTRL_PERF_RATED_TDP_CLIENT client, NV2080_CTRL_PERF_RATED_TDP_ACTION action,
                       NV2080_CTRL_PERF_RATED_TDP_VPSTATE_TYPE vp) {
    NV2080_CTRL_PERF_RATED_TDP_CONTROL_PARAMS p = { .client = client, .input = action, .vPstateType = vp };
    char what[96];
    snprintf(what, sizeof(what), "set_control client %s %s at %s", client_name[client], action_name[action],
             vpstate_name[vp]);
    if (rm_control(h_subdevice, NV2080_CTRL_CMD_PERF_RATED_TDP_SET_CONTROL, &p, sizeof(p), what, 0) != 0) return -1;
    stamp();
    printf("%s: sent\n", what);
    return 0;
}

static int run_reset(NV2080_CTRL_PERF_RATED_TDP_CLIENT client) {
    if (set_control(client, NV2080_CTRL_PERF_RATED_TDP_ACTION_DEFAULT, NV2080_CTRL_PERF_VPSTATE_RATED_TDP) != 0) {
        printf("the request may still be in place: it ends when the last process closes the card (persistence mode "
               "off), or with the reset mode\n");
        return -1;
    }
    return read_state("after reset");
}

/* The lock lives only while this process holds the card; whatever happens after the lock is sent, reset follows. */
static int run_lock(NV2080_CTRL_PERF_RATED_TDP_CLIENT client, long seconds) {
    if (read_state("before") != 0) {
        printf("refusing to send the lock: a read failed\n");
        return -1;
    }
    int rc = set_control(client, NV2080_CTRL_PERF_RATED_TDP_ACTION_FORCE_LOCK, NV2080_CTRL_PERF_VPSTATE_TURBO_BOOST);
    if (rc == 0) rc = read_state("after lock");
    if (rc == 0) {
        stamp();
        printf("holding %ld s (Ctrl-C ends the hold early)\n", seconds);
        struct timespec t0, t;
        clock_gettime(CLOCK_MONOTONIC, &t0);
        do {
            sleep(1);
            clock_gettime(CLOCK_MONOTONIC, &t);
        } while (!stop_hold && t.tv_sec - t0.tv_sec < seconds);
        stamp();
        printf("hold ended%s\n", stop_hold ? " by a signal" : "");
    }
    return (run_reset(client) != 0 || rc != 0) ? -1 : 0;
}

static void usage(void) {
    printf("usage: rated_clock status <card>\n"
           "       rated_clock lock-boost <card> <seconds 1-3600> [os|global]\n"
           "       rated_clock reset <card> [os|global]\n"
           "<card>: index in PCI bus order (as nvidia-smi -i) or a bus id such as 00000000:01:00.0; client: os by default\n");
}

static int parse_client(const char *s, NV2080_CTRL_PERF_RATED_TDP_CLIENT *client) {
    if (!s || !strcmp(s, "os")) *client = NV2080_CTRL_PERF_RATED_TDP_CLIENT_OS;
    else if (!strcmp(s, "global")) *client = NV2080_CTRL_PERF_RATED_TDP_CLIENT_GLOBAL;
    else return -1;
    return 0;
}

int main(int argc, char **argv) {
    setvbuf(stdout, NULL, _IOLBF, 0);
    const char *mode = argc > 1 ? argv[1] : "";
    NV2080_CTRL_PERF_RATED_TDP_CLIENT client = NV2080_CTRL_PERF_RATED_TDP_CLIENT_OS;
    long seconds = 0;
    char *end = NULL;
    int ok = (!strcmp(mode, "status") && argc == 3) ||
             (!strcmp(mode, "reset") && (argc == 3 || argc == 4) && parse_client(argc == 4 ? argv[3] : NULL, &client) == 0) ||
             (!strcmp(mode, "lock-boost") && (argc == 4 || argc == 5) &&
              (seconds = strtol(argv[3], &end, 10), *end == '\0' && seconds >= 1 && seconds <= 3600) &&
              parse_client(argc == 5 ? argv[4] : NULL, &client) == 0);
    if (!ok) {
        usage();
        return 2;
    }

    struct sigaction sa;
    memset(&sa, 0, sizeof(sa));
    sa.sa_handler = on_signal;
    sa.sa_flags = SA_RESTART; /* a signal never cuts a driver call short; it only ends the hold */
    sigaction(SIGINT, &sa, NULL);
    sigaction(SIGTERM, &sa, NULL);
    sigaction(SIGHUP, &sa, NULL);

    int rc = -1;
    ctl_fd = open("/dev/nvidiactl", O_RDWR | O_CLOEXEC);
    if (ctl_fd < 0) {
        fail("/dev/nvidiactl", errno, 0);
    } else if (check_version() == 0 && open_card(argv[2]) == 0) {
        if (!strcmp(mode, "status")) rc = read_state("status");
        else if (!strcmp(mode, "reset")) {
            int before = read_state("before");
            rc = (run_reset(client) != 0 || before != 0) ? -1 : 0;
        }
        else rc = run_lock(client, seconds);
    }

    /* free in reverse order of allocation; freeing the client would also free the rest */
    if (h_subdevice && rm_free(h_device, &h_subdevice, "free subdevice") != 0) rc = -1;
    if (h_device && rm_free(h_client, &h_device, "free device") != 0) rc = -1;
    if (h_client && rm_free(0, &h_client, "free root client") != 0) rc = -1;
    if (dev_fd >= 0) close(dev_fd);
    if (ctl_fd >= 0) close(ctl_fd);
    return rc == 0 ? 0 : 1;
}

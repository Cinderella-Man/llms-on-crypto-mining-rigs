/*
 * Excerpts from NVIDIA's open-gpu-kernel-modules at tag 580.178.04 (commit c8e6998):
 * only the definitions rated_clock.c needs, copied as they stand in the files named above each block.
 * abi_check.h asserts every size, offset and value used here; build.sh compiles those asserts against the
 * upstream headers too when NVSRC points at a checkout of that tag.
 *
 * Copyright notices of the source files:
 * SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved. (nvtypes.h, nvos.h)
 * SPDX-FileCopyrightText: Copyright (c) 2020-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved. (nv-ioctl.h)
 * SPDX-FileCopyrightText: Copyright (c) 2020-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved. (nv-ioctl-numbers.h)
 * SPDX-FileCopyrightText: Copyright (c) 1999-2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved. (nv_escape.h)
 * SPDX-FileCopyrightText: Copyright (c) 2014-2019 NVIDIA CORPORATION & AFFILIATES. All rights reserved. (nvstatus.h)
 * SPDX-FileCopyrightText: Copyright (c) 2001-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved. (cl0000.h)
 * SPDX-FileCopyrightText: Copyright (c) 2001-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved. (cl0080.h)
 * SPDX-FileCopyrightText: Copyright (c) 2002-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved. (cl2080.h)
 * SPDX-FileCopyrightText: Copyright (c) 2005-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved. (ctrl0000gpu.h, ctrl2080perf.h)
 * SPDX-License-Identifier: MIT
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */
#ifndef RATED_CLOCK_NVRM_ABI_H
#define RATED_CLOCK_NVRM_ABI_H

#include <stdint.h>

/* The driver version these definitions come from (version.mk: NVIDIA_VERSION). */
#define RC_HEADERS_VERSION "580.178.04"

#if !defined(__x86_64__)
#error "written for x86_64 Linux, the rigs' platform (nvtypes.h NV_64_BITS branch)"
#endif

/* ---- src/common/sdk/nvidia/inc/nvtypes.h (x86_64 / gcc branches) ---- */
typedef uint8_t            NvU8;
typedef uint16_t           NvU16;
typedef uint32_t           NvV32;
typedef uint32_t           NvU32;
typedef  int32_t           NvS32;
typedef uint64_t           NvU64;
typedef NvU32 NvHandle;
typedef NvU8 NvBool;
typedef void*              NvP64;
#define NV_ALIGN_BYTES(size) __attribute__ ((aligned (size)))
#define NV_DECLARE_ALIGNED(TYPE_VAR, ALIGN) TYPE_VAR __attribute__ ((aligned (ALIGN)))

/* ---- src/common/sdk/nvidia/inc/nvstatus.h ---- */
typedef NvU32 NV_STATUS;

/* ---- kernel-open/common/inc/nv-ioctl-numbers.h ---- */
#define NV_IOCTL_MAGIC      'F'
#define NV_IOCTL_BASE       200
#define NV_ESC_CARD_INFO             (NV_IOCTL_BASE + 0)
#define NV_ESC_CHECK_VERSION_STR     (NV_IOCTL_BASE + 10)

/* ---- src/nvidia/arch/nvalloc/unix/include/nv_escape.h ---- */
#define NV_ESC_RM_FREE                              0x29
#define NV_ESC_RM_CONTROL                           0x2A
#define NV_ESC_RM_ALLOC                             0x2B

/* ---- kernel-open/common/inc/nv-ioctl.h ---- */
typedef struct {
    NvU32    domain;        /* PCI domain number   */
    NvU8     bus;           /* PCI bus number      */
    NvU8     slot;          /* PCI slot number     */
    NvU8     function;      /* PCI function number */
    NvU16    vendor_id;     /* PCI vendor ID       */
    NvU16    device_id;     /* PCI device ID       */
} nv_pci_info_t;

typedef struct nv_ioctl_card_info
{
    NvBool        valid;
    nv_pci_info_t pci_info;            /* PCI config information      */
    NvU32         gpu_id;
    NvU16         interrupt_line;
    NvU64         reg_address    NV_ALIGN_BYTES(8);
    NvU64         reg_size       NV_ALIGN_BYTES(8);
    NvU64         fb_address     NV_ALIGN_BYTES(8);
    NvU64         fb_size        NV_ALIGN_BYTES(8);
    NvU32         minor_number;
    NvU8          dev_name[10];  /* device names such as vmgfx[0-32] for vmkernel */
} nv_ioctl_card_info_t;

#define NV_RM_API_VERSION_STRING_LENGTH 64

typedef struct nv_ioctl_rm_api_version
{
    NvU32 cmd;
    NvU32 reply;
    char versionString[NV_RM_API_VERSION_STRING_LENGTH];
} nv_ioctl_rm_api_version_t;

#define NV_RM_API_VERSION_CMD_STRICT         0
#define NV_RM_API_VERSION_REPLY_UNRECOGNIZED 0
#define NV_RM_API_VERSION_REPLY_RECOGNIZED   1

/* ---- src/common/sdk/nvidia/inc/nvos.h ---- */
typedef struct
{
  NvHandle  hRoot;
  NvHandle  hObjectParent;
  NvHandle  hObjectOld;
  NvV32     status;
} NVOS00_PARAMETERS;

typedef struct
{
    NvHandle hRoot;
    NvHandle hObjectParent;
    NvHandle hObjectNew;
    NvV32    hClass;
    NvP64    pAllocParms NV_ALIGN_BYTES(8);
    NvU32    paramsSize;
    NvV32    status;
} NVOS21_PARAMETERS;

typedef struct
{
    NvHandle hClient;
    NvHandle hObject;
    NvV32    cmd;
    NvU32    flags;
    NvP64    params NV_ALIGN_BYTES(8);
    NvU32    paramsSize;
    NvV32    status;
} NVOS54_PARAMETERS;

/* ---- src/common/sdk/nvidia/inc/class/cl0000.h ---- */
#define NV01_ROOT        (0x0U) /* finn: Evaluated from "NV0000_ALLOC_PARAMETERS_MESSAGE_ID" */

/* ---- src/common/sdk/nvidia/inc/class/cl0080.h ---- */
#define NV01_DEVICE_0      (0x80U) /* finn: Evaluated from "NV0080_ALLOC_PARAMETERS_MESSAGE_ID" */

typedef struct NV0080_ALLOC_PARAMETERS {
    NvU32    deviceId;
    NvHandle hClientShare;
    NvHandle hTargetClient;
    NvHandle hTargetDevice;
    NvV32    flags;
    NV_DECLARE_ALIGNED(NvU64 vaSpaceSize, 8);
    NV_DECLARE_ALIGNED(NvU64 vaStartInternal, 8);
    NV_DECLARE_ALIGNED(NvU64 vaLimitInternal, 8);
    NvV32    vaMode;
} NV0080_ALLOC_PARAMETERS;

/* ---- src/common/sdk/nvidia/inc/class/cl2080.h ---- */
#define NV20_SUBDEVICE_0      (0x2080U) /* finn: Evaluated from "NV2080_ALLOC_PARAMETERS_MESSAGE_ID" */

typedef struct NV2080_ALLOC_PARAMETERS {
    NvU32 subDeviceId;
} NV2080_ALLOC_PARAMETERS;

/* ---- src/common/sdk/nvidia/inc/ctrl/ctrl0000/ctrl0000gpu.h ---- */
#define NV0000_CTRL_CMD_GPU_GET_ID_INFO_V2       (0x205U) /* finn: Evaluated from "(FINN_NV01_ROOT_GPU_INTERFACE_ID << 8) | NV0000_CTRL_GPU_GET_ID_INFO_V2_PARAMS_MESSAGE_ID" */

typedef struct NV0000_CTRL_GPU_GET_ID_INFO_V2_PARAMS {
    NvU32 gpuId;
    NvU32 gpuFlags;
    NvU32 deviceInstance;
    NvU32 subDeviceInstance;
    NvU32 sliStatus;
    NvU32 boardId;
    NvU32 gpuInstance;
    NvS32 numaId;
} NV0000_CTRL_GPU_GET_ID_INFO_V2_PARAMS;

/* ---- src/common/sdk/nvidia/inc/ctrl/ctrl2080/ctrl2080perf.h (the RATED_TDP section; comments shortened) ---- */
/*
 * Enumeration of the RATED_TDP arbitration clients which make requests to force
 * enable/disable VF points above the RATED_TDP point.
 *
 * These clients are sorted in descending priority - the RM will arbitrate
 * between all clients in order of priority, taking as output the first client
 * whose input action != @ref NV2080_CTRL_PERF_RATED_TDP_ACTION_DEFAULT.
 */
typedef enum NV2080_CTRL_PERF_RATED_TDP_CLIENT {
    NV2080_CTRL_PERF_RATED_TDP_CLIENT_RM = 0,
    NV2080_CTRL_PERF_RATED_TDP_CLIENT_WAR_BUG_1785342 = 1,
    NV2080_CTRL_PERF_RATED_TDP_CLIENT_GLOBAL = 2,
    NV2080_CTRL_PERF_RATED_TDP_CLIENT_OS = 3,
    NV2080_CTRL_PERF_RATED_TDP_CLIENT_PROFILE = 4,
    NV2080_CTRL_PERF_RATED_TDP_CLIENT_NUM_CLIENTS = 5,
} NV2080_CTRL_PERF_RATED_TDP_CLIENT;

typedef enum NV2080_CTRL_PERF_RATED_TDP_ACTION {
    NV2080_CTRL_PERF_RATED_TDP_ACTION_DEFAULT = 0,
    NV2080_CTRL_PERF_RATED_TDP_ACTION_FORCE_EXCEED = 1,
    NV2080_CTRL_PERF_RATED_TDP_ACTION_FORCE_LIMIT = 2,
    NV2080_CTRL_PERF_RATED_TDP_ACTION_FORCE_LOCK = 3,
    NV2080_CTRL_PERF_RATED_TDP_ACTION_FORCE_FLOOR = 4,
} NV2080_CTRL_PERF_RATED_TDP_ACTION;

typedef enum NV2080_CTRL_PERF_RATED_TDP_VPSTATE_TYPE {
    NV2080_CTRL_PERF_VPSTATE_RATED_TDP = 0,
    NV2080_CTRL_PERF_VPSTATE_TURBO_BOOST = 1,
    NV2080_CTRL_PERF_VPSTATE_NUM_VPSTATES = 2,
} NV2080_CTRL_PERF_RATED_TDP_VPSTATE_TYPE;

#define NV2080_CTRL_PERF_RATED_TDP_STATUS_PARAMS_MESSAGE_ID (0x6DU)

typedef struct NV2080_CTRL_PERF_RATED_TDP_STATUS_PARAMS {
    struct {
        NvU32 clientActiveMask;
        NvU8  bRegkeyLimitRatedTdp;
    } rm;
    NV2080_CTRL_PERF_RATED_TDP_ACTION       output;
    NV2080_CTRL_PERF_RATED_TDP_VPSTATE_TYPE outputVPstate;
    NV2080_CTRL_PERF_RATED_TDP_ACTION       inputs[NV2080_CTRL_PERF_RATED_TDP_CLIENT_NUM_CLIENTS];
    NV2080_CTRL_PERF_RATED_TDP_VPSTATE_TYPE vPstateTypes[NV2080_CTRL_PERF_RATED_TDP_CLIENT_NUM_CLIENTS];
} NV2080_CTRL_PERF_RATED_TDP_STATUS_PARAMS;

#define NV2080_CTRL_CMD_PERF_RATED_TDP_GET_CONTROL (0x2080206e) /* finn: Evaluated from "(FINN_NV20_SUBDEVICE_0_PERF_INTERFACE_ID << 8) | NV2080_CTRL_PERF_RATED_TDP_GET_CONTROL_PARAMS_MESSAGE_ID" */

typedef struct NV2080_CTRL_PERF_RATED_TDP_CONTROL_PARAMS {
    NV2080_CTRL_PERF_RATED_TDP_CLIENT       client;
    NV2080_CTRL_PERF_RATED_TDP_ACTION       input;
    NV2080_CTRL_PERF_RATED_TDP_VPSTATE_TYPE vPstateType;
} NV2080_CTRL_PERF_RATED_TDP_CONTROL_PARAMS;

#define NV2080_CTRL_CMD_PERF_RATED_TDP_SET_CONTROL (0x2080206f) /* finn: Evaluated from "(FINN_NV20_SUBDEVICE_0_PERF_INTERFACE_ID << 8) | NV2080_CTRL_PERF_RATED_TDP_SET_CONTROL_PARAMS_MESSAGE_ID" */

/* ---- not in the 580.178.04 headers ----
 * The headers define the status structure (message id 0x6D) but no command for it, and the open modules export no
 * control 0x2080206d. Every other perf command is (0x2080 << 16) | (0x20 << 8) | its message id, so this is the id
 * the status structure's command would have. The closed driver that runs Pascal may or may not answer it; an unknown
 * id is refused with NV_ERR_NOT_SUPPORTED and a wrong parameter size with NV_ERR_INVALID_PARAM_STRUCT. */
#define RC_CMD_PERF_RATED_TDP_GET_STATUS (0x20802000U | NV2080_CTRL_PERF_RATED_TDP_STATUS_PARAMS_MESSAGE_ID)

#endif /* RATED_CLOCK_NVRM_ABI_H */

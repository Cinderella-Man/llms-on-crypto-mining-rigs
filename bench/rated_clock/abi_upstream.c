/*
 * Compiled with -fsyntax-only by build.sh when NVSRC names a checkout of NVIDIA's open-gpu-kernel-modules at tag
 * 580.178.04: the same checks as rated_clock.c, against NVIDIA's own headers instead of the excerpts. Never run.
 * SPDX-License-Identifier: MIT
 */
#include "nvtypes.h"
#include "nvstatus.h"
#include "nv-ioctl.h"
#include "nv_escape.h"
#include "nvos.h"
#include "class/cl0000.h"
#include "class/cl0080.h"
#include "class/cl2080.h"
#include "ctrl/ctrl0000/ctrl0000gpu.h"
#include "ctrl/ctrl2080/ctrl2080perf.h"

#include "abi_check.h"

# ROCM RCCL 2
# A package of optimized primitives for collective multi-GPU communication.

licenses(["notice"])

exports_files(["LICENSE.txt"])

load("@//tensorflow:tensorflow.bzl", "tf_gpu_library")
load("@//third_party/rccl:build_defs.bzl", "process_srcs")
load("@local_config_rocm//rocm:build_defs.bzl", "rocm_default_copts")

# Internally, during the build of librccl.so, source files include 'nccl.h'.
# RCCL creates two identical headers but then only packages rccl.h.
# We do the same here.
cc_library(
    name = "rccl_hdr",
    hdrs = process_srcs([
        "src/nccl.h.in",
    ]),
)
cc_library(
    name = "nccl_hdr",
    hdrs = process_srcs([
        "rccl.h",
    ]),
    deps = [":rccl_hdr"],
)

device_includes = [
    "src/collectives/collectives.h",
    "src/collectives/device/all_gather.cu",
    "src/collectives/device/all_gather.h",
    "src/collectives/device/all_reduce.cu",
    "src/collectives/device/all_reduce.h",
    "src/collectives/device/broadcast.cu",
    "src/collectives/device/broadcast.h",
    "src/collectives/device/common.h",
    "src/collectives/device/common_kernel.h",
    "src/collectives/device/ll_kernel.h",
    "src/collectives/device/primitives.h",
    "src/collectives/device/reduce.cu",
    "src/collectives/device/reduce.h",
    "src/collectives/device/reduce_kernel.h",
    "src/collectives/device/reduce_scatter.cu",
    "src/collectives/device/reduce_scatter.h",
]

src_includes = [
    "src/include/bootstrap.h",
    "src/include/common_coll.h",
    "src/include/core.h",
    "src/include/debug.h",
    "src/include/enqueue.h",
    "src/include/group.h",
    "src/include/ibvwrap.h",
    "src/include/nccl_net.h",
    "src/include/net.h",
    "src/include/nvlink.h",
    "src/include/nvlink_stub.h",
    "src/include/nvmlwrap.h",
    "src/include/param.h",
    "src/include/ring.h",
    "src/include/rings.h",
    "src/include/shm.h",
    "src/include/socket.h",
    "src/include/topo.h",
    "src/include/transport.h",
    "src/include/utils.h",
]

cc_library(
    name = "include_hdrs",
    hdrs = src_includes + device_includes,
    deps = ["@local_config_rocm//rocm:rocm_headers"],
)

cu_sources = process_srcs([
    "src/bootstrap.cu",
    "src/collectives/all_gather.cu",
    "src/collectives/all_reduce.cu",
    "src/collectives/broadcast.cu",
    "src/collectives/reduce.cu",
    "src/collectives/reduce_scatter.cu",
    "src/init.cu",
    "src/misc/enqueue.cu",
    "src/misc/group.cu",
    "src/misc/ibvwrap.cu",
    "src/misc/nvmlwrap_stub.cu",
    "src/misc/rings.cu",
    "src/misc/utils.cu",
    "src/ring.cu",
    "src/transport.cu",
    "src/transport/net.cu",
    "src/transport/net_ib.cu",
    "src/transport/net_socket.cu",
    "src/transport/p2p.cu",
    "src/transport/shm.cu",
])

device_sources = process_srcs([
    "src/collectives/device/functions.cu",
    ]) + [
    "src/collectives/device/all_gather_0.cpp",
    "src/collectives/device/all_reduce_0.cpp",
    "src/collectives/device/all_reduce_1.cpp",
    "src/collectives/device/all_reduce_2.cpp",
    "src/collectives/device/all_reduce_3.cpp",
    "src/collectives/device/broadcast_0.cpp",
    "src/collectives/device/reduce_0.cpp",
    "src/collectives/device/reduce_1.cpp",
    "src/collectives/device/reduce_2.cpp",
    "src/collectives/device/reduce_3.cpp",
    "src/collectives/device/reduce_scatter_0.cpp",
    "src/collectives/device/reduce_scatter_1.cpp",
    "src/collectives/device/reduce_scatter_2.cpp",
    "src/collectives/device/reduce_scatter_3.cpp",
]

tf_gpu_library(
    name = "device",
    srcs = device_sources,
    deps = [
        ":include_hdrs",
        ":src_hdrs",
    ],
)

# Primary RCCL target.
cc_library(
    name = "rccl",
    srcs = cu_sources + device_sources,
    hdrs = ["rccl.h"],
    includes = [
        "src",
        "src/include",
        "src/collectives",
        "src/collectives/device",
    ],
    copts = rocm_default_copts(),
    linkopts = ["-fgpu-rdc", "-hc-function-calls"],
    include_prefix = "third_party/rccl",
    visibility = ["//visibility:public"],
    deps = [
        ":device",
        ":include_hdrs",
        ":nccl_hdr",
    ],
)

filegroup(
    name = "LICENSE",
    visibility = ["//visibility:public"],
)

cc_library(
    name = "rccl",
    srcs = ["librccl.so"],
    hdrs = ["rccl.h"],
    include_prefix = "third_party/rccl",
    visibility = ["//visibility:public"],
    deps = [
        "@local_config_rocm//rocm:rocm_headers",
    ],
)

genrule(
    name = "rccl-files",
    outs = [
        "librccl.so",
        "rccl.h",
    ],
    cmd = """
cp "%{rccl_header_dir}/rccl.h" "$(@D)/rccl.h" &&
cp "%{rccl_library_dir}/librccl.so" "$(@D)/librccl.so"
""",
)

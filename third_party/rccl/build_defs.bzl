"""Repository rule for RCCL."""

load("@local_config_rocm//rocm:build_defs.bzl", "rocm_default_copts")
load("@bazel_tools//tools/cpp:toolchain_utils.bzl", "find_cpp_toolchain")

def _process_src_impl(ctx):
    """Applies various patches to the RCCL source."""
    substitutions = {}
    if ctx.file.src.basename == "nccl.h.in":
        substitutions.update({
          "${NCCL_MAJOR}": "2",
          "${NCCL_MINOR}": "3",
          "${NCCL_PATCH}": "7",
          "${NCCL_SUFFIX}": "",
          "${NCCL_VERSION}": "2307",
        })
    ctx.actions.expand_template(
        output = ctx.outputs.out,
        template = ctx.file.src,
        substitutions = substitutions,
    )

_process_src = rule(
    implementation = _process_src_impl,
    attrs = {
        "src": attr.label(allow_single_file = True),
        "out": attr.output(),
    },
)
"""Processes one RCCL source file so it can be compiled with bazel and clang."""

def _out(src):
    if src == "src/nccl.h.in":
      return "rccl.h"
    if src == "rccl.h":
      return "nccl.h"
    if src.endswith(".cu"):
      return src + ".cpp"
    return src

def process_srcs(srcs):
    """Processes files under src/ and copies them to the parent directory."""
    [_process_src(
      name = "_" + src,
      src = src,
      out = _out(src),
    ) for src in srcs]
    return ["_" + src for src in srcs]


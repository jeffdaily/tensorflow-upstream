# -*- Python -*-
"""Repository rule for RCCL configuration.

`rccl_configure` depends on the following environment variables:

  * `TF_RCCL_INSTALL_PATH`: Path of installed RCCL or empty to build from source.

"""

load(
    "//third_party/gpus:cuda_configure.bzl",
    "auto_configure_fail",
    "enable_cuda",
    "get_cpu_value",
)
load(
    "//third_party/gpus:rocm_configure.bzl",
    "enable_rocm",
)

_TF_RCCL_INSTALL_PATH = "TF_RCCL_INSTALL_PATH"
_TF_NEED_ROCM = "TF_NEED_ROCM"

_DEFINE_NCCL_MAJOR = "#define NCCL_MAJOR"
_DEFINE_NCCL_MINOR = "#define NCCL_MINOR"
_DEFINE_NCCL_PATCH = "#define NCCL_PATCH"

_RCCL_DUMMY_BUILD_CONTENT = """
filegroup(
  name = "LICENSE",
  visibility = ["//visibility:public"],
)

cc_library(
  name = "rccl",
  visibility = ["//visibility:public"],
)
"""

_RCCL_ARCHIVE_BUILD_CONTENT = """
filegroup(
  name = "LICENSE",
  data = ["@rccl_archive//:LICENSE.txt"],
  visibility = ["//visibility:public"],
)

alias(
  name = "rccl",
  actual = "@rccl_archive//:rccl",
  visibility = ["//visibility:public"],
)
"""

def _label(file):
    return Label("//third_party/rccl:{}".format(file))

def _rccl_configure_impl(repository_ctx):
    """Implementation of the rccl_configure repository rule."""
    if ((not enable_cuda(repository_ctx) and not enable_rocm(repository_ctx))
        or get_cpu_value(repository_ctx) not in ("Linux", "FreeBSD")):
        # Add a dummy build file to make bazel query happy.
        repository_ctx.file("BUILD", _RCCL_DUMMY_BUILD_CONTENT)
        return

    rccl_install_path = ""
    if _TF_RCCL_INSTALL_PATH in repository_ctx.os.environ:
        rccl_install_path = repository_ctx.os.environ[_TF_RCCL_INSTALL_PATH].strip()

    if rccl_install_path == "":
        # Alias to open source build from @rccl_archive.
        repository_ctx.file("BUILD", _RCCL_ARCHIVE_BUILD_CONTENT)
    else:
        rccl_include_dir =  "%s/include" % rccl_install_path
        rccl_library_dir =  "%s/lib" % rccl_install_path
        header_path = repository_ctx.path("%s/rccl.h" % rccl_include_dir)
        if not header_path.exists:
            auto_configure_fail(
                ("RCCL header not found at %s. To fix this rerun configure again " +
                 "with correct RCCL install path.") % header_path
            )
        # Create target for locally installed RCCL.
        repository_ctx.template("BUILD", _label("system.BUILD.tpl"), {
            "%{rccl_header_dir}": rccl_include_dir,
            "%{rccl_library_dir}": rccl_library_dir,
        })

rccl_configure = repository_rule(
    implementation = _rccl_configure_impl,
    environ = [
        _TF_RCCL_INSTALL_PATH,
        _TF_NEED_ROCM,
    ],
)
"""Detects and configures the RCCL configuration.

Add the following to your WORKSPACE FILE:

```python
rccl_configure(name = "local_config_rccl")
```

Args:
  name: A unique name for this workspace rule.
"""

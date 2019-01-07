/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "tensorflow/stream_executor/rocm/rocm_driver.h"

#include <stdint.h>
#include <stdlib.h>
#include <map>
#include <set>
#include <utility>

#include "absl/base/casts.h"
#include "absl/container/inlined_vector.h"
#include "absl/strings/str_cat.h"
#include "tensorflow/stream_executor/rocm/rocm_diagnostics.h"
#include "tensorflow/stream_executor/lib/env.h"
#include "tensorflow/stream_executor/lib/error.h"
#include "tensorflow/stream_executor/lib/human_readable.h"
#include "tensorflow/stream_executor/lib/notification.h"
#include "tensorflow/stream_executor/lib/ptr_util.h"
#include "tensorflow/stream_executor/lib/stacktrace.h"
#include "tensorflow/stream_executor/lib/static_threadlocal.h"
#include "tensorflow/stream_executor/lib/stringprintf.h"
#include "tensorflow/stream_executor/lib/threadpool.h"
#include "tensorflow/stream_executor/platform/logging.h"
#include "tensorflow/stream_executor/platform/mutex.h"
#include "tensorflow/stream_executor/platform/port.h"

bool FLAGS_gpuexec_rocm_driver_inject_init_error = false;
bool FLAGS_gpuexec_rocm_sync_around_driver_calls = false;
bool FLAGS_gpuexec_rocm_device_0_only = false;

// Debugging: on each push and pop of a rocm context, verify the current context
// matches the expected one.
bool kVerifyRocmContext = (getenv("HU_VERIFY") != NULL);

namespace stream_executor {
namespace rocm {

namespace {

// Manages the singleton map of contexts that we've created, mapping
// from the HUcontext to the RocmContext* that we pass around internally.
// This also manages assignment of unique ids to RocmContexts, to allow
// for fast comparison of a context against the current context.
//
// ROCM-runtime-created contexts are avoided, if triple angle
// brace launches are required, by using the scoped activations in
// rocm_activation.h.
class CreatedContexts {
 public:
  // Returns whether context is a member of the live set.
  static bool Has(HUcontext context) {
    tf_shared_lock lock(mu_);
    return Live()->find(context) != Live()->end();
  }

  // Adds context to the live set, or returns it if it's already present.
  static RocmContext* Add(HUcontext context) {
    CHECK(context != nullptr);
    mutex_lock lock(mu_);
    auto insert_result = Live()->insert(std::make_pair(context, nullptr));
    auto it = insert_result.first;
    if (insert_result.second) {
      // context was not present in the map.  Add it.
      it->second = MakeUnique<RocmContext>(context, next_id_++);
    }
    return it->second.get();
  }

  // Removes context from the live set.
  static void Remove(HUcontext context) {
    CHECK(context != nullptr);
    mutex_lock lock(mu_);
    auto it = Live()->find(context);
    CHECK(it != Live()->end()) << context;
    Live()->erase(it);
  }

 private:
  // Returns the live map singleton.
  static std::map<HUcontext, std::unique_ptr<RocmContext>> *Live() {
    static auto singleton =
        new std::map<HUcontext, std::unique_ptr<RocmContext>>;
    return singleton;
  }

  // Lock that guards access-to/mutation-of the live set.
  static mutex mu_;
  static int64 next_id_;
};

/* static */ mutex CreatedContexts::mu_{LINKER_INITIALIZED};
/* static */ int64 CreatedContexts::next_id_ = 1;  // 0 means "no context"

// Formats HUresult to output prettified values into a log stream.
string ToString(HUresult result) {
  const char *error_name;
  if (huGetErrorName(result, &error_name)) {
    return absl::StrCat("UNKNOWN ERROR (", static_cast<int>(result), ")");
  }
  const char *error_string;
  if (huGetErrorString(result, &error_string)) {
    return error_name;
  }
  return absl::StrCat(error_name, ": ", error_string);
}

// Formats hipError_t to output prettified values into a log stream.
// Error summaries taken from:
//
// TODO(leary) switch to cuGetErrorName when updated rocm.h is available.
string ToString(hipError_t result) {
#define OSTREAM_ROCM_ERROR(__name) \
  case hipError##__name:        \
    return "HIP_ERROR_" #__name;

///////////////
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wswitch"
  switch (result) {
    OSTREAM_ROCM_ERROR(InvalidValue)
    OSTREAM_ROCM_ERROR(OutOfMemory)
    OSTREAM_ROCM_ERROR(NotInitialized)
    OSTREAM_ROCM_ERROR(Deinitialized)
    OSTREAM_ROCM_ERROR(NoDevice)
    OSTREAM_ROCM_ERROR(InvalidDevice)
    OSTREAM_ROCM_ERROR(InvalidImage)
    OSTREAM_ROCM_ERROR(InvalidContext)
    OSTREAM_ROCM_ERROR(InvalidHandle)
    OSTREAM_ROCM_ERROR(NotFound)
    OSTREAM_ROCM_ERROR(NotReady)
    OSTREAM_ROCM_ERROR(NoBinaryForGpu)

    // Encountered an uncorrectable ECC error during execution.
    OSTREAM_ROCM_ERROR(ECCNotCorrectable)

    // Load/store on an invalid address. Must reboot all context.
    case 700:
      return "ROCM_ERROR_ILLEGAL_ADDRESS";
    // Passed too many / wrong arguments, too many threads for register count.
    case 701:
      return "ROCM_ERROR_LAUNCH_OUT_OF_RESOURCES";

    OSTREAM_ROCM_ERROR(ContextAlreadyInUse)
    OSTREAM_ROCM_ERROR(PeerAccessUnsupported)
    OSTREAM_ROCM_ERROR(Unknown)  // Unknown internal error to ROCM.
    default:
      return absl::StrCat("hipError_t(", static_cast<int>(result), ")");
  }
#pragma GCC diagnostic pop
}


// Returns the current context and checks that it is in the set of ROCM contexts
// created by StreamExecutor (to ensure that the ROCM runtime didn't create a
// context behind our backs).
HUcontext CurrentContext() {
  HUcontext current = ROCMDriver::CurrentContextOrDie();
  if (current != nullptr && !CreatedContexts::Has(current)) {
    LOG(FATAL) << "current context was not created by the StreamExecutor "
                  "rocm_driver API: "
               << current
               << "; a ROCM runtime call "
                  "was likely performed without using a StreamExecutor context";
  }
  return current;
}

// ROCM driver routines may require a large amount of stack (particularly
// hipModuleLoadDataEx, in our experience). To avoid stack overflow when using
// stack-limited threads (such as those spawned by a default-argument
// thread::ThreadPool on some platforms), we run certain routines in this pool
// and wait for completion.
static mutex driver_executor_threadpool_mu(LINKER_INITIALIZED);
static port::ThreadPool *InitializeDriverExecutor() {
  return new port::ThreadPool(port::Env::Default(), port::ThreadOptions(),
                              "rocm_driver", 1);
}

port::ThreadPool *GetDriverExecutor() {
  mutex_lock lock(driver_executor_threadpool_mu);
  static port::ThreadPool *thread_pool = InitializeDriverExecutor();
  return thread_pool;
}

}  // namespace

string MemorySpaceString(MemorySpace memory_space) {
  switch (memory_space) {
    case MemorySpace::kHost:
      return "host";
    case MemorySpace::kDevice:
      return "device";
    default:
      LOG(FATAL) << "impossible memory space";
  }
}

namespace {

// Call huCtxSynchronize and crash if it doesn't succeed.
void SynchronizeOrDie() {
  auto res = huCtxSynchronize();
  if (res != HIP_SUCCESS) {
    LOG(FATAL) << "Synchronize found "
               << ToString(res) << " :: " << port::CurrentStackTrace();
  }
}

struct ThreadLocalData {
  int64 id;
  RocmContext* context;  // Only valid if id == a known good context.
  int depth;
};

SE_STATIC_THREAD_LOCAL_POD(ThreadLocalData, tls_data);

}  // namespace

ScopedActivateContext::ScopedActivateContext(RocmContext* rocm_context) {

  if (FLAGS_gpuexec_rocm_sync_around_driver_calls) { SynchronizeOrDie(); }

  auto* tls = &tls_data.get();
  tls->depth++;
  if (tls->id == rocm_context->id()) {
    if (kVerifyRocmContext) {
      CHECK_EQ(CurrentContext(), rocm_context->context());
    }
    DCHECK_EQ(CurrentContext(), rocm_context->context());
    return;
  }

  VLOG(3) << "ScopedActivateContext switching context from " << tls->id
          << " to " << rocm_context->id();

  to_restore_ = (tls->depth == 1 ? nullptr : tls->context);

  // Set the context and update thread local.
  CHECK_EQ(HIP_SUCCESS, huCtxSetCurrent(rocm_context->context()));
  tls->id = rocm_context->id();
  tls->context = rocm_context;
}

ScopedActivateContext::~ScopedActivateContext() {

  if (FLAGS_gpuexec_rocm_sync_around_driver_calls) { SynchronizeOrDie(); }

  auto* tls = &tls_data.get();

  if (kVerifyRocmContext) {
    // Note that if kVerifyRocmContext is used, and contexts are deleted, it's
    // possible this could fail in the CurrentContext() call.
    CHECK_EQ(CurrentContext(),
             tls->context == nullptr ? nullptr : tls->context->context());
  }

  tls->depth--;
  DCHECK_GE(tls->depth, 0);
  if (to_restore_ == nullptr) {
    // Leave context, tls->id, and tls->context set.
    return;
  }

  // Set context and update thread local.
  CHECK_EQ(HIP_SUCCESS, huCtxSetCurrent(to_restore_->context()));
  tls->id = to_restore_->id();
  tls->context = to_restore_;
}

namespace {

// Returns a stringified device number associated with pointer, primarily for
// logging purposes. Returns "?" if the device could not be successfully
// queried.
string ROCMPointerToDeviceString(HUdeviceptr pointer) {
  auto value = ROCMDriver::GetPointerDevice(pointer);
  if (value.ok()) {
    return absl::StrCat(value.ValueOrDie());
  }
  LOG(ERROR) << "could not query device: " << value.status();
  return "?";
}

// Returns a stringified memory space associated with pointer, primarily for
// logging purposes. Returns "?" if the memory space could not be successfully
// queried.
string ROCMPointerToMemorySpaceString(HUdeviceptr pointer) {
  auto value = ROCMDriver::GetPointerMemorySpace(pointer);
  if (value.ok()) {
    return MemorySpaceString(value.ValueOrDie());
  }
  LOG(ERROR) << "could not query device: " << value.status();
  return "?";
}

// Returns a stringified representation of whether or not peer access is
// permitted between the "from" and "to" pointers' associated contexts,
// primarily for logging purposes. Returns "error" if an error is encountered
// in the process of querying.
string ROCMPointersToCanAccessString(HUdeviceptr from, HUdeviceptr to) {
  auto from_context = ROCMDriver::GetPointerContext(from);
  if (!from_context.ok()) {
    LOG(ERROR) << "could not retrieve source pointer's context: "
               << from_context.status();
    return "error";
  }
  auto to_context = ROCMDriver::GetPointerContext(to);
  if (!to_context.ok()) {
    LOG(ERROR) << "could not retrieve destination pointer's context: "
               << to_context.status();
    return "error";
  }
  return ROCMDriver::CanEnablePeerAccess(from_context.ValueOrDie(),
                                         to_context.ValueOrDie())
             ? "true"
             : "false";
}


// Actually performs the work of ROCM initialization. Wrapped up in one-time
// execution guard.
static port::Status InternalInit() {
  HUresult res = HIP_ERROR_NO_DEVICE;
  if (FLAGS_gpuexec_rocm_driver_inject_init_error) {
    LOG(ERROR) << "injecting ROCM init error; initialization will fail";
  } else {
    res = huInit(0 /* = flags */);
  }

  if (res == HIP_SUCCESS) {
    return port::Status::OK();
  }

  LOG(ERROR) << "failed call to huInit: " << ToString(res);
  Diagnostician::LogDiagnosticInformation();
  return port::Status(port::error::ABORTED,
                      absl::StrCat("failed call to huInit: ", ToString(res)));
}

}  // namespace

/* static */ port::Status ROCMDriver::Init() {
  // Cached return value from calling InternalInit(), as huInit need only be
  // called once, but ROCMDriver::Init may be called many times.
  static port::Status init_retval;
  static bool set = false;
  static mutex *init_mu = new mutex;

  mutex_lock lock(*init_mu);
  if (!set) {
    init_retval = InternalInit();
    set = true;
  }

  return init_retval;
}

/* static */ port::Status ROCMDriver::GetDevice(int device_ordinal,
                                                HUdevice *device) {
  HUresult res = huDeviceGet(device, device_ordinal);
  if (res == HIP_SUCCESS) {
    return port::Status::OK();
  }

  return port::Status(
      port::error::INTERNAL,
      absl::StrCat("failed call to huDeviceGet: ", ToString(res)));
}

/* static */ bool ROCMDriver::GetDeviceName(HUdevice device,
                                            string *device_name) {
  static const size_t kCharLimit = 64;
  absl::InlinedVector<char, 4> chars(kCharLimit);
  HUresult res = huDeviceGetName(chars.begin(), kCharLimit - 1, device);
  if (res != HIP_SUCCESS) {
    LOG(ERROR) << "failed to get device name for " << device << ": "
               << ToString(res);
    return false;
  }
  chars[kCharLimit - 1] = '\0';
  *device_name = chars.begin();
  return true;
}

bool DeviceOptionsToContextFlags(const DeviceOptions &device_options,
                                 int *flags) {
  static_assert(DeviceOptions::kMask == 0xf,
                "needs update for new device options");
  return true;
}

/* static */ port::Status ROCMDriver::CreateContext(
    HUdevice device, const DeviceOptions &device_options,
    RocmContext **context) {
  *context = nullptr;

  int flags = hipDeviceMapHost;
  if (!DeviceOptionsToContextFlags(device_options, &flags)) {
    LOG(WARNING) << "could not convert all device options into context flags";
  }

  HUresult res;
  HUcontext former_context;
  HUcontext new_context;

  unsigned int former_primary_context_flags;
  int former_primary_context_is_active;
  CHECK_EQ(HIP_SUCCESS,
           huDevicePrimaryCtxGetState(device, &former_primary_context_flags,
                                      &former_primary_context_is_active));
  if (former_primary_context_flags != flags) {
    if (former_primary_context_is_active) {
      LOG(ERROR)
          << "The primary context is active and has a different flag set ("
          << former_primary_context_flags << ") than the desired flag set ("
          << flags << ").";
    } else {
      CHECK_EQ(HIP_SUCCESS, huDevicePrimaryCtxSetFlags(device, flags));
    }
  }

  former_context = ROCMDriver::CurrentContextOrDie();
  res = huDevicePrimaryCtxRetain(&new_context, device);
  if (former_context != nullptr) {
    HUdevice former_device;
    if (huCtxGetDevice(&former_device) == HIP_SUCCESS) {
      if (former_device == device) {
        if (former_context == new_context) {
          VLOG(2) << "The primary context " << former_context << " for device "
                  << device
                  << " exists before initializing the StreamExecutor.";
        } else {
          LOG(WARNING) << "A non-primary context " << former_context
                       << " for device " << device
                       << " exists before initializing the StreamExecutor. The "
                       << "primary context is now " << new_context << ". We "
                       << "haven't verified StreamExecutor works with that.";
        }
      }
    } else {
      LOG(ERROR) << "Failed to get the device of the current context "
                 << former_context;
    }
  }
  CHECK_EQ(HIP_SUCCESS, huCtxSetCurrent(former_context));

  if (res == HIP_SUCCESS) {
    *context = CreatedContexts::Add(new_context);
    CHECK(*context != nullptr)
        << "success in this call must entail non-null result";
    VLOG(2) << "created or reused context " << context << " for this thread";
    return port::Status::OK();
  }

  string message = "failed call to huDevicePrimaryCtxRetain: " + ToString(res);
  if (res == HIP_ERROR_OUT_OF_MEMORY) {
    uint64 total_memory;
    if (GetDeviceTotalMemory(device, &total_memory)) {
      absl::StrAppend(&message, "; total memory reported: ", total_memory);
    } else {
      absl::StrAppend(&message, "; could not query total memory");
    }
  }

  return port::Status(port::error::INTERNAL, message);
}

/* static */ void ROCMDriver::DestroyContext(RocmContext* context) {
  if (context == nullptr) {
    return;
  }
  HUcontext former_context = CurrentContext();
  HUresult res = huCtxSetCurrent(context->context());
  HUdevice device;
  huCtxGetDevice(&device);
  huCtxSetCurrent(former_context);

  res = huDevicePrimaryCtxRelease(device);

  if (res != HIP_SUCCESS) {
    LOG(ERROR) << "failed to release ROCM context; leaking: " << ToString(res);
  }

  CreatedContexts::Remove(context->context());
}

/* static */ bool ROCMDriver::FuncGetAttribute(HUfunction_attribute attribute,
                                               HUfunction func,
                                               int *attribute_value) {
  // ROCM TODO properly implement this feature in HIP
  //HUresult res = huFuncGetAttribute(attribute_value, attribute, func);
  HUresult res = HIP_SUCCESS;
  if (res != HIP_SUCCESS) {
    LOG(ERROR) << "failed to query kernel attribute. kernel: " << func
               << ", attribute: " << attribute;
    return false;
  }
  return true;
}

/* static */ bool ROCMDriver::FuncSetCacheConfig(HUfunction function,
                                                 HUfunc_cache cache_config) {
  HUresult res = huFuncSetCacheConfig(function, cache_config);
  if (res != HIP_SUCCESS) {
    LOG(ERROR) << "failed to set ROCM kernel cache config. kernel: " << function
               << ", config: " << cache_config << ", result: " << ToString(res);
    return false;
  }

  return true;
}

/* static */ port::StatusOr<HUsharedconfig>
ROCMDriver::ContextGetSharedMemConfig(RocmContext* context) {
  HUsharedconfig shared_mem_config;
  ScopedActivateContext activation(context);
  HUresult result = huCtxGetSharedMemConfig(&shared_mem_config);
  if (result != HIP_SUCCESS) {
    HUdevice device;
    huCtxGetDevice(&device);
    LOG(ERROR) << "failed to get ROCM device shared memory config. "
               << "Context device ID: " << device
               << ", result: " << ToString(result);
    return port::Status(
        port::error::INTERNAL,
        absl::StrCat("failed to get shared memory config: ", ToString(result)));
  }
  return shared_mem_config;
}

/* static */ port::Status ROCMDriver::ContextSetSharedMemConfig(
    RocmContext* context, HUsharedconfig shared_mem_config) {
  ScopedActivateContext activation(context);
  HUresult result = huCtxSetSharedMemConfig(shared_mem_config);
  if (result != HIP_SUCCESS) {
    HUdevice device;
    huCtxGetDevice(&device);
    LOG(ERROR) << "failed to set ROCM device shared memory config. "
               << "Context device ID: " << device
               << ", config: " << shared_mem_config
               << ", result: " << ToString(result);
    return port::Status(
        port::error::INTERNAL,
        absl::StrCat("failed to set shared memory config: ", ToString(result)));
  }
  return port::Status::OK();
}

/* static */ bool ROCMDriver::LaunchKernel(
    RocmContext* context, HUfunction function, unsigned int grid_dim_x,
    unsigned int grid_dim_y, unsigned int grid_dim_z, unsigned int block_dim_x,
    unsigned int block_dim_y, unsigned int block_dim_z,
    unsigned int shared_mem_bytes, HUstream stream, void **kernel_params,
    void **extra) {
  ScopedActivateContext activation(context);
  VLOG(2) << "launching kernel: " << function << "; gdx: " << grid_dim_x
          << " gdy: " << grid_dim_y << " gdz: " << grid_dim_z
          << " bdx: " << block_dim_x << " bdy: " << block_dim_y
          << " bdz: " << block_dim_z << " smem: " << shared_mem_bytes;
  hipError_t res = hipModuleLaunchKernel(
     function, grid_dim_x, grid_dim_y, grid_dim_z, block_dim_x, block_dim_y,
     block_dim_z, shared_mem_bytes, stream, kernel_params, extra);
  if (res != hipSuccess) {
    LOG(ERROR) << "failed to launch ROCM kernel: " << function
               << "; result: " << ToString(res);
    return false;
  }
  VLOG(2) << "successfully launched kernel";
  return true;
}

/* static */ bool ROCMDriver::LoadHsaco(RocmContext* context,
                                        const char *hsaco_contents,
                                        hipModule_t *module) {
  port::Notification notification;
  bool ret = true;
  GetDriverExecutor()->Schedule([context, hsaco_contents, module, &ret,
                                 &notification]() {
    ScopedActivateContext activation{context};
    void *hsaco_data = const_cast<char *>(hsaco_contents);

    hipError_t res = hipModuleLoadData(module, hsaco_data);

    if (res != hipSuccess) {
      LOG(ERROR) << "failed to load HSACO: " << ToString(res);
      ret = false;
      notification.Notify();
    }

    CHECK(module != nullptr);
    notification.Notify();
  });
  notification.WaitForNotification();

  return ret;
}

/* static */ bool ROCMDriver::SynchronousMemsetUint8(RocmContext* context,
                                                     HUdeviceptr location,
                                                     uint8 value, size_t size) {
  ScopedActivateContext activation(context);
  HUresult res = huMemsetD8(location, value, size);
  if (res != HIP_SUCCESS) {
    LOG(ERROR) << "failed to memset memory: " << ToString(res);
    return false;
  }
  return true;
}

/* static */ bool ROCMDriver::SynchronousMemsetUint32(RocmContext* context,
                                                      HUdeviceptr location,
                                                      uint32 value,
                                                      size_t uint32_count) {
  ScopedActivateContext activation(context);
  HUresult res = huMemsetD32(location, value, uint32_count);
  if (res != HIP_SUCCESS) {
    LOG(ERROR) << "failed to memset memory: " << ToString(res);
    return false;
  }
  return true;
}

/* static */ bool ROCMDriver::AsynchronousMemsetUint8(RocmContext* context,
                                                      HUdeviceptr location,
                                                      uint8 value,
                                                      size_t uint32_count,
                                                      HUstream stream) {
  ScopedActivateContext activation(context);
  HUresult res = huMemsetD8Async(location, value, uint32_count, stream);
  if (res != HIP_SUCCESS) {
    LOG(ERROR) << "failed to enqueue async memset operation: " << ToString(res);
    return false;
  }
  VLOG(2) << "successfully enqueued async memset operation";
  return true;
}

/* static */ bool ROCMDriver::AsynchronousMemsetUint32(RocmContext* context,
                                                       HUdeviceptr location,
                                                       uint32 value,
                                                       size_t uint32_count,
                                                       HUstream stream) {
  ScopedActivateContext activation(context);
  HUresult res = huMemsetD32Async(location, value, uint32_count, stream);
  if (res != HIP_SUCCESS) {
    LOG(ERROR) << "failed to enqueue async memset operation: " << ToString(res);
    return false;
  }
  VLOG(2) << "successfully enqueued async memset operation";
  return true;
}

/* static */ bool ROCMDriver::AddStreamCallback(RocmContext* context,
                                                HUstream stream,
                                                StreamCallback callback,
                                                void *data) {
  HUresult res = huStreamAddCallback(stream, callback, data, 0 /* = flags */);
  if (res != HIP_SUCCESS) {
    LOG(ERROR) << "unable to add host callback: " << ToString(res);
    return false;
  }
  return true;
}

/* static */ bool ROCMDriver::GetModuleFunction(RocmContext *context,
                                                HUmodule module,
                                                const char *kernel_name,
                                                HUfunction *function) {
  ScopedActivateContext activated{context};
  CHECK(module != nullptr && kernel_name != nullptr);
  HUresult res = huModuleGetFunction(function, module, kernel_name);
  if (res != HIP_SUCCESS) {
    LOG(ERROR) << "failed to get kernel \"" << kernel_name
               << "\" from module: " << ToString(res);
    return false;
  }

  return true;
}

/* static */ bool ROCMDriver::GetModuleSymbol(RocmContext* context,
                                              HUmodule module,
                                              const char *symbol_name,
                                              HUdeviceptr *dptr,
                                              size_t *bytes) {
  ScopedActivateContext activated{context};
  CHECK(module != nullptr && symbol_name != nullptr &&
        (dptr != nullptr || bytes != nullptr));
  HUresult res = huModuleGetGlobal(dptr, bytes, module, symbol_name);
  if (res != HIP_SUCCESS) {
    // symbol may not be found in the current module, but it may reside in
    // another module.
    VLOG(2) << "failed to get symbol \"" << symbol_name
            << "\" from module: " << ToString(res);
    return false;
  }

  return true;
}

/* static */ void ROCMDriver::UnloadModule(RocmContext *context,
                                           HUmodule module) {
  ScopedActivateContext activated{context};
  HUresult res = huModuleUnload(module);
  if (res != HIP_SUCCESS) {
    LOG(ERROR) << "failed to unload module " << module
               << "; leaking: " << ToString(res);
  }
}

/* static */ port::StatusOr<HUdevice> ROCMDriver::DeviceFromContext(
    RocmContext* context) {
  ScopedActivateContext activated{context};
  HUdevice device = -1;
  HUresult result = huCtxGetDevice(&device);
  if (result == HIP_SUCCESS) {
    return device;
  }

  return port::Status(
      port::error::INTERNAL,
      absl::StrCat("failed to get device for context: ", ToString(result)));
}

/* static */ bool ROCMDriver::CreateStream(RocmContext *context,
                                           HUstream *out) {
  // TODO(leary) can we switch this to CU_STREAM_NON_BLOCKING or will that mess
  // up synchronization with respect to memsets and any other things that have
  // to occur on the default stream?
  ScopedActivateContext activated{context};
  HUresult res = huStreamCreate(out, 0);
  if (res != HIP_SUCCESS) {
    LOG(ERROR) << "could not allocate ROCM stream for context " << context
               << ": " << ToString(res);
    return false;
  }

  VLOG(2) << "successfully created stream " << *out << " for context "
          << context << " on thread";
  return true;
}

/* static */ void ROCMDriver::DestroyStream(RocmContext* context,
                                            HUstream *stream) {
  if (*stream == nullptr) {
    return;
  }

  ScopedActivateContext activated{context};
  HUresult res = huStreamDestroy(*stream);
  if (res != HIP_SUCCESS) {
    LOG(ERROR) << "failed to destroy ROCM stream for context " << context
               << ": " << ToString(res);
  } else {
    VLOG(2) << "successfully destroyed stream " << *stream << " for context "
            << context;
    *stream = nullptr;
  }
}

/* static */ void *ROCMDriver::DeviceAllocate(RocmContext *context,
                                              uint64 bytes) {
  ScopedActivateContext activated{context};
  HUdeviceptr result = 0;
  HUresult res = huMemAlloc(&result, bytes);
  if (res != HIP_SUCCESS) {
    LOG(ERROR) << "failed to allocate "
               << port::HumanReadableNumBytes::ToString(bytes) << " (" << bytes
               << " bytes) from device: " << ToString(res);
    return nullptr;
  }
  void *ptr = reinterpret_cast<void *>(result);
  VLOG(2) << "allocated " << ptr << " for context " << context << " of "
          << bytes << " bytes";
  return ptr;
}

/* static */ void ROCMDriver::DeviceDeallocate(RocmContext* context,
                                               void *location) {
  ScopedActivateContext activation(context);
  HUdeviceptr pointer = absl::bit_cast<HUdeviceptr>(location);
  HUresult res = huMemFree(pointer);
  if (res != HIP_SUCCESS) {
    LOG(ERROR) << "failed to free device memory at " << location
               << "; result: " << ToString(res);
  } else {
    VLOG(2) << "deallocated " << location << " for context " << context;
  }
}

/* static */ void *ROCMDriver::HostAllocate(RocmContext *context,
                                            uint64 bytes) {
  ScopedActivateContext activation(context);
  void *host_mem = nullptr;
  // "Portable" memory is visible to all ROCM contexts. Safe for our use model.
  HUresult res = huMemHostAlloc(&host_mem, bytes, HU_MEMHOSTALLOC_PORTABLE);
  if (res != HIP_SUCCESS) {
    LOG(ERROR) << "failed to alloc " << bytes
               << " bytes on host: " << ToString(res);
  }
  return host_mem;
}

/* static */ void ROCMDriver::HostDeallocate(RocmContext* context,
                                             void *location) {
  ScopedActivateContext activation(context);
  HUresult res = huMemFreeHost(location);
  if (res != HIP_SUCCESS) {
    LOG(ERROR) << "error deallocating host memory at " << location << ": "
               << ToString(res);
  }
}

/* static */ bool ROCMDriver::HostRegister(RocmContext* context, void *location,
                                           uint64 bytes) {
  ScopedActivateContext activation(context);
  // "Portable" memory is visible to all ROCM contexts. Safe for our use model.
  HUresult res =
      huMemHostRegister(location, bytes, HU_MEMHOSTREGISTER_PORTABLE);
  if (res != HIP_SUCCESS) {
    LOG(ERROR) << "error registering host memory at " << location << ": "
               << ToString(res);
    return false;
  }
  return true;
}

/* static */ bool ROCMDriver::HostUnregister(RocmContext* context,
                                             void *location) {
  ScopedActivateContext activation(context);
  HUresult res = huMemHostUnregister(location);
  if (res != HIP_SUCCESS) {
    LOG(ERROR) << "error unregistering host memory at " << location << ": "
               << ToString(res);
    return false;
  }
  return true;
}

/* static */ port::Status ROCMDriver::DestroyEvent(RocmContext* context,
                                                   HUevent *event) {
  if (*event == nullptr) {
    return port::Status(port::error::INVALID_ARGUMENT,
                        "input event cannot be null");
  }

  ScopedActivateContext activated{context};
  HUresult res = huEventDestroy(*event);
  *event = nullptr;

  switch (res) {
    case HIP_SUCCESS:
      return port::Status::OK();
    case HIP_ERROR_DEINITIALIZED:
    case HIP_ERROR_NOT_INITIALIZED:
      return port::Status(
          port::error::FAILED_PRECONDITION,
          port::Printf("error destroying ROCM event in context %p: %s", context,
                       ToString(res).c_str()));
    default:
      return port::Status(
          port::error::INTERNAL,
          port::Printf("error destroying ROCM event in context %p: %s", context,
                       ToString(res).c_str()));
  }
}

/* static */ port::Status ROCMDriver::RecordEvent(RocmContext* context,
                                                  HUevent event,
                                                  HUstream stream) {
  ScopedActivateContext activated{context};
  HUresult res = huEventRecord(event, stream);
  switch (res) {
    case HIP_SUCCESS:
      return port::Status::OK();
    case HIP_ERROR_DEINITIALIZED:
    case HIP_ERROR_NOT_INITIALIZED:
      return port::Status(
          port::error::FAILED_PRECONDITION,
          port::Printf("error recording ROCM event on stream %p: %s", stream,
                       ToString(res).c_str()));
    default:
      return port::Status(
          port::error::INVALID_ARGUMENT,
          port::Printf("error recording ROCM event on stream %p: %s", stream,
                       ToString(res).c_str()));
  }
}

/* static */ port::StatusOr<HUresult> ROCMDriver::QueryEvent(
    RocmContext *context, HUevent event) {
  ScopedActivateContext activated{context};
  HUresult res = huEventQuery(event);
  if (res != HIP_SUCCESS && res != HIP_ERROR_NOT_READY) {
    return port::Status(
        port::error::INTERNAL,
        port::Printf("failed to query event: %s", ToString(res).c_str()));
  }

  return res;
}

/* static */ bool ROCMDriver::GetEventElapsedTime(RocmContext* context,
                                                  float *elapsed_milliseconds,
                                                  HUevent start, HUevent stop) {
  ScopedActivateContext activated{context};
  // The stop event must have completed in order for huEventElapsedTime to
  // work.
  HUresult res = huEventSynchronize(stop);
  if (res != HIP_SUCCESS) {
    LOG(ERROR) << "failed to synchronize the stop event: " << ToString(res);
    return false;
  }
  res = huEventElapsedTime(elapsed_milliseconds, start, stop);
  if (res != HIP_SUCCESS) {
    LOG(ERROR) << "failed to get elapsed time between events: "
               << ToString(res);
    return false;
  }

  return true;
}

/* static */ bool ROCMDriver::WaitStreamOnEvent(RocmContext* context,
                                                HUstream stream,
                                                HUevent event) {
  ScopedActivateContext activation(context);
  HUresult res = huStreamWaitEvent(stream, event, 0 /* = flags */);
  if (res != HIP_SUCCESS) {
    LOG(ERROR) << "could not wait stream on event: " << ToString(res);
    return false;
  }

  return true;
}

/* static */ bool ROCMDriver::SynchronizeContext(RocmContext* context) {
  ScopedActivateContext activation(context);
  HUresult res = huCtxSynchronize();
  if (res != HIP_SUCCESS) {
    LOG(ERROR) << "could not synchronize on ROCM context: " << ToString(res)
               << " :: " << port::CurrentStackTrace();
    return false;
  }

  return true;
}

/* static */ port::Status ROCMDriver::SynchronizeStream(RocmContext *context,
                                                        HUstream stream) {
  ScopedActivateContext activated{context};
  CHECK(stream != nullptr);
  HUresult res = huStreamSynchronize(stream);
  if (res != HIP_SUCCESS) {
    port::Status status = port::InternalError(
        absl::StrCat("could not synchronize on ROCM stream: ", ToString(res)));
    LOG(ERROR) << status << " :: " << port::CurrentStackTrace();
    return status;
  }
  VLOG(2) << "successfully synchronized stream " << stream << " on context "
          << context;
  return port::Status::OK();
}

/* static */ bool ROCMDriver::IsStreamIdle(RocmContext *context,
                                           HUstream stream) {
  ScopedActivateContext activated{context};
  CHECK(stream != nullptr);
  HUresult res = huStreamQuery(stream);
  if (res == HIP_SUCCESS) {
    return true;
  }

  if (res != HIP_ERROR_NOT_READY) {
    LOG(ERROR) << "stream in bad state on status query: " << ToString(res);
  }
  return false;
}

/* static */ port::Status ROCMDriver::SynchronousMemcpyD2H(RocmContext *context,
                                                           void *host_dst,
                                                           HUdeviceptr gpu_src,
                                                           uint64 size) {
  ScopedActivateContext activation(context);
  HUresult res = huMemcpyDtoH(host_dst, gpu_src, size);
  if (res != HIP_SUCCESS) {
    return port::InternalError(
        port::Printf("failed to synchronous memcpy from device to host: %s; "
                     "host dst: %p; GPU src: %p; size: %llu=0x%llx",
                     ToString(res).c_str(), host_dst,
                     absl::bit_cast<void *>(gpu_src), size, size));
  }
  VLOG(2) << "successfully sync memcpy'd d2h of " << size << " bytes to "
          << host_dst;
  return port::Status::OK();
}

/* static */ port::Status ROCMDriver::SynchronousMemcpyH2D(RocmContext *context,
                                                           HUdeviceptr gpu_dst,
                                                           const void *host_src,
                                                           uint64 size) {
  ScopedActivateContext activation(context);
  HUresult res = huMemcpyHtoD(gpu_dst, host_src, size);
  if (res != HIP_SUCCESS) {
    return port::InternalError(port::Printf(
        "failed to synchronous memcpy from host to device: %s; GPU dst: %p;"
        " host src: %p; size: %llu=0x%llx",
        ToString(res).c_str(), absl::bit_cast<void *>(gpu_dst), host_src, size,
        size));
  }
  VLOG(2) << "successfully enqueued sync memcpy h2d of " << size << " bytes";
  return port::Status::OK();
}

/* static */ port::Status ROCMDriver::SynchronousMemcpyD2D(RocmContext *context,
                                                           HUdeviceptr gpu_dst,
                                                           HUdeviceptr gpu_src,
                                                           uint64 size) {
  ScopedActivateContext activation(context);
  HUresult res = huMemcpyDtoD(gpu_dst, gpu_src, size);
  if (res != HIP_SUCCESS) {
    return port::InternalError(port::Printf(
        "failed to synchronous memcpy from host to device: %s; GPU dst: %p; "
        "GPU src: %p; size: %llu=0x%llx",
        ToString(res).c_str(), absl::bit_cast<void *>(gpu_dst),
        absl::bit_cast<void *>(gpu_src), size, size));
  }
  VLOG(2) << "successfully sync memcpy'd d2d of " << size << " bytes";
  return port::Status::OK();
}

/* static */ bool ROCMDriver::AsynchronousMemcpyD2H(RocmContext* context,
                                                    void *host_dst,
                                                    HUdeviceptr gpu_src,
                                                    uint64 size,
                                                    HUstream stream) {
  ScopedActivateContext activation(context);
  HUresult res = huMemcpyDtoHAsync(host_dst, gpu_src, size, stream);
  if (res != HIP_SUCCESS) {
    LOG(ERROR) << port::Printf(
        "failed to enqueue async memcpy from device to host: %s; host dst: %p; "
        "GPU src: %p; size: %llu=0x%llx",
        ToString(res).c_str(), host_dst, absl::bit_cast<void *>(gpu_src), size,
        size);
    return false;
  }
  VLOG(2) << "successfully enqueued async memcpy d2h of " << size
          << " bytes from " << absl::bit_cast<void *>(gpu_src) << " to "
          << host_dst << " on stream " << stream;
  return true;
}

/* static */ bool ROCMDriver::AsynchronousMemcpyH2D(RocmContext* context,
                                                    HUdeviceptr gpu_dst,
                                                    const void *host_src,
                                                    uint64 size,
                                                    HUstream stream) {
  ScopedActivateContext activation(context);
  HUresult res = huMemcpyHtoDAsync(gpu_dst, host_src, size, stream);
  if (res != HIP_SUCCESS) {
    LOG(ERROR) << port::Printf(
        "failed to enqueue async memcpy from host to device: %s; GPU dst: %p; "
        "host src: %p; size: %llu=0x%llx",
        ToString(res).c_str(), absl::bit_cast<void *>(gpu_dst), host_src, size,
        size);
    return false;
  }
  VLOG(2) << "successfully enqueued async memcpy h2d of " << size << " bytes"
          << " on stream " << stream;
  return true;
}

/* static */ bool ROCMDriver::AsynchronousMemcpyD2D(RocmContext* context,
                                                    HUdeviceptr gpu_dst,
                                                    HUdeviceptr gpu_src,
                                                    uint64 size,
                                                    HUstream stream) {
  ScopedActivateContext activation(context);
  HUresult result = huMemcpyDtoDAsync(gpu_dst, gpu_src, size, stream);
  if (result != HIP_SUCCESS) {
    LOG(ERROR) << port::Printf(
        "failed to enqueue async memcpy from device to device: %s"
        "; GPU dst: %p on %s %s"
        "; GPU src: %p on %s %s"
        "; can access? %s; size: %llu=0x%llx",
        ToString(result).c_str(), absl::bit_cast<void *>(gpu_dst),
        ROCMPointerToMemorySpaceString(gpu_dst).c_str(),
        ROCMPointerToDeviceString(gpu_dst).c_str(),
        absl::bit_cast<void *>(gpu_src),
        ROCMPointerToMemorySpaceString(gpu_src).c_str(),
        ROCMPointerToDeviceString(gpu_src).c_str(),
        ROCMPointersToCanAccessString(gpu_src, gpu_dst).c_str(), size, size);

    return false;
  }
  VLOG(2) << "successfully enqueued async memcpy d2d of " << size << " bytes";
  return true;
}

/* static */ port::Status ROCMDriver::CreateEvent(RocmContext* context,
                                                  HUevent *result,
                                                  EventFlags flags) {
  int huflags;
  switch (flags) {
    case EventFlags::kDefault:
      huflags = HU_EVENT_DEFAULT;
      break;
    case EventFlags::kDisableTiming:
      huflags = HU_EVENT_DISABLE_TIMING; // and hipEventReleaseToSystem
      break;
    default:
      LOG(FATAL) << "impossible event flags: " << int(flags);
  }

  ScopedActivateContext activated{context};
  HUresult res = huEventCreate(result, huflags);

  if (res == HIP_SUCCESS) {
    return port::Status::OK();
  } else if (res == HIP_ERROR_OUT_OF_MEMORY) {
    return port::Status(port::error::RESOURCE_EXHAUSTED,
                        "could not create ROCM event: out of device memory");
  } else {
    return port::Status(
        port::error::FAILED_PRECONDITION,
        absl::StrCat("could not create ROCM event: ", ToString(res)));
  }
}

/* static */ int ROCMDriver::GetDeviceCount() {
  int device_count = 0;
  HUresult res = huDeviceGetCount(&device_count);
  if (res != HIP_SUCCESS) {
    LOG(ERROR) << "could not retrieve ROCM device count: " << ToString(res);
    return 0;
  }

  if (FLAGS_gpuexec_rocm_device_0_only && device_count > 1) {
    device_count = 1;
  }
  return device_count;
}

/* static */ port::StatusOr<RocmContext*> ROCMDriver::GetPointerContext(
    HUdeviceptr pointer) {
  RocmContext* context = nullptr;
  HUresult result =
      huPointerGetAttribute(&context, HU_POINTER_ATTRIBUTE_CONTEXT, pointer);
  if (result == HIP_SUCCESS) {
    CHECK(context != nullptr) << "success should entail non-null context";
    return context;
  }

  return port::Status(
      port::error::INTERNAL,
      absl::StrCat("failed to query device pointer for context: ",
                   ToString(result)));
}

/* static */ port::StatusOr<MemorySpace> ROCMDriver::GetPointerMemorySpace(
    HUdeviceptr pointer) {
  unsigned int value;
  HUresult result =
      huPointerGetAttribute(&value, HU_POINTER_ATTRIBUTE_MEMORY_TYPE, pointer);
  if (result == HIP_SUCCESS) {
    switch (value) {
      case HU_MEMORYTYPE_DEVICE:
        return MemorySpace::kDevice;
      case HU_MEMORYTYPE_HOST:
        return MemorySpace::kHost;
      default:
        return port::Status(
            port::error::INTERNAL,
            absl::StrCat("unknown memory space provided by ROCM API: ", value));
    }
  }

  return port::Status(
      port::error::INTERNAL,
      absl::StrCat("failed to query device pointer for memory space: ",
                   ToString(result)));
}

/* static */ port::Status ROCMDriver::GetPointerAddressRange(HUdeviceptr dptr,
                                                             HUdeviceptr *base,
                                                             size_t *size) {
  HUresult result = huMemGetAddressRange(base, size, dptr);
  if (result == HIP_SUCCESS) {
    return port::Status::OK();
  } else if (result == HIP_ERROR_NOT_FOUND) {
    // We differentiate between "this pointer is unknown" (return here) and
    // "there was an internal error while performing this operation" (return
    // below).
    return port::Status(
        port::error::NOT_FOUND,
        port::Printf("not a device pointer %p; %s",
                     reinterpret_cast<void *>(dptr), ToString(result).c_str()));
  }

  return port::Status(
      port::error::INTERNAL,
      port::Printf("failed to get pointer into for device pointer %p; %s",
                   reinterpret_cast<void *>(dptr), ToString(result).c_str()));
}

/* static */ port::StatusOr<HUdevice> ROCMDriver::GetPointerDevice(
    HUdeviceptr pointer) {
  auto result = GetPointerContext(pointer);
  if (!result.ok()) {
    return result.status();
  }

  return DeviceFromContext(result.ValueOrDie());
}

/* static */ port::Status ROCMDriver::GetAMDGPUISAVersion(int *version,
                                                          hipDevice_t device) {
  hipDeviceProp_t props;
  hipError_t result = hipGetDeviceProperties(&props, device);
  if (result == hipSuccess) {
    *version = props.gcnArch;
    return port::Status::OK();
  }
  *version = 0;
  return port::Status{
      port::error::INTERNAL,
      port::Printf("failed to determine AMDGPU ISA version for device %d", device)};
}

// Helper function that turns the integer output of huDeviceGetAttribute to type
// T and wraps it in a StatusOr.
template <typename T>
static port::StatusOr<T> GetSimpleAttribute(HUdevice device,
                                            HUdevice_attribute attribute) {
  int value = -1;
  HUresult result = huDeviceGetAttribute(&value, attribute, device);
  if (result != HIP_SUCCESS) {
    return port::Status(
        port::error::NOT_FOUND,
        absl::StrCat("could not retrieve ROCM device attribute (", attribute,
                     "): ", ToString(result)));
  }
  T converted = value;
  return converted;
}

/* static */ port::StatusOr<int> ROCMDriver::GetMultiprocessorCount(
    HUdevice device) {
  return GetSimpleAttribute<int>(device,
                                 hipDeviceAttributeMultiprocessorCount);
}

/* static */ port::StatusOr<int64> ROCMDriver::GetMaxSharedMemoryPerCore(
    HUdevice device) {
  return GetSimpleAttribute<int64>(
      device, hipDeviceAttributeMaxSharedMemoryPerMultiprocessor);
}

/* static */ port::StatusOr<int64> ROCMDriver::GetMaxSharedMemoryPerBlock(
    HUdevice device) {
  return GetSimpleAttribute<int64>(
      device, hipDeviceAttributeMaxSharedMemoryPerBlock);
}

/* static */ port::StatusOr<int64> ROCMDriver::GetMaxThreadsPerMultiprocessor(
    HUdevice device) {
  return GetSimpleAttribute<int64>(
      device, hipDeviceAttributeMaxThreadsPerMultiProcessor);
}

/* static */ port::StatusOr<int64> ROCMDriver::GetMaxThreadsPerBlock(
    HUdevice device) {
  return GetSimpleAttribute<int64>(device,
                                   hipDeviceAttributeMaxThreadsPerBlock);
}

/* static */ port::StatusOr<int64> ROCMDriver::GetMaxRegistersPerBlock(
    HUdevice device) {
  return GetSimpleAttribute<int64>(device,
                                   hipDeviceAttributeMaxRegistersPerBlock);
}

/* static */ port::StatusOr<int64> ROCMDriver::GetThreadsPerWarp(
    HUdevice device) {
  return GetSimpleAttribute<int64>(device, hipDeviceAttributeWarpSize);
}

/* static */ bool ROCMDriver::GetGridLimits(int *x, int *y, int *z,
                                            HUdevice device) {
  int value;
  HUresult res =
      huDeviceGetAttribute(&value, hipDeviceAttributeMaxGridDimX, device);
  if (res != HIP_SUCCESS) {
    LOG(ERROR) << "failed to query max grid dim x: " << ToString(res);
    return false;
  }
  *x = value;

  res =
      huDeviceGetAttribute(&value, hipDeviceAttributeMaxGridDimY, device);
  if (res != HIP_SUCCESS) {
    LOG(ERROR) << "failed to query max grid dim y: " << ToString(res);
    return false;
  }
  *y = value;

  res =
      huDeviceGetAttribute(&value, hipDeviceAttributeMaxGridDimZ, device);
  if (res != HIP_SUCCESS) {
    LOG(ERROR) << "failed to query max grid dim z: " << ToString(res);
    return false;
  }
  *z = value;
  return true;
}

/* static */ bool ROCMDriver::GetDriverVersion(int *driver_version) {
  HUresult res = huDriverGetVersion(driver_version);
  if (res != HIP_SUCCESS) {
    LOG(ERROR) << "failed to query driver version: " << ToString(res);
    return false;
  }

  return true;
}

/* static */ bool ROCMDriver::GetDeviceProperties(HUdevprop *device_properties,
                                                  int device_ordinal) {
  HUresult res = huDeviceGetProperties(device_properties, device_ordinal);
  if (res != HIP_SUCCESS) {
    LOG(ERROR) << "failed to query device properties: " << ToString(res);
    return false;
  }

  return true;
}

/* static */ port::StatusOr<int> ROCMDriver::GetDeviceAttribute(
    HUdevice_attribute attribute, HUdevice device) {
  int val;
  HUresult res = huDeviceGetAttribute(&val, attribute, device);
  if (res != HIP_SUCCESS) {
    return port::Status(
        port::error::INTERNAL,
        port::Printf("failed to get device attribute %d for device %d: %s",
                     attribute, device, ToString(res).c_str()));
  }
  return val;
}

/* static */ bool ROCMDriver::IsEccEnabled(HUdevice device, bool *result) {
  int value = -1;
  HUresult res = HIP_SUCCESS;
  // ROCM TODO implement this feature in HIP
  if (res != hipSuccess) {
    LOG(ERROR) << "failed to query ECC status: " << ToString(res);
    return false;
  }

  *result = value;
  return true;
}

/* static */ bool ROCMDriver::GetDeviceMemoryInfo(RocmContext* context,
                                                  int64 *free_out,
                                                  int64 *total_out) {
  ScopedActivateContext activation(context);
  size_t free = 0;
  size_t total = 0;
  HUresult res = huMemGetInfo(&free, &total);
  if (res != HIP_SUCCESS) {
    LOG(ERROR) << "failed to query device memory info: " << ToString(res);
    return false;
  }

  *free_out = free;
  *total_out = total;
  return true;
}

/* static */ bool ROCMDriver::GetDeviceTotalMemory(HUdevice device,
                                                   uint64 *result) {
  size_t value = -1;
  HUresult res = huDeviceTotalMem(&value, device);
  if (res != HIP_SUCCESS) {
    LOG(ERROR) << "failed to query total available memory: " << ToString(res);
    return false;
  }

  *result = value;
  return true;
}

/* static */ string ROCMDriver::GetPCIBusID(HUdevice device) {
  string pci_bus_id;
  static const int kBufferSize = 64;
  absl::InlinedVector<char, 4> chars(kBufferSize);
  chars[kBufferSize - 1] = '\0';
  HUresult res = huDeviceGetPCIBusId(chars.begin(), kBufferSize - 1, device);
  if (res != HIP_SUCCESS) {
    LOG(ERROR) << "failed to query PCI bus id for device: " << ToString(res);
    return pci_bus_id;
  }
  pci_bus_id = chars.begin();
  return pci_bus_id;
}

/* static */ bool ROCMDriver::CanEnablePeerAccess(RocmContext* from,
                                                  RocmContext* to) {
  if (from == to) {
    return true;  // A context can always access its own memory.
  }

  int can_access_peer = -1;
  auto from_device = DeviceFromContext(from);
  if (!from_device.ok()) {
    LOG(ERROR) << "failed to resolve 'from' peer access context to a device: "
               << from_device.status();
    return false;
  }
  auto to_device = DeviceFromContext(to);
  if (!to_device.ok()) {
    LOG(ERROR) << "failed to resolve 'to' peer access context to a device: "
               << to_device.status();
    return false;
  }
  HUresult res = huDeviceCanAccessPeer(
      &can_access_peer, from_device.ValueOrDie(), to_device.ValueOrDie());
  if (res != HIP_SUCCESS) {
    LOG(ERROR) << "failed to detect peer access capability: " << ToString(res);
    return false;
  }

  return can_access_peer;
}

/* static */ port::Status ROCMDriver::EnablePeerAccess(RocmContext* from,
                                                       RocmContext* to) {
  if (from == to) {
    return port::Status::OK();  // A context can always access its own memory.
  }

  ScopedActivateContext activated{from};
  HUresult result = huCtxEnablePeerAccess(to->context(), 0 /* = flags */);
  if (result != HIP_SUCCESS &&
      result != HIP_ERROR_PEER_ACCESS_ALREADY_ENABLED) {
    return port::Status(
        port::error::INTERNAL,
        port::Printf("failed to enable peer access from %p to %p: %s", from, to,
                     ToString(result).c_str()));
  }

  return port::Status::OK();
}

/* static */ port::StatusOr<int> ROCMDriver::GetMaxOccupiedBlocksPerCore(
    RocmContext* context, HUfunction kernel, int threads_per_block,
    size_t dynamic_shared_memory_bytes) {
  ScopedActivateContext activation(context);

  int max_blocks = 0;
  HUresult result = huOccupancyMaxActiveBlocksPerMultiprocessor(
      &max_blocks, kernel, threads_per_block, dynamic_shared_memory_bytes);
  if (result != HIP_SUCCESS) {
    return port::Status(
        port::error::INTERNAL,
        port::Printf("failed to calculate occupancy of kernel %p: %s", kernel,
                     ToString(result).c_str()));
  }

  return max_blocks;
}

/* static */ HUcontext ROCMDriver::CurrentContextOrDie() {
  HUcontext current = nullptr;
  HUresult result = huCtxGetCurrent(&current);
  if (result != HIP_SUCCESS) {
    LOG(FATAL) << "failed to query current context: " << ToString(result);
  }
  return current;
}

}  // namespace rocm
}  // namespace stream_executor

/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM

#if TENSORFLOW_USE_ROCM
#include "rocm/include/hip/hip_runtime.h"
#endif

#define EIGEN_USE_GPU

#include "tensorflow/core/kernels/random_op_gpu.h"
#include "tensorflow/core/kernels/stateful_random_ops_cpu_gpu.h"
#include "tensorflow/core/util/gpu_kernel_helper.h"
#include "tensorflow/core/util/gpu_launch_config.h"

namespace tensorflow {

using random::PhiloxRandom;
using GPUDevice = Eigen::GpuDevice;

__device__ int thread_counter;

template <typename Distribution>
__global__ void FillKernel(
    Distribution dist, int64 state_size, int64 output_size,
    StateElementType* __restrict__ state_data,
    typename Distribution::ResultElementType* __restrict__ output_data) {
  // Threads in this block share `philox`. Thread 0 is responsible for
  // initializing it.
  __shared__ char philox_raw[sizeof(PhiloxRandom)];
  auto philox = reinterpret_cast<PhiloxRandom*>(philox_raw);
  if (threadIdx.x == 0) {
    *philox = GetPhiloxRandomFromMem(state_data);
  }
  __syncthreads();
  functor::FillPhiloxRandomKernel<Distribution,
                                  Distribution::kVariableSamplesPerOutput>()
      .Run(*philox, output_data, output_size, dist);
  // The last thread updates the state.
  auto total_thread_count = gridDim.x * blockDim.x;
  auto old_counter_value = atomicAdd(&thread_counter, 1);
  if (old_counter_value == total_thread_count - 1) {
    UpdateMemWithPhiloxRandom(*philox, output_size, state_data);
  }
}

template <typename Distribution>
struct UpdateVariableAndFill_Philox<GPUDevice, Distribution> {
  void operator()(OpKernelContext* ctx, const GPUDevice& d,
                  Distribution dist, int64 output_size, int64 alg_tag_skip,
                  ScopedUnlockUnrefVar* not_used, Tensor* state_tensor,
                  typename Distribution::ResultElementType* output_data) {
    OP_REQUIRES(
        ctx, alg_tag_skip == 0,
        errors::InvalidArgument(
            "GPU kernel doesn't support reading algorithm from state variable, "
            "so alg_tag_skip must be 0; got",
            alg_tag_skip));
    auto state_tensor_flat = state_tensor->flat<StateElementType>();
    auto state_size = state_tensor_flat.size();
    auto state_data = state_tensor_flat.data();

    // maximize occupancy
    const int kGroupSize = Distribution::kResultElementCount;
    int work_element_count = (output_size + kGroupSize - 1) / kGroupSize;
    GpuLaunchConfig cfg =
        GetGpuLaunchConfig(work_element_count, d, FillKernel<Distribution>, 0, 0);

    int zero = 0;
#if GOOGLE_CUDA
    cudaMemcpyToSymbol(thread_counter, &zero, sizeof(int));
#else  // TENSORFLOW_USE_ROCM
    hipMemcpyToSymbol(HIP_SYMBOL(thread_counter), &zero, sizeof(int));
#endif
    TF_CHECK_OK(GpuLaunchKernel(
        FillKernel<Distribution>, cfg.block_count, cfg.thread_per_block, 0,
        d.stream(), dist, state_size, output_size, state_data, output_data));
  }
};

// Precondition: there is only 1 block and 1 thread.
__global__ void SkipKernel(int64 delta,
                           StateElementType* __restrict__ state_data) {
  auto philox = GetPhiloxRandomFromMem(state_data);
  UpdateMemWithPhiloxRandom(philox, delta, state_data);
}

template <>
struct RngSkip_Philox<GPUDevice> {
  void operator()(const GPUDevice& d, int64 delta, Tensor* state_tensor) {
    TF_CHECK_OK(GpuLaunchKernel(SkipKernel, 1, 1, 0, d.stream(), delta,
                                state_tensor->flat<StateElementType>().data()));
  }
};

#define REGISTER_FloatOps_GPU(TYPE) REGISTER_FloatOps(GPU, TYPE)
TF_CALL_half(REGISTER_FloatOps_GPU);
TF_CALL_float(REGISTER_FloatOps_GPU);
TF_CALL_double(REGISTER_FloatOps_GPU);

#define REGISTER_StatefulUniformInt_GPU(TYPE) \
  REGISTER_StatefulUniformInt(GPU, TYPE)
TF_CALL_int32(REGISTER_StatefulUniformInt_GPU);
TF_CALL_int64(REGISTER_StatefulUniformInt_GPU);

#define REGISTER_StatefulUniformFullInt_GPU(TYPE) \
  REGISTER_StatefulUniformFullInt(GPU, TYPE)
TF_CALL_int32(REGISTER_StatefulUniformFullInt_GPU);
TF_CALL_int64(REGISTER_StatefulUniformFullInt_GPU);
TF_CALL_uint32(REGISTER_StatefulUniformFullInt_GPU);
TF_CALL_uint64(REGISTER_StatefulUniformFullInt_GPU);

REGISTER_RngSkip(GPU);

}  // end namespace tensorflow

#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM

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

#define EIGEN_USE_THREADS

#include "tensorflow/core/kernels/random_op_cpu.h"
#include "tensorflow/core/kernels/stateful_random_ops_cpu_gpu.h"
#include "tensorflow/core/kernels/training_op_helpers.h"
#include "tensorflow/core/lib/random/random.h"

namespace tensorflow {

using CPUDevice = Eigen::ThreadPoolDevice;

template <typename Distribution>
struct UpdateVariableAndFill_Philox<CPUDevice, Distribution> {
  void operator()(OpKernelContext* ctx, const CPUDevice& device,
                  Distribution dist, int64 output_size, int64 alg_tag_skip,
                  ScopedUnlockUnrefVar* state_var_guard, Tensor* state_tensor,
                  typename Distribution::ResultElementType* output_data)
      UNLOCK_FUNCTION() {
    auto state_tensor_flat = state_tensor->flat<StateElementType>();
    auto state_data = state_tensor_flat.data();
    // Delegates to PhiloxRandom to do the actual increasing.
    auto philox = GetPhiloxRandomFromMem(state_data + alg_tag_skip);
    UpdateMemWithPhiloxRandom(philox, output_size, state_data + alg_tag_skip);
    // No longer needs the lock.
    state_var_guard->Release();
    functor::FillPhiloxRandom<CPUDevice, Distribution>()(
        ctx, device, philox, output_data, output_size, dist);
  }
};

template <>
struct RngSkip_Philox<CPUDevice> {
  void operator()(const CPUDevice& device, int64 delta, Tensor* state_tensor) {
    auto state_data = state_tensor->flat<StateElementType>().data();
    // Delegates to PhiloxRandom to do the actual increasing.
    auto philox = GetPhiloxRandomFromMem(state_data);
    UpdateMemWithPhiloxRandom(philox, delta, state_data);
  }
};

// CPU also has the deprecated 'StatefulStandardNormal' op for backward
// compatibility.
#define REGISTER_FloatOps_CPU(TYPE)                     \
  REGISTER_FloatOps(CPU, TYPE) REGISTER_KERNEL_BUILDER( \
      Name("StatefulStandardNormal")                    \
          .Device(DEVICE_CPU)                           \
          .HostMemory("resource")                       \
          .HostMemory("shape")                          \
          .TypeConstraint<TYPE>("dtype"),               \
      StatefulRandomOp<CPUDevice,                       \
                       random::NormalDistribution<PhiloxRandom, TYPE> >);

TF_CALL_half(REGISTER_FloatOps_CPU);
TF_CALL_bfloat16(REGISTER_FloatOps_CPU);
TF_CALL_float(REGISTER_FloatOps_CPU);
TF_CALL_double(REGISTER_FloatOps_CPU);

#define REGISTER_StatefulUniformInt_CPU(TYPE) \
  REGISTER_StatefulUniformInt(CPU, TYPE)
TF_CALL_int32(REGISTER_StatefulUniformInt_CPU);
TF_CALL_int64(REGISTER_StatefulUniformInt_CPU);

#define REGISTER_StatefulUniformFullInt_CPU(TYPE) \
  REGISTER_StatefulUniformFullInt(CPU, TYPE)
TF_CALL_int32(REGISTER_StatefulUniformFullInt_CPU);
TF_CALL_int64(REGISTER_StatefulUniformFullInt_CPU);
TF_CALL_uint32(REGISTER_StatefulUniformFullInt_CPU);
TF_CALL_uint64(REGISTER_StatefulUniformFullInt_CPU);

REGISTER_RngSkip(CPU);

#undef REGISTER_StatefulUniformFullInt_CPU
#undef REGISTER_StatefulUniformFullInt
#undef REGISTER_StatefulUniformInt_CPU
#undef REGISTER_StatefulUniformInt
#undef REGISTER_FloatOps_CPU
#undef REGISTER_FloatOps

#define REGISTER_NonDeterministicInts(TYPE)                   \
  REGISTER_KERNEL_BUILDER(Name("NonDeterministicInts")        \
                              .Device(DEVICE_CPU)             \
                              .HostMemory("shape")            \
                              .TypeConstraint<TYPE>("dtype"), \
                          NonDeterministicIntsOp<TYPE>);

TF_CALL_int32(REGISTER_NonDeterministicInts);
TF_CALL_uint32(REGISTER_NonDeterministicInts);
TF_CALL_int64(REGISTER_NonDeterministicInts);
TF_CALL_uint64(REGISTER_NonDeterministicInts);

#undef REGISTER_NonDeterministicInts

// TODO(wangpeng): Add RNG ops for other distributions.

}  // end namespace tensorflow

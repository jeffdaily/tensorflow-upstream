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

#ifndef TENSORFLOW_CORE_KERNELS_STATEFUL_RANDOM_OPS_CPU_GPU_H_
#define TENSORFLOW_CORE_KERNELS_STATEFUL_RANDOM_OPS_CPU_GPU_H_

#include "tensorflow/core/framework/resource_var.h"
#include "tensorflow/core/kernels/stateful_random_ops.h"
#include "tensorflow/core/kernels/training_op_helpers.h"
#include "tensorflow/core/lib/random/random.h"

namespace tensorflow {

// The following 5 functions are made templates to avoid duplicate symbols when
// linking.

// The following 2 functions use the contract "lower 32 bits for the first
// uint32, higher 32 bits for the second". Note that this is endian-neutral,
// unlike a direct memory copy `memcpy(output, &input, 8)`.
PHILOX_DEVICE_INLINE void Int64ToUint32s(int64 input, uint32* output1,
                                         uint32* output2) {
  auto u64 = static_cast<uint64>(input);
  *output1 = static_cast<uint32>(u64);
  *output2 = static_cast<uint32>(u64 >> 32);
}

PHILOX_DEVICE_INLINE int64 Uint32sToInt64(uint32 input1, uint32 input2) {
  auto u64_1 = static_cast<uint64>(input1);
  auto u64_2 = static_cast<uint64>(input2);
  return static_cast<int64>(u64_1 | (u64_2 << 32));
}

PHILOX_DEVICE_INLINE PhiloxRandom
GetPhiloxRandomFromMem(StateElementType const* ptr) {
  PhiloxRandom::ResultType counter;
  PhiloxRandom::Key key;
  Int64ToUint32s(ptr[0], &counter[0], &counter[1]);
  Int64ToUint32s(ptr[1], &counter[2], &counter[3]);
  Int64ToUint32s(ptr[2], &key[0], &key[1]);
  return PhiloxRandom(counter, key);
}

PHILOX_DEVICE_INLINE void WritePhiloxRandomToMem(PhiloxRandom const& philox,
                                                 StateElementType* ptr) {
  PhiloxRandom::ResultType const& counter = philox.counter();
  PhiloxRandom::Key const& key = philox.key();
  ptr[0] = Uint32sToInt64(counter[0], counter[1]);
  ptr[1] = Uint32sToInt64(counter[2], counter[3]);
  ptr[2] = Uint32sToInt64(key[0], key[1]);
}

PHILOX_DEVICE_INLINE void UpdateMemWithPhiloxRandom(PhiloxRandom const& philox,
                                                    int64 output_size,
                                                    StateElementType* ptr) {
  auto new_philox = philox;
  // Multiplier 256 is the same as in `FillPhiloxRandomTask`; do not change
  // it just here.
  auto delta = output_size * 256;
  new_philox.Skip(delta);  // do the actual increasing
  WritePhiloxRandomToMem(new_philox, ptr);
}

// A per-device helper function that does the actual work for
// `UpdateVariableAndFill`.
// Reason to use functor: C++ doesn't allow function-template partial
// specialization.
template <typename Device, typename Distribution>
struct UpdateVariableAndFill_Philox;

template <typename Device>
struct RngSkip_Philox;

static Status CheckState(const Tensor& state) {
  if (state.dtype() != STATE_ELEMENT_DTYPE) {
    return errors::InvalidArgument("dtype of RNG state variable must be ",
                                   DataTypeString(STATE_ELEMENT_DTYPE),
                                   ", not ", DataTypeString(state.dtype()));
  }
  if (state.dims() != 1) {
    return errors::InvalidArgument(
        "RNG state must have one and only one dimension, not ", state.dims());
  }
  return Status::OK();
}

static Status CheckPhiloxState(const Tensor& state, int64 alg_tag_skip = 0) {
  static_assert(std::is_same<StateElementType, int64>::value,
                "StateElementType must be int64");
  static_assert(std::is_same<PhiloxRandom::ResultElementType, uint32>::value,
                "PhiloxRandom::ResultElementType must be uint32");
  if (state.NumElements() < alg_tag_skip + PHILOX_MIN_STATE_SIZE) {
    return errors::InvalidArgument(
        "For the Philox algorithm, the size of state"
        " must be at least ",
        alg_tag_skip + PHILOX_MIN_STATE_SIZE, "; got ", state.NumElements());
  }
  return Status::OK();
}

template <typename Device, typename Distribution>
Status UpdateVariableAndFill(
    OpKernelContext* ctx, Distribution dist, int state_input_idx,
    bool read_alg_from_state, Algorithm alg, int64 output_size,
    typename Distribution::ResultElementType* output_data) {
  Var* var = nullptr;
  TF_RETURN_IF_ERROR(
      LookupResource(ctx, HandleFromInput(ctx, state_input_idx), &var));
  // Use `ScopedUnlockUnrefVar` here instead of `mutex_lock` and `ScopedUnref`
  // because the former supports early releasing which is needed by
  // `UpdateVariableAndFill_Philox<CPU>` to avoid holding the lock while
  // filling.
  ScopedUnlockUnrefVar state_var_guard(var);
  Tensor* var_tensor = var->tensor();
  TF_RETURN_IF_ERROR(CheckState(*var_tensor));
  auto var_tensor_flat = var_tensor->flat<StateElementType>();
  int64 alg_tag_skip = 0;
  if (read_alg_from_state) {
    alg_tag_skip = 1;
    if (var_tensor_flat.size() < 1) {
      return errors::InvalidArgument("Size of tensor must be at least 1");
    }
    alg = var_tensor_flat(0);
  }
  if (alg == RNG_ALG_PHILOX) {
    TF_RETURN_IF_ERROR(CheckPhiloxState(*var_tensor, alg_tag_skip));
    TF_RETURN_IF_ERROR(PrepareToUpdateVariable<Device, StateElementType>(
        ctx, var_tensor, var->copy_on_read_mode.load()));
    UpdateVariableAndFill_Philox<Device, Distribution>()(
        ctx, ctx->eigen_device<Device>(), dist, output_size, alg_tag_skip,
        &state_var_guard, var_tensor, output_data);
    return Status::OK();
  } else {
    return errors::InvalidArgument("Unsupported algorithm id: ", alg);
  }
}

// Preconditon: input(0) is an existing resource.
template <typename Device, class Distribution>
void StatefulRandomCompute(OpKernelContext* ctx, Distribution dist,
                           int state_input_idx, int shape_input_idx,
                           bool read_alg_from_state, Algorithm alg) {
  using T = typename Distribution::ResultElementType;
  const Tensor& shape_t = ctx->input(shape_input_idx);
  TensorShape shape;
  OP_REQUIRES_OK(ctx, ctx->op_kernel().MakeShape(shape_t, &shape));
  Tensor* output;
  OP_REQUIRES_OK(ctx, ctx->allocate_output(0, shape, &output));
  auto output_flat = output->flat<T>();
  OP_REQUIRES_OK(ctx, UpdateVariableAndFill<Device>(
                          ctx, dist, state_input_idx, read_alg_from_state, alg,
                          output_flat.size(), output_flat.data()));
}

template <typename Device, class Distribution>
class StatefulRandomOp : public OpKernel {
 public:
  explicit StatefulRandomOp(OpKernelConstruction* ctx) : OpKernel(ctx) {}

  void Compute(OpKernelContext* ctx) override {
    StatefulRandomCompute<Device>(ctx, Distribution(), 0, 1, true, 0);
  }
};

template <typename T>
Status GetScalar(const Tensor& tensor, int input_idx, T* result) {
  auto dtype = DataTypeToEnum<T>::v();
  if (tensor.dims() != 0) {
    return errors::InvalidArgument("input ", std::to_string(input_idx),
                                   " (0-based) must have shape [], not ",
                                   tensor.shape().DebugString());
  }
  if (tensor.dtype() != dtype) {
    return errors::InvalidArgument("dtype of input ", std::to_string(input_idx),
                                   " (0-based) must be ", DataTypeString(dtype),
                                   ", not ", DataTypeString(tensor.dtype()));
  }
  *result = tensor.flat<T>()(0);
  return Status::OK();
}

template <typename Device, class Distribution>
class StatefulRandomOpV2 : public OpKernel {
 public:
  explicit StatefulRandomOpV2(OpKernelConstruction* ctx) : OpKernel(ctx) {}

  void Compute(OpKernelContext* ctx) override {
    Algorithm alg;
    OP_REQUIRES_OK(ctx, GetScalar(ctx->input(1), 1, &alg));
    StatefulRandomCompute<Device>(ctx, Distribution(), /*state_input_idx=*/0,
                                  /*shape_input_idx=*/2,
                                  /*read_alg_from_state=*/false, alg);
  }
};

template <typename Device, class IntType>
class StatefulUniformIntOp : public OpKernel {
 public:
  explicit StatefulUniformIntOp(OpKernelConstruction* ctx) : OpKernel(ctx) {}

  void Compute(OpKernelContext* ctx) override {
    Algorithm alg;
    OP_REQUIRES_OK(ctx, GetScalar(ctx->input(1), 1, &alg));
    const Tensor& minval = ctx->input(3);
    const Tensor& maxval = ctx->input(4);
    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(minval.shape()),
                errors::InvalidArgument("minval must be 0-D, got shape ",
                                        minval.shape().DebugString()));
    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(maxval.shape()),
                errors::InvalidArgument("maxval must be 0-D, got shape ",
                                        maxval.shape().DebugString()));

    // Verify that minval < maxval.  This check intentionally happens after the
    // early exit for empty output.  Zero impossible things are fine.
    IntType lo = minval.scalar<IntType>()();
    IntType hi = maxval.scalar<IntType>()();
    OP_REQUIRES(
        ctx, lo < hi,
        errors::InvalidArgument("Need minval < maxval, got ", lo, " >= ", hi));

    // Build distribution
    typedef random::UniformDistribution<random::PhiloxRandom, IntType>
        Distribution;
    Distribution dist(lo, hi);

    StatefulRandomCompute<Device>(ctx, dist, /*state_input_idx=*/0,
                                  /*shape_input_idx=*/2,
                                  /*read_alg_from_state=*/false, alg);
  }
};

template <typename Device, class IntType>
class StatefulUniformFullIntOp : public OpKernel {
 public:
  explicit StatefulUniformFullIntOp(OpKernelConstruction* ctx)
      : OpKernel(ctx) {}

  void Compute(OpKernelContext* ctx) override {
    Algorithm alg;
    OP_REQUIRES_OK(ctx, GetScalar(ctx->input(1), 1, &alg));
    StatefulRandomCompute<Device>(
        ctx,
        random::UniformFullIntDistribution<random::PhiloxRandom, IntType>(),
        /*state_input_idx=*/0, /*shape_input_idx=*/2,
        /*read_alg_from_state=*/false, alg);
  }
};

template <typename Device>
class RngSkipOp : public OpKernel {
 public:
  explicit RngSkipOp(OpKernelConstruction* ctx) : OpKernel(ctx) {}

  void Compute(OpKernelContext* ctx) override {
    auto state_input_idx = 0;
    Algorithm alg;
    OP_REQUIRES_OK(ctx, GetScalar(ctx->input(1), 1, &alg));
    int64 delta;
    OP_REQUIRES_OK(ctx, GetScalar(ctx->input(2), 2, &delta));
    Var* var = nullptr;
    OP_REQUIRES_OK(
        ctx, LookupResource(ctx, HandleFromInput(ctx, state_input_idx), &var));
    ScopedUnlockUnrefVar state_var_guard(var);
    Tensor* var_tensor = var->tensor();
    OP_REQUIRES_OK(ctx, CheckState(*var_tensor));
    if (alg == RNG_ALG_PHILOX) {
      OP_REQUIRES_OK(ctx, CheckPhiloxState(*var_tensor));
      OP_REQUIRES_OK(ctx, PrepareToUpdateVariable<Device, StateElementType>(
                              ctx, var_tensor, var->copy_on_read_mode.load()));
      RngSkip_Philox<Device>()(ctx->eigen_device<Device>(), delta, var_tensor);
    } else {
      OP_REQUIRES(ctx, false,
                  errors::InvalidArgument("Unsupported algorithm id: ", alg));
    }
  }
};

template <typename T>
class NonDeterministicIntsOp : public OpKernel {
 public:
  explicit NonDeterministicIntsOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("dtype", &dtype_));
  }

  void Compute(OpKernelContext* ctx) override {
    const Tensor& shape_t = ctx->input(0);
    TensorShape shape;
    OP_REQUIRES_OK(ctx, ctx->op_kernel().MakeShape(shape_t, &shape));
    Tensor* output;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, shape, &output));
    if (shape.num_elements() == 0) return;

    switch (dtype_) {
      case DT_INT32:
      case DT_UINT32:
      case DT_INT64:
      case DT_UINT64: {
        auto output_flat = output->flat<T>();
        auto data = output_flat.data();
        for (int64 i = 0; i < output_flat.size(); ++i) {
          data[i] = static_cast<T>(random::New64());
        }
        break;
      }
      default:
        OP_REQUIRES(ctx, false,
                    errors::InvalidArgument("Unsupported dtype: ",
                                            DataTypeString(dtype_)));
    }
  }

 private:
  DataType dtype_;
};

// So far the 'Distribution' type parameter is only used when the algorithm is
// philox, so 'NormalDistribution<PhiloxRandom, ...>' is fine for now.
#define REGISTER_FloatOps(DEVICE, TYPE)                                      \
  REGISTER_KERNEL_BUILDER(                                                   \
      Name("StatefulStandardNormalV2")                                       \
          .Device(DEVICE_##DEVICE)                                           \
          .HostMemory("resource")                                            \
          .HostMemory("algorithm")                                           \
          .HostMemory("shape")                                               \
          .TypeConstraint<TYPE>("dtype"),                                    \
      StatefulRandomOpV2<DEVICE##Device,                                     \
                         random::NormalDistribution<PhiloxRandom, TYPE> >);  \
  REGISTER_KERNEL_BUILDER(                                                   \
      Name("StatefulUniform")                                                \
          .Device(DEVICE_##DEVICE)                                           \
          .HostMemory("resource")                                            \
          .HostMemory("algorithm")                                           \
          .HostMemory("shape")                                               \
          .TypeConstraint<TYPE>("dtype"),                                    \
      StatefulRandomOpV2<DEVICE##Device,                                     \
                         random::UniformDistribution<PhiloxRandom, TYPE> >); \
  REGISTER_KERNEL_BUILDER(                                                   \
      Name("StatefulTruncatedNormal")                                        \
          .Device(DEVICE_##DEVICE)                                           \
          .HostMemory("resource")                                            \
          .HostMemory("algorithm")                                           \
          .HostMemory("shape")                                               \
          .TypeConstraint<TYPE>("dtype"),                                    \
      StatefulRandomOpV2<                                                    \
          DEVICE##Device,                                                    \
          random::TruncatedNormalDistribution<                               \
              random::SingleSampleAdapter<PhiloxRandom>, TYPE> >);

#define REGISTER_StatefulUniformInt(DEVICE, TYPE)             \
  REGISTER_KERNEL_BUILDER(Name("StatefulUniformInt")          \
                              .Device(DEVICE_##DEVICE)        \
                              .HostMemory("resource")         \
                              .HostMemory("algorithm")        \
                              .HostMemory("shape")            \
                              .HostMemory("minval")           \
                              .HostMemory("maxval")           \
                              .TypeConstraint<TYPE>("dtype"), \
                          StatefulUniformIntOp<DEVICE##Device, TYPE>);

#define REGISTER_StatefulUniformFullInt(DEVICE, TYPE)         \
  REGISTER_KERNEL_BUILDER(Name("StatefulUniformFullInt")      \
                              .Device(DEVICE_##DEVICE)        \
                              .HostMemory("resource")         \
                              .HostMemory("algorithm")        \
                              .HostMemory("shape")            \
                              .TypeConstraint<TYPE>("dtype"), \
                          StatefulUniformFullIntOp<DEVICE##Device, TYPE>);

#define REGISTER_RngSkip(DEVICE)                       \
  REGISTER_KERNEL_BUILDER(Name("RngSkip")              \
                              .Device(DEVICE_##DEVICE) \
                              .HostMemory("resource")  \
                              .HostMemory("algorithm") \
                              .HostMemory("delta"),    \
                          RngSkipOp<DEVICE##Device>);

}  // end namespace tensorflow

#endif  // TENSORFLOW_CORE_KERNELS_STATEFUL_RANDOM_OPS_CPU_GPU_H_

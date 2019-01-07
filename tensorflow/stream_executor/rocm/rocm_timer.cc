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

#include "tensorflow/stream_executor/rocm/rocm_timer.h"

#include "tensorflow/stream_executor/rocm/rocm_driver.h"
#include "tensorflow/stream_executor/rocm/rocm_gpu_executor.h"
#include "tensorflow/stream_executor/rocm/rocm_stream.h"
#include "tensorflow/stream_executor/lib/status.h"

namespace stream_executor {
namespace rocm {

bool ROCMTimer::Init() {
  CHECK(start_event_ == nullptr && stop_event_ == nullptr);
  RocmContext* context = parent_->rocm_context();
  port::Status status = ROCMDriver::CreateEvent(
      context, &start_event_, ROCMDriver::EventFlags::kDefault);
  if (!status.ok()) {
    LOG(ERROR) << status;
    return false;
  }

  status = ROCMDriver::CreateEvent(context, &stop_event_,
                                   ROCMDriver::EventFlags::kDefault);
  if (!status.ok()) {
    LOG(ERROR) << status;
    status = ROCMDriver::DestroyEvent(context, &start_event_);
    if (!status.ok()) {
      LOG(ERROR) << status;
    }
    return false;
  }

  CHECK(start_event_ != nullptr && stop_event_ != nullptr);
  return true;
}

void ROCMTimer::Destroy() {
  RocmContext* context = parent_->rocm_context();
  port::Status status = ROCMDriver::DestroyEvent(context, &start_event_);
  if (!status.ok()) {
    LOG(ERROR) << status;
  }

  status = ROCMDriver::DestroyEvent(context, &stop_event_);
  if (!status.ok()) {
    LOG(ERROR) << status;
  }
}

float ROCMTimer::GetElapsedMilliseconds() const {
  CHECK(start_event_ != nullptr && stop_event_ != nullptr);
  // TODO(leary) provide a way to query timer resolution?
  // ROCM docs say a resolution of about 0.5us
  float elapsed_milliseconds = NAN;
  (void)ROCMDriver::GetEventElapsedTime(parent_->rocm_context(),
                                        &elapsed_milliseconds, start_event_,
                                        stop_event_);
  return elapsed_milliseconds;
}

bool ROCMTimer::Start(ROCMStream* stream) {
  port::Status status = ROCMDriver::RecordEvent(
      parent_->rocm_context(), start_event_, stream->rocm_stream());
  if (!status.ok()) {
    LOG(ERROR) << status;
  }
  return status.ok();
}

bool ROCMTimer::Stop(ROCMStream* stream) {
  port::Status status = ROCMDriver::RecordEvent(
      parent_->rocm_context(), stop_event_, stream->rocm_stream());
  if (!status.ok()) {
    LOG(ERROR) << status;
  }
  return status.ok();
}

}  // namespace rocm
}  // namespace stream_executor

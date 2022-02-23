/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/pjrt/pjrt_event.h"

#include <string>

#include "tensorflow/compiler/xla/pjrt/utils.h"
#include "tensorflow/core/platform/logging.h"
#include "tfrt/host_context/concurrent_work_queue.h"  // from @tf_runtime
#include "tfrt/host_context/diagnostic.h"  // from @tf_runtime
#include "tfrt/host_context/host_allocator.h"  // from @tf_runtime
#include "tfrt/host_context/host_context.h"  // from @tf_runtime

namespace xla {

PjRtEventContext PjRtEventContext::Create() {
  return PjRtEventContext(std::make_unique<tfrt::HostContext>(
      [](const tfrt::DecodedDiagnostic& diag) {
        LOG(ERROR) << "Encountered runtime error: " << diag.message << "\n";
      },
      tfrt::CreateMallocAllocator(),
      tfrt::CreateMultiThreadedWorkQueue(
          /*num_threads=*/DefaultThreadPoolSize(),
          /*num_blocking_threads=*/16)));
}

}  // namespace xla

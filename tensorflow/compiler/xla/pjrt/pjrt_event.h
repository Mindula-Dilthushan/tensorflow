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

#ifndef TENSORFLOW_COMPILER_XLA_PJRT_PJRT_EVENT_H_
#define TENSORFLOW_COMPILER_XLA_PJRT_PJRT_EVENT_H_

#include <functional>
#include <utility>

#include "absl/types/span.h"
#include "tfrt/host_context/async_value.h"  // from @tf_runtime
#include "tfrt/host_context/async_value_ref.h"  // from @tf_runtime
#include "tfrt/host_context/host_context.h"  // from @tf_runtime
#include "tfrt/support/ref_count.h"  // from @tf_runtime

namespace xla {

// Helpers for using PjRtEvents.
struct PjRtEventContext {
 public:
  // Keys that are returned by an implementation-specific handler when a client
  // starts to block on an event.
  //
  // For now, contains a single UID that can be used to identify a TraceMe, but
  // made extensible to allow support for other profilers such as endoscope.
  struct ProfilingKeys {
    uint64_t traceme_context_id = -1;
  };
  // Signature of handler called by the PjRtEvent class before it starts to
  // block a thread.
  using OnBlockStartFn = std::function<ProfilingKeys()>;
  // Signature of handler called by the PjRtEvent class after it finishes
  // blocking a thread.
  using OnBlockEndFn = std::function<void(ProfilingKeys)>;

  // Returns a context that can be used in the constructor of a PjRtEvent, for
  // clients that do not use TFRT events.
  static PjRtEventContext Create();

 private:
  template <class T>
  friend class PjRtEvent;

  explicit PjRtEventContext(std::unique_ptr<tfrt::HostContext> ctx)
      : host_ctx(std::move(ctx)) {}

  // Dummy TFRT HostContext used by PjRtEvents to call Await, for clients that
  // do not use TFRT events.
  //
  // host_ctx cannot be used for other purposes, e.g., it does not contain a
  // working thread pool so cannot enqueue work.
  std::unique_ptr<tfrt::HostContext> host_ctx;
};

// PjRtEvent<T> is a simple event that is returned by PjRt APIs that
// enqueue asynchronous work, reporting a value of type T (frequently T=Status)
// when the work is complete.
//
// PjRtEvent can be used by the client to wait for work to complete, either via
// a blocking call or a callback.
//
// The implementation wraps a TFRT AsyncValueRef<T>, but we prefer to
// encapsulate the AVR rather than returning it directly for two reasons.
//
// First, we want to retain portability in case a future implementation moves
// away from AsyncValueRef ---- we don't want clients to call arbitrary
// AsyncValueRef APIs.
//
// Second, we want to export different semantics, for
// example we block without the client supplying a HostContext, and support
// integration between blocking and profiling (e.g., TraceMe).
//
// There are two ways to construct a PjRtEvent, one used by clients that
// natively use TFRT, which already have a HostContext and import APIs for
// constructing AsyncValueRefs; and another that avoids exposing TFRT APIs and
// can be used by non-TFRT clients.
template <class T>
class PjRtEvent {
 public:
  // Wrapper for AsyncValueRef<T> that can be used by clients that don't
  // natively use TFRT.
  struct Event {
   public:
    // Creates an empty event with !this == true.
    explicit Event() = default;
    Event(Event&& other) = default;
    Event(const Event& other) : avr(other.avr.CopyRef()) {}
    Event& operator=(const Event& other) {
      avr = other.avr.CopyRef();
      return *this;
    }
    bool operator!() { return !avr; }

    // Sets the value of the event. Must be called at most once.
    //
    // After Set is called, value will be delivered to waiters on the parent
    // PjRtEvent, via blocking or callbacks.
    void Set(T value) { avr.emplace(std::move(value)); }

   private:
    friend class PjRtEvent<T>;
    explicit Event(tfrt::AsyncValueRef<T> ref) : avr(std::move(ref)) {}
    // The underlying TFRT event that can be waited on.
    tfrt::AsyncValueRef<T> avr;
  };

  // Returns an Event that can be used to construct a PjRtEvent, and then Set
  // later.
  //
  // Used by clients that do not use TFRT natively.
  static Event CreateUnSetEvent() {
    return Event(tfrt::MakeUnconstructedAsyncValueRef<T>());
  }

  // Constructor for an already-available PjRtEvent.
  //
  // Typically used to eagerly return error values when async work will not
  // be enqueued, e.g., due to invalid arguments.
  explicit PjRtEvent(T t)
      : event_(tfrt::MakeAvailableAsyncValueRef<T>(t)),
        on_block_start_([]() { return PjRtEventContext::ProfilingKeys(); }),
        on_block_end_([](PjRtEventContext::ProfilingKeys) {}),
        host_ctx_(nullptr) {}

  // Constructor used by clients that natively use TFRT and already have a
  // host_ctx that should be used for awaiting events.
  //
  // on_block_start is called before BlockHostUntilReady starts to block.
  // on_block_end is called after BlockHostUntilReady finishes blocking.
  explicit PjRtEvent(
      tfrt::HostContext* host_ctx, tfrt::AsyncValueRef<T> event,
      PjRtEventContext::OnBlockStartFn on_block_start =
          []() { return PjRtEventContext::ProfilingKeys(); },
      PjRtEventContext::OnBlockEndFn on_block_end =
          [](PjRtEventContext::ProfilingKeys) {})
      : event_(std::move(event)),
        on_block_start_(std::move(on_block_start)),
        on_block_end_(std::move(on_block_end)),
        host_ctx_(host_ctx) {}

  // Constructor used by clients that don't natively use TFRT and want to use
  // the wrapped PjRtEventContext and PjrtEvent<T>::Event classes.
  //
  // on_block_start is called before BlockHostUntilReady starts to block.
  // on_block_end is called after BlockHostUntilReady finishes blocking.
  explicit PjRtEvent(
      const PjRtEventContext& ctx, Event event,
      PjRtEventContext::OnBlockStartFn on_block_start =
          []() { return PjRtEventContext::ProfilingKeys(); },
      PjRtEventContext::OnBlockEndFn on_block_end =
          [](PjRtEventContext::ProfilingKeys) {})
      : event_(std::move(event.avr)),
        on_block_start_(std::move(on_block_start)),
        on_block_end_(std::move(on_block_end)),
        host_ctx_(ctx.host_ctx.get()) {}

  // Blocks the calling thread until the event is ready, then returns the
  // final value.
  T BlockHostUntilReady() {
    if (!event_.IsAvailable()) {
      const auto keys = on_block_start_();
      host_ctx_->Await({event_.CopyRCRef()});
      on_block_end_(keys);
    }
    DCHECK(event_.IsConcrete());
    return *event_;
  }

  // Registers callback to be called once the event is ready, with the final
  // value.
  //
  // callback may be called immediately, potentially on the calling thread.
  void OnReady(std::function<void(T)> callback) {
    event_.AndThen(
        [event = event_.CopyRef(), callback = std::move(callback)]() {
          DCHECK(event.IsConcrete());
          callback(*event);
        });
  }

 private:
  // Wrapped object to wait on.
  tfrt::AsyncValueRef<T> event_;
  // Function that is called before a thread starts blocking on the event.
  PjRtEventContext::OnBlockStartFn on_block_start_;
  // Function that is called after a thread finishes blocking on the event.
  PjRtEventContext::OnBlockEndFn on_block_end_;
  // Used only to await event_
  tfrt::HostContext* host_ctx_;  // not owned
};

}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_PJRT_PJRT_EVENT_H_

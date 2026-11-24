# Copyright 2026 Huawei Technologies Co., Ltd.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy at http://www.apache.org/licenses/LICENSE-2.0
# Distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND.
"""Real-device cases. A fresh process/session isolates every case and mode."""

import gc
import os
import time
import traceback

from pool import digest, pattern, rpc
from transport import Acl, AclBuffer, ProbeError, Session, make_options


CASES = ("acl_fixed", "torch_fixed", "local_only_fixed", "torch_dynamic", "local_only_dynamic")


class TorchBuffer:
    def __init__(self, size, torch):
        self.torch = torch
        self.tensor = torch.empty(size, dtype=torch.uint8, device="npu")
        self.addr = int(self.tensor.data_ptr())
        self.size = size

    def fill(self, data):
        import numpy as np

        host = self.torch.from_numpy(np.frombuffer(data, dtype=np.uint8).copy())
        self.tensor[:len(data)].copy_(host)

    def read(self, size):
        return self.tensor[:size].cpu().numpy().tobytes()

    def close(self):
        self.tensor = None


def memory(torch, session, acl_bytes=0):
    torch.npu.synchronize()
    return {"allocated_bytes": torch.npu.memory_allocated(),
            "reserved_bytes": torch.npu.memory_reserved(),
            "acl_client_bytes": acl_bytes,
            "active_registrations": len(session.registrations),
            "active_requests": len(session.pending)}


def ready(torch):
    # HIXL does not accept a torch stream. Publish data only after this event.
    event = torch.npu.Event()
    event.record(torch.npu.current_stream())
    event.synchronize()


def transfer_group(session, direction, buffers, remote, size, stride, asynchronous):
    start = time.perf_counter()
    requests = [session.transfer(direction, buf.addr, remote + slot * stride, size, asynchronous)
                for slot, buf in enumerate(buffers)]
    submitted = time.perf_counter()
    if asynchronous:
        session.wait(requests)
    finished = time.perf_counter()
    return {"submit_ms": (submitted - start) * 1000,
            "wait_ms": (finished - submitted) * 1000,
            "complete_ms": (finished - start) * 1000,
            "bytes": size * len(buffers)}


def client_worker(config, case, mode, remote, conn, result_path, log_path, events, killed):
    from report import append_json, redirect_logs

    redirect_logs(log_path)
    session = None
    context = {"case": case, "mode": mode, "requested_protocol": config["transport"],
               "actual_protocol": "unknown", "completion_path": "unknown",
               "application_staging": False, "library_zero_staging": "unverified"}
    buffers_held = []
    try:
        import torch
        import torch_npu  # noqa: F401
        import hixl

        torch.npu.set_device(config["devices"][0])
        torch.npu.init()
        session = Session(hixl, config["client_endpoint"], make_options(config, hixl), config["timeout"])
        asynchronous = mode == "async"
        if asynchronous and not all(hasattr(session.engine, key) for key in ("transfer_async", "get_transfer_status")):
            raise ProbeError("async API absent", unsupported=True)
        dynamic = case in ("torch_dynamic", "local_only_dynamic", "pool_exit")
        local_only = case in ("local_only_fixed", "local_only_dynamic", "pool_exit")
        width = config["inflight"] if asynchronous else 1
        stride = max(config["sizes"])
        context.update(registration_mode="local_only" if local_only else "normal",
                       registration_timing="after_connect" if dynamic else "before_connect",
                       inflight_limit=width)
        allocator = (lambda n: AclBuffer(Acl(), n)) if case == "acl_fixed" else (lambda n: TorchBuffer(n, torch))
        fixed_src, fixed_dst = [], []
        if not dynamic:
            fixed_src = [allocator(stride) for _ in range(width)]
            fixed_dst = [allocator(stride) for _ in range(width)]
            buffers_held.extend(fixed_src + fixed_dst)
            for buf in buffers_held:
                session.register(buf.addr, buf.size, local_only)
        session.connect(config["pool_endpoint"])
        append_json(result_path, {**context, "phase": "setup", "status": "PASS",
                                  "operations": session.drain_operations()})

        if case == "pool_exit":
            # Kill only the child owned by the runner. Register destination before
            # killing so an unrelated registration failure cannot satisfy this test.
            buf = allocator(config["sizes"][0])
            buffers_held.append(buf)
            session.register(buf.addr, buf.size, True)
            events.put("kill_pool")
            if not killed.wait(config["timeout"]):
                raise RuntimeError("parent did not acknowledge pool termination")
            start = time.perf_counter()
            try:
                req = session.transfer("READ", buf.addr, remote, buf.size, True)
                session.wait([req])
            except ProbeError as exc:
                append_json(result_path, {**context, "phase": "fault", "status": "PASS",
                                          "expected_failure": str(exc), "code": exc.code,
                                          "duration_ms": (time.perf_counter() - start) * 1000,
                                          "operations": session.drain_operations(),
                                          "cleanup": "isolated_process_exit; no buffer reuse"})
                os._exit(0)
            raise RuntimeError("READ reported success after pool process exited")

        for size in config["sizes"]:
            previous = set()
            for iteration in range(config["warmup"] + config["loops"]):
                context.update(size_bytes=size, iteration=iteration,
                               warmup=iteration < config["warmup"])
                start = time.perf_counter()
                before = memory(torch, session, 2 * width * stride if case == "acl_fixed" else 0)
                source = [allocator(size) for _ in range(width)] if dynamic else fixed_src
                if dynamic:
                    buffers_held.extend(source)
                    for buf in source:
                        session.register(buf.addr, size, local_only)
                seeds = [iteration * width + slot + 1 for slot in range(width)]
                for buf, seed in zip(source, seeds):
                    buf.fill(pattern(size, seed))
                ready(torch)
                source_addresses = [buf.addr for buf in source]
                live = memory(torch, session, 2 * width * stride if case == "acl_fixed" else 0)
                write = transfer_group(session, "WRITE", source, remote, size, stride, asynchronous)
                for slot, seed in enumerate(seeds):
                    verification = rpc(conn, {"op": "verify", "offset": slot * stride,
                                               "size": size, "seed": seed}, config["timeout"])
                    if not verification["match"]:
                        raise RuntimeError(f"pool WRITE mismatch: {verification}")
                # No unregister/free until WRITE and independent pool verification finish.
                if dynamic:
                    for item in source:
                        session.unregister(item.addr)
                        item.close()
                        buffers_held.remove(item)
                    source.clear()
                    gc.collect()
                released = memory(torch, session, 2 * width * stride if case == "acl_fixed" else 0)
                # Overwrite fresh allocator blocks to expose stale-address reads.
                noise = []
                if dynamic:
                    noise = [allocator(size) for _ in range(width)]
                    buffers_held.extend(noise)
                    for item in noise:
                        item.fill(bytes([0xE7]) * size)
                    ready(torch)
                target = [allocator(size) for _ in range(width)] if dynamic else fixed_dst
                if dynamic:
                    buffers_held.extend(target)
                    for item in target:
                        session.register(item.addr, size, local_only)
                for item in target:
                    item.fill(bytes([0xFF]) * size)
                ready(torch)
                target_addresses = [item.addr for item in target]
                read = transfer_group(session, "READ", target, remote, size, stride, asynchronous)
                for item, seed in zip(target, seeds):
                    if digest(item.read(size)) != digest(pattern(size, seed)):
                        raise RuntimeError("client READ mismatch")
                reused = bool(set(source_addresses) & ({item.addr for item in noise} | set(target_addresses)))
                repeated = dynamic and bool(previous & set(source_addresses))
                previous.update(source_addresses)
                if dynamic:
                    for item in target:
                        session.unregister(item.addr)
                        item.close()
                        buffers_held.remove(item)
                    for item in noise:
                        item.close()
                        buffers_held.remove(item)
                    target.clear()
                    noise.clear()
                    gc.collect()
                after = memory(torch, session, 2 * width * stride if case == "acl_fixed" else 0)
                if dynamic and (after["active_registrations"] or after["active_requests"]):
                    raise RuntimeError("dynamic case leaked registrations or requests")
                append_json(result_path, {**context, "phase": "iteration", "status": "PASS",
                                          "write_verified": True, "read_verified": True,
                                          "source_addresses": source_addresses, "target_addresses": target_addresses,
                                          "address_reused_after_release": reused, "source_reused_across_iterations": repeated,
                                          "memory_before": before, "memory_source_live": live,
                                          "memory_source_released": released, "memory_after": after,
                                          "write": write, "read": read,
                                          "end_to_end_ms": (time.perf_counter() - start) * 1000,
                                          "operations": session.drain_operations()})
        session.close()
        for item in buffers_held:
            item.close()
        buffers_held.clear()
        rpc(conn, {"op": "stop"}, config["timeout"])
        append_json(result_path, {**context, "phase": "cleanup", "status": "PASS",
                                  "active_registrations": len(session.registrations),
                                  "active_requests": len(session.pending),
                                  "operations": session.drain_operations()})
        conn.close()
    except BaseException as exc:
        traceback.print_exc()
        append_json(result_path, {**context, "phase": "failure",
                                  "status": "UNSUPPORTED" if isinstance(exc, ProbeError) and exc.unsupported else "FAIL",
                                  "error": str(exc), "code": getattr(exc, "code", None),
                                  "operations": session.drain_operations() if session else [],
                                  "cleanup": "isolated_process_exit; no uncertain buffers reused"})
        # Keep buffers_held alive until process exit. No destructive cleanup on DMA errors.
        os._exit(2 if isinstance(exc, ProbeError) and exc.unsupported else 1)

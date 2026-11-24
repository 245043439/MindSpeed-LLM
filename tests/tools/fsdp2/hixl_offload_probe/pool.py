# Copyright 2026 Huawei Technologies Co., Ltd.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy at http://www.apache.org/licenses/LICENSE-2.0
# Distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND.
"""Single-client, fixed-allocation pool and local control channel."""

import hashlib
import os
import traceback

from transport import Acl, AclBuffer, Session, make_options


def pattern(size, seed):
    # Position dependent, with a different pattern for every iteration and slot.
    # Generated and checked outside the communication timing interval.
    import numpy as np

    generator = np.random.Generator(np.random.PCG64(seed))
    return generator.integers(0, 256, size=size, dtype=np.uint8).tobytes()


def digest(data):
    return hashlib.sha256(data).hexdigest()


def rpc(conn, message, timeout):
    conn.send(message)
    if not conn.poll(timeout):
        raise TimeoutError(f"pool control timeout: {message['op']}")
    result = conn.recv()
    if not result.get("ok"):
        raise RuntimeError(f"pool control failed: {result}")
    return result


def pool_worker(config, conn, log_path, operation_path):
    from report import append_json, redirect_logs

    redirect_logs(log_path)
    session = None
    try:
        import torch
        import torch_npu  # noqa: F401
        import hixl

        torch.npu.set_device(config["devices"][1])
        torch.npu.init()
        session = Session(hixl, config["pool_endpoint"], make_options(config, hixl), config["timeout"])
        buffer = AclBuffer(Acl(), config["pool_bytes"])
        session.register(buffer.addr, buffer.size)
        append_json(operation_path, {"phase": "pool_init", "operations": session.drain_operations()})
        conn.send({"ok": True, "addr": buffer.addr, "size": buffer.size})
        while True:
            if not conn.poll(config["case_timeout"]):
                raise TimeoutError("client did not finish before pool control deadline")
            message = conn.recv()
            if message["op"] == "verify":
                size, offset = message["size"], message["offset"]
                actual = digest(buffer.read(size, offset))
                expected = digest(pattern(size, message["seed"]))
                conn.send({"ok": True, "match": actual == expected,
                           "actual_sha256": actual, "expected_sha256": expected})
            elif message["op"] == "stop":
                # Only sent after client disconnects and finalizes successfully.
                session.close()
                buffer.close()
                append_json(operation_path, {"phase": "pool_cleanup", "operations": session.drain_operations()})
                conn.send({"ok": True})
                conn.close()
                return
            else:
                raise ValueError("unknown pool control operation")
    except BaseException as exc:
        traceback.print_exc()
        append_json(operation_path, {"phase": "pool_failure", "error": str(exc),
                                     "operations": session.drain_operations() if session else []})
        try:
            conn.send({"ok": False, "error": str(exc)})
        except (OSError, EOFError):
            pass
        # Do not run Python destructors that could free buffers during DMA.
        os._exit(1)

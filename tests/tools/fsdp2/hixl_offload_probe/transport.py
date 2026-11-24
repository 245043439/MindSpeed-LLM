# Copyright 2026 Huawei Technologies Co., Ltd.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy at http://www.apache.org/licenses/LICENSE-2.0
# Distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND.
"""Small HIXL/ACL adapters. Importing this module does not initialize a device."""

import ctypes
import time


class ProbeError(RuntimeError):
    def __init__(self, stage, code=None, unsupported=False):
        self.stage = stage
        self.code = code
        self.unsupported = unsupported
        super().__init__(f"{stage}: code={code}")


class Acl:
    """Use the current torch_npu device context; do not independently finalize ACL."""

    def __init__(self):
        self.lib = ctypes.CDLL("libascendcl.so")
        signatures = {
            "aclrtMalloc": [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t, ctypes.c_int],
            "aclrtFree": [ctypes.c_void_p],
            "aclrtMemcpy": [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p,
                            ctypes.c_size_t, ctypes.c_int],
        }
        for name, args in signatures.items():
            fn = getattr(self.lib, name)
            fn.argtypes = args
            fn.restype = ctypes.c_int

    def call(self, name, *args):
        ret = getattr(self.lib, name)(*args)
        if ret != 0:
            raise ProbeError(name, ret)


class AclBuffer:
    def __init__(self, acl, size):
        self.acl = acl
        self.size = size
        pointer = ctypes.c_void_p()
        # ACL_MEM_MALLOC_HUGE_ONLY, matching the HIXL device-registration sample.
        acl.call("aclrtMalloc", ctypes.byref(pointer), size, 1)
        self.addr = pointer.value

    def fill(self, data):
        assert len(data) <= self.size
        src = ctypes.create_string_buffer(data, len(data))
        self.acl.call("aclrtMemcpy", self.addr, self.size, ctypes.cast(src, ctypes.c_void_p), len(data), 1)

    def read(self, size, offset=0):
        if offset < 0 or offset + size > self.size:
            raise ValueError("ACL read outside buffer")
        dst = ctypes.create_string_buffer(size)
        self.acl.call("aclrtMemcpy", ctypes.cast(dst, ctypes.c_void_p), size, self.addr + offset, size, 2)
        return dst.raw

    def close(self):
        if self.addr is not None:
            self.acl.call("aclrtFree", self.addr)
            self.addr = None


class Session:
    """Each instance has one completion-status owner and one peer.

    Buffers must outlive all requests. On an uncertain transfer failure the caller
    exits its isolated worker, rather than deregistering/freeing in-flight memory.
    """

    def __init__(self, module, endpoint, options, timeout):
        self.hixl = module
        self.engine = module.Hixl()
        self.timeout = timeout
        self.peer = None
        self.registrations = {}
        self.pending = set()
        self.operations = []
        self._call("initialize", endpoint, options)

    def _call(self, name, *args, **kwargs):
        start = time.perf_counter()
        value = getattr(self.engine, name)(*args, **kwargs)
        code = value[0] if isinstance(value, tuple) else value
        # finalize returns None in the Python API.
        code = 0 if code is None else int(code)
        self.operations.append({"stage": name, "code": code,
                                "duration_ms": (time.perf_counter() - start) * 1000})
        if code != int(self.hixl.SUCCESS):
            raise ProbeError(name, code, code == int(getattr(self.hixl, "UNSUPPORTED", -1)))
        return value

    def register(self, addr, size, local_only=False):
        if addr in self.registrations:
            raise ValueError("address already registered by this session")
        desc = self.hixl.MemDesc(addr, size)
        if local_only:
            if not hasattr(desc, "local_only"):
                raise ProbeError("local_only API absent", unsupported=True)
            desc.local_only = True
        _, handle = self._call("register_mem", desc, self.hixl.MemType.MEM_DEVICE)
        self.registrations[addr] = handle

    def unregister(self, addr):
        if self.pending:
            raise RuntimeError("cannot unregister while a transfer is in flight")
        self._call("deregister_mem", self.registrations[addr])
        del self.registrations[addr]

    def connect(self, peer):
        self._call("connect", peer, timeout_in_millis=int(self.timeout * 1000))
        self.peer = peer

    def transfer(self, direction, local, remote, size, asynchronous):
        op = getattr(self.hixl.TransferOp, direction)
        desc = self.hixl.TransferOpDesc(local_addr=local, remote_addr=remote, len=size)
        if asynchronous:
            _, request = self._call("transfer_async", self.peer, op, [desc])
            self.pending.add(request)
            return request
        self._call("transfer_sync", self.peer, op, [desc], timeout_in_millis=int(self.timeout * 1000))
        return None

    def wait(self, requests):
        todo = {req for req in requests if req is not None}
        if not todo.issubset(self.pending):
            raise RuntimeError("request has already been consumed or is unknown")
        deadline = time.monotonic() + self.timeout
        polls = 0
        start = time.perf_counter()
        while todo:
            for req in list(todo):
                ret, status = self.engine.get_transfer_status(req)
                polls += 1
                if int(ret) != int(self.hixl.SUCCESS):
                    self.operations.append({"stage": "get_transfer_status", "code": int(ret)})
                    raise ProbeError("get_transfer_status", int(ret))
                if status == self.hixl.TransferStatus.COMPLETED:
                    todo.remove(req)
                    self.pending.remove(req)
                elif status != self.hixl.TransferStatus.WAITING:
                    # A terminal query consumes the HIXL handle, even on failure.
                    self.pending.remove(req)
                    self.operations.append({"stage": "get_transfer_status", "code": int(ret),
                                            "status": str(status)})
                    raise ProbeError("transfer terminal status", str(status))
            if todo and time.monotonic() >= deadline:
                raise ProbeError("transfer timeout", "deadline_exceeded")
            if todo:
                time.sleep(0.0005)
        self.operations.append({"stage": "wait", "code": 0, "polls": polls,
                                "duration_ms": (time.perf_counter() - start) * 1000})

    def close(self):
        if self.pending:
            raise RuntimeError("refusing cleanup of in-flight buffers")
        if self.peer is not None:
            self._call("disconnect", self.peer, timeout_in_millis=int(self.timeout * 1000))
            self.peer = None
        for addr in list(self.registrations):
            self.unregister(addr)
        self._call("finalize")

    def drain_operations(self):
        result, self.operations = self.operations, []
        return result


def make_options(config, hixl):
    """Do not enable AutoConnect: pre/post-connect registration is under test."""
    import json

    options = {}
    for key, value in config.get("hixl_options", {}).items():
        options[key] = value
    if config["transport"] != "default":
        key = hixl.OPTION_GLOBAL_RESOURCE_CONFIG
        resource = json.loads(options.get(key, "{}"))
        resource["comm_resource_config.protocol_desc"] = [config["transport"]]
        options[key] = json.dumps(resource)
    return options

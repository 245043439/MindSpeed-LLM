# Copyright 2026 Huawei Technologies Co., Ltd.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy at http://www.apache.org/licenses/LICENSE-2.0
# Distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND.
"""Host-only tests for request ownership, cleanup, CLI and feedback artifacts."""

import json
from pathlib import Path
import tarfile
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from report import append_json, build_report, bundle, read_rows
from run_probe import parse_args
from transport import ProbeError, Session, make_options


class FakeEngine:
    def __init__(self):
        self.calls = []
        self.statuses = {}
        self.next_req = 1
        self.next_handle = 100
        self.fail_stage = None

    def initialize(self, *args):
        return 0

    def register_mem(self, desc, kind):
        self.next_handle += 1
        self.calls.append(("register", desc.addr, desc.local_only))
        return 0, self.next_handle

    def deregister_mem(self, handle):
        self.calls.append(("deregister", handle))
        return 0

    def connect(self, *args, **kwargs):
        self.calls.append(("connect",))
        return 0

    def disconnect(self, *args, **kwargs):
        self.calls.append(("disconnect",))
        return 7 if self.fail_stage == "disconnect" else 0

    def transfer_async(self, *args):
        req = self.next_req
        self.next_req += 1
        self.statuses[req] = [0, 1]
        return 0, req

    def get_transfer_status(self, req):
        values = self.statuses[req]
        value = values.pop(0) if len(values) > 1 else values[0]
        if value != 0:
            del self.statuses[req]
        return 0, value

    def finalize(self):
        self.calls.append(("finalize",))


class Desc:
    def __init__(self, addr, size):
        self.addr, self.size, self.local_only = addr, size, False


def fake_module():
    engine = FakeEngine()
    return SimpleNamespace(Hixl=lambda: engine, SUCCESS=0, UNSUPPORTED=9,
                           MemDesc=Desc, MemType=SimpleNamespace(MEM_DEVICE=0),
                           TransferOp=SimpleNamespace(WRITE=1, READ=0),
                           TransferStatus=SimpleNamespace(WAITING=0, COMPLETED=1),
                           TransferOpDesc=lambda **kwargs: kwargs,
                           OPTION_GLOBAL_RESOURCE_CONFIG="GlobalResourceConfig"), engine


class SessionTests(unittest.TestCase):
    def setUp(self):
        module, self.engine = fake_module()
        self.session = Session(module, "local", {}, 1)
        self.session.register(1000, 64, True)
        self.session.connect("pool")

    def test_completion_consumed_once_and_cleanup_order(self):
        req = self.session.transfer("WRITE", 1000, 2000, 64, True)
        with self.assertRaises(RuntimeError):
            self.session.unregister(1000)
        with self.assertRaises(RuntimeError):
            self.session.close()
        self.session.wait([req])
        with self.assertRaises(RuntimeError):
            self.session.wait([req])
        self.session.close()
        self.assertEqual([call[0] for call in self.engine.calls][-3:], ["disconnect", "deregister", "finalize"])

    def test_failed_request_is_terminal_and_not_success(self):
        req = self.session.transfer("READ", 1000, 2000, 64, True)
        self.engine.statuses[req] = [3]
        with self.assertRaises(ProbeError):
            self.session.wait([req])
        self.assertNotIn(req, self.session.pending)
        with self.assertRaises(RuntimeError):
            self.session.wait([req])

    def test_timeout_retains_inflight_ownership(self):
        req = self.session.transfer("READ", 1000, 2000, 64, True)
        self.engine.statuses[req] = [0]
        with patch("transport.time.monotonic", side_effect=[0, 2]):
            with self.assertRaises(ProbeError):
                self.session.wait([req])
        self.assertIn(req, self.session.pending)
        with self.assertRaises(RuntimeError):
            self.session.unregister(1000)

    def test_disconnect_failure_does_not_free_memory(self):
        self.engine.fail_stage = "disconnect"
        with self.assertRaises(ProbeError):
            self.session.close()
        self.assertIn(1000, self.session.registrations)
        self.assertNotIn("deregister", [call[0] for call in self.engine.calls])

    def test_address_reuse_requires_unregister(self):
        with self.assertRaises(ValueError):
            self.session.register(1000, 64)
        self.session.unregister(1000)
        self.session.register(1000, 128)
        self.assertEqual(len(self.session.registrations), 1)

    def test_missing_local_only_is_explicit(self):
        self.session.hixl.MemDesc = lambda addr, size: SimpleNamespace(addr=addr, size=size)
        with self.assertRaises(ProbeError) as raised:
            self.session.register(3000, 64, True)
        self.assertTrue(raised.exception.unsupported)


class InterfaceTests(unittest.TestCase):
    def test_cli_defaults_and_validation(self):
        with patch.dict("os.environ", {}, clear=True):
            config = parse_args(["--output", "/tmp/probe-test"])
            self.assertEqual(config["loops"], 3)
            self.assertEqual(config["devices"], [0, 1])
            self.assertEqual(config["warmup"], 0)
            for options in (["--devices", "1,1"], ["--sizes", "0"], ["--pool-mib", "1"], ["--inflight", "5"]):
                with self.assertRaises(SystemExit):
                    parse_args(["--output", "/tmp/probe-test", *options])

    def test_explicit_protocol_overrides_resource_filter_only(self):
        module, _ = fake_module()
        config = {"transport": "hccs:device", "hixl_options": {
            "GlobalResourceConfig": json.dumps({"comm_resource_config.qos": 2}), "BufferPool": "0:0"}}
        result = make_options(config, module)
        self.assertEqual(json.loads(result["GlobalResourceConfig"]), {
            "comm_resource_config.qos": 2, "comm_resource_config.protocol_desc": ["hccs:device"]})
        self.assertEqual(result["BufferPool"], "0:0")
        self.assertNotIn("AutoConnect", result)

    def test_report_archive_and_truncated_record(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            path = root / "results.jsonl"
            append_json(path, {"case": "dynamic", "mode": "async", "phase": "case_result", "status": "FAIL", "reason": "timeout"})
            with path.open("a") as stream:
                stream.write('{"unfinished":')
            self.assertEqual(read_rows(path)[-1]["status"], "FAIL")
            build_report(root)
            self.assertIn("timeout", (root / "summary.md").read_text())
            self.assertIn("unverified", (root / "summary.md").read_text())
            bundle(root)
            with tarfile.open(root / "artifacts.tar.gz") as archive:
                self.assertIn("manifest.json", archive.getnames())
                self.assertNotIn("artifacts.tar.gz", archive.getnames())


if __name__ == "__main__":
    unittest.main()

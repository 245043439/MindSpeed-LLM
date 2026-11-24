# Copyright 2026 Huawei Technologies Co., Ltd.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy at http://www.apache.org/licenses/LICENSE-2.0
# Distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND.
"""Isolated dual-NPU experiments. --help and report generation work without CANN."""

import argparse
import hashlib
import importlib.metadata
import json
import multiprocessing as mp
import os
from pathlib import Path
import platform
import queue
import socket
import subprocess
import sys
import time
import traceback

from cases import CASES, client_worker
from pool import pool_worker
from report import append_json, build_report, bundle, read_rows, redirect_logs, write_json


ENV_KEYS = ("ASCEND_RT_VISIBLE_DEVICES", "ASCEND_VISIBLE_DEVICES", "ASCEND_HOME_PATH",
            "ASCEND_LOCAL_COMM_RES", "ASCEND_GLOBAL_RESOURCE_CONFIG", "ASCEND_BUFFER_POOL",
            "HCCL_INTRA_ROCE_ENABLE", "ASCEND_ENABLE_USE_FABRIC_MEM",
            "ASCEND_GLOBAL_LOG_LEVEL", "ASCEND_SLOG_PRINT_TO_STDOUT")


def positive(value):
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--suite", choices=("smoke", "stress"), default="smoke")
    parser.add_argument("--devices", default="0,1", help="Two distinct process-visible logical device IDs")
    parser.add_argument("--transport", default="default", choices=("default", "hccs:device", "roce:device",
                                                                 "ub_ctp:device", "uboe:device", "ub_rtp:device"))
    parser.add_argument("--output", required=True, type=Path, help="New or empty output directory")
    parser.add_argument("--host", default="127.0.0.1", help="Local IPv4 address used for HIXL endpoints")
    parser.add_argument("--sizes", default="4096,131072,2097152,16777216,67108864", help="Comma-separated byte counts")
    parser.add_argument("--pool-mib", type=positive, default=256)
    parser.add_argument("--loops", type=positive)
    parser.add_argument("--inflight", type=positive, default=4)
    parser.add_argument("--timeout", type=positive, default=30, help="Seconds per transfer/control request")
    parser.add_argument("--case-timeout", type=positive, default=1800, help="Hard deadline per isolated case")
    parser.add_argument("--options-json", type=Path, help="HIXL initialization options, string-to-string JSON object")
    args = parser.parse_args(argv)
    try:
        devices = [int(item) for item in args.devices.split(",")]
        sizes = [int(item) for item in args.sizes.split(",")]
        if len(devices) != 2 or min(devices) < 0 or devices[0] == devices[1]:
            raise ValueError("devices must be two distinct nonnegative logical IDs")
        if not sizes or min(sizes) <= 0 or len(set(sizes)) != len(sizes):
            raise ValueError("sizes must be distinct positive byte counts")
        if args.inflight > 4:
            raise ValueError("this validation package supports at most 4 in-flight requests")
        if max(sizes) * args.inflight > args.pool_mib * 1024 ** 2:
            raise ValueError("pool must hold max(sizes) * inflight bytes")
        if args.case_timeout <= args.timeout:
            raise ValueError("case-timeout must exceed timeout")
        socket.inet_aton(args.host)
        options = {}
        for env, key in (("ASCEND_LOCAL_COMM_RES", "LocalCommRes"),
                         ("ASCEND_GLOBAL_RESOURCE_CONFIG", "GlobalResourceConfig"),
                         ("ASCEND_BUFFER_POOL", "BufferPool")):
            if os.getenv(env):
                options[key] = os.environ[env]
        if args.options_json:
            supplied = json.loads(args.options_json.read_text(encoding="utf-8"))
            if not isinstance(supplied, dict):
                raise ValueError("options JSON must be an object")
            options.update(supplied)
        if not all(isinstance(key, str) and isinstance(value, str) for key, value in options.items()):
            raise ValueError("HIXL options must be string-to-string")
        if options.get("AutoConnect", "0") != "0":
            raise ValueError("AutoConnect must be disabled to test registration timing")
        for key in ("LocalCommRes", "GlobalResourceConfig"):
            if key in options and not isinstance(json.loads(options[key]), dict):
                raise ValueError(f"{key} must encode a JSON object")
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    return {"preflight": args.preflight, "suite": args.suite, "devices": devices,
            "transport": args.transport, "output": str(args.output.resolve()), "host": args.host,
            "sizes": sizes, "pool_bytes": args.pool_mib * 1024 ** 2,
            "loops": args.loops or (3 if args.suite == "smoke" else 100),
            "warmup": 0 if args.suite == "smoke" else 5, "inflight": args.inflight,
            "timeout": args.timeout, "case_timeout": args.case_timeout, "hixl_options": options}


def preflight_worker(device, output, log_path):
    redirect_logs(log_path)
    result = {"device_id": device, "status": "FAIL"}
    try:
        import torch
        import torch_npu
        import hixl
        from transport import Acl

        torch.npu.set_device(device)
        torch.npu.init()
        acl = Acl()
        engine = hixl.Hixl()
        required = ("initialize", "register_mem", "deregister_mem", "connect", "disconnect", "transfer_sync", "finalize")
        missing = [method for method in required if not hasattr(engine, method)]
        if missing:
            raise RuntimeError(f"missing HIXL API: {missing}")
        result.update(status="PASS", device_name=torch.npu.get_device_name(device),
                      torch_version=torch.__version__, torch_npu_version=torch_npu.__version__,
                      hixl_module=hixl.__file__, hixl_version=getattr(hixl, "__version__", "unknown"),
                      local_only=hasattr(hixl.MemDesc(0, 1), "local_only"),
                      async_api=all(hasattr(engine, x) for x in ("transfer_async", "get_transfer_status")),
                      acl_library=acl.lib._name)
        maps = Path("/proc/self/maps")
        if maps.exists():
            result["loaded_runtime_libraries"] = sorted({line.split()[-1] for line in maps.read_text().splitlines()
                                                        if any(key in line for key in ("libhcomm", "libcann_hixl", "libascendcl", "libadxl"))})
    except BaseException as exc:
        result["error"] = str(exc)
        traceback.print_exc()
    write_json(output, result)


def stop(process):
    if process is not None and process.pid is not None:
        if process.is_alive():
            process.terminate()
        process.join(5)
        if process.is_alive():
            process.kill()
            process.join(5)


def collect_environment(ctx, config, root):
    info = {"python": sys.version, "platform": platform.platform(),
            "environment": {key: os.environ[key] for key in ENV_KEYS if key in os.environ},
            "devices": [], "packages": {}, "runtime_version_notes": "Unreported versions remain unknown; no source SHA inferred from installed binaries."}
    info["probe_source_sha256"] = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(Path(__file__).resolve().parent.glob("*.py"))
    }
    for package in ("torch", "torch-npu", "hixl"):
        try:
            info["packages"][package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            info["packages"][package] = "not found in package metadata"
    try:
        result = subprocess.run(["npu-smi", "info"], capture_output=True, text=True, timeout=15, check=False)
        info["npu_smi"] = {"code": result.returncode, "stdout": result.stdout[:20000], "stderr": result.stderr[:4000]}
    except (OSError, subprocess.TimeoutExpired) as exc:
        info["npu_smi"] = {"error": str(exc)}
    info["version_files"] = {}
    for name in ("/usr/local/Ascend/ascend-toolkit/latest/version.cfg", "/usr/local/Ascend/cann/version.cfg",
                 "/usr/local/Ascend/driver/version.info"):
        try:
            info["version_files"][name] = Path(name).read_text()[:16000]
        except OSError:
            pass
    for device in config["devices"]:
        path = root / f"preflight_device_{device}.json"
        child = ctx.Process(target=preflight_worker, args=(device, str(path), str(root / "client.log")))
        try:
            child.start()
            child.join(config["timeout"])
            if child.is_alive():
                data = {"device_id": device, "status": "FAIL", "error": "preflight deadline exceeded"}
            elif path.exists():
                data = json.loads(path.read_text())
            else:
                data = {"device_id": device, "status": "FAIL", "error": f"preflight exited {child.exitcode} without result"}
        finally:
            stop(child)
        info["devices"].append(data)
    write_json(root / "environment.json", info)
    okay = all(item["status"] == "PASS" for item in info["devices"])
    append_json(root / "results.jsonl", {"phase": "preflight", "status": "PASS" if okay else "FAIL",
                                         "error": "" if okay else "see environment.json and client.log"})
    return okay


def endpoints(host):
    sockets = []
    try:
        for _ in range(2):
            sock = socket.socket()
            sock.bind((host, 0))
            sockets.append(sock)
        return [f"{host}:{sock.getsockname()[1]}" for sock in sockets]
    finally:
        for sock in sockets:
            sock.close()


def run_case(ctx, config, case, mode, root):
    case_dir = root / "cases" / f"{case}_{mode}"
    case_dir.mkdir(parents=True)
    result_path = case_dir / "results.jsonl"
    cfg = dict(config)
    cfg["client_endpoint"], cfg["pool_endpoint"] = endpoints(config["host"])
    write_json(case_dir / "config.json", cfg)
    client_conn, pool_conn = ctx.Pipe()
    events = ctx.Queue()
    killed = ctx.Event()
    pool_process = ctx.Process(target=pool_worker, args=(cfg, pool_conn, str(case_dir / "pool.log"),
                                                        str(case_dir / "pool_operations.jsonl")))
    client = None
    deadline = time.monotonic() + cfg["case_timeout"]
    failure = None
    try:
        pool_process.start()
        pool_conn.close()
        if not client_conn.poll(cfg["timeout"]):
            raise TimeoutError("pool initialization deadline exceeded")
        remote = client_conn.recv()
        if not remote.get("ok"):
            raise RuntimeError(f"pool initialization failed: {remote}")
        client = ctx.Process(target=client_worker, args=(cfg, case, mode, remote["addr"], client_conn,
                                                        str(result_path), str(case_dir / "client.log"), events, killed))
        client.start()
        client_conn.close()
        while client.is_alive():
            try:
                event = events.get(timeout=0.2)
                if event == "kill_pool" and case == "pool_exit":
                    stop(pool_process)
                    if pool_process.is_alive():
                        raise RuntimeError("could not terminate owned pool process")
                    killed.set()
            except queue.Empty:
                pass
            if time.monotonic() >= deadline:
                raise TimeoutError("case hard deadline exceeded; worker forcibly terminated")
            client.join(0)
        if client.exitcode == 0 and case != "pool_exit":
            pool_process.join(cfg["timeout"])
            if pool_process.is_alive() or pool_process.exitcode != 0:
                raise RuntimeError(f"pool cleanup failed, exit code {pool_process.exitcode}")
    except (Exception, KeyboardInterrupt) as exc:
        failure = str(exc) or "interrupted"
        if isinstance(exc, KeyboardInterrupt):
            failure = "interrupted"
    finally:
        # Stop transfer initiator before pool; never reuse an uncertain session.
        stop(client)
        if pool_process.pid is not None:
            stop(pool_process)
        client_conn.close()
        pool_conn.close()
        events.close()
        events.join_thread()
    rows = read_rows(result_path)
    if failure:
        status, reason = "FAIL", failure
    elif any(row["status"] == "FAIL" for row in rows):
        status, reason = "FAIL", "worker failure; see case results and logs"
    elif any(row["status"] == "UNSUPPORTED" for row in rows):
        status, reason = "UNSUPPORTED", "API explicitly unavailable/unsupported"
    elif client is not None and client.exitcode == 0 and any(row.get("phase") in ("cleanup", "fault") for row in rows):
        status, reason = "PASS", ""
    else:
        status, reason = "FAIL", f"worker exited without completion, code={client.exitcode if client else None}"
    for row in rows:
        append_json(root / "results.jsonl", row)
    append_json(root / "results.jsonl", {"case": case, "mode": mode, "phase": "case_result",
                                         "status": status, "reason": reason})
    for role in ("client", "pool"):
        source = case_dir / f"{role}.log"
        with (root / f"{role}.log").open("a", encoding="utf-8") as stream:
            stream.write(f"\n=== {case}/{mode} ===\n")
            if source.exists():
                stream.write(source.read_text(errors="replace"))
    print(f"{case}/{mode}: {status} {reason}", flush=True)
    if failure == "interrupted":
        raise KeyboardInterrupt
    return status


def main(argv=None):
    config = parse_args(argv)
    root = Path(config["output"])
    if root.exists() and any(root.iterdir()):
        raise SystemExit("Output directory must be empty; use a new run directory.")
    root.mkdir(parents=True, exist_ok=True)
    write_json(root / "config.json", config)
    for name in ("results.jsonl", "client.log", "pool.log"):
        (root / name).touch()
    ctx = mp.get_context("spawn")
    code = 0
    try:
        if not collect_environment(ctx, config, root):
            return 1
        if config["preflight"]:
            return 0
        outcomes = {}
        for mode in ("sync", "async"):
            for case in CASES:
                if case != "acl_fixed" and outcomes.get(("acl_fixed", "sync")) != "PASS":
                    status = "SKIP"
                    append_json(root / "results.jsonl", {"case": case, "mode": mode, "phase": "case_result",
                                                         "status": status, "reason": "fixed ACL sync baseline failed"})
                else:
                    status = run_case(ctx, config, case, mode, root)
                outcomes[(case, mode)] = status
        if config["suite"] == "stress":
            if outcomes.get(("local_only_dynamic", "async")) == "PASS":
                outcomes[("pool_exit", "async")] = run_case(ctx, config, "pool_exit", "async", root)
            else:
                append_json(root / "results.jsonl", {"case": "pool_exit", "mode": "async", "phase": "case_result",
                                                     "status": "SKIP", "reason": "requires successful dynamic local_only async case"})
        code = 1 if any(value == "FAIL" for value in outcomes.values()) else (2 if any(value != "PASS" for value in outcomes.values()) else 0)
    except KeyboardInterrupt:
        append_json(root / "results.jsonl", {"case": "runner", "phase": "case_result", "status": "FAIL", "error": "interrupted"})
        code = 130
    except Exception as exc:
        traceback.print_exc()
        append_json(root / "results.jsonl", {"case": "runner", "phase": "case_result", "status": "FAIL", "error": str(exc)})
        code = 1
    finally:
        build_report(root)
        bundle(root)
        print(f"Feedback bundle: {root / 'artifacts.tar.gz'}", flush=True)
    return code


if __name__ == "__main__":
    sys.exit(main())

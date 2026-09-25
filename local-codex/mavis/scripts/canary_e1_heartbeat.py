#!/usr/bin/env python3
"""Minimal host canary for the E1 lease-heartbeat and process-group runtime repair.

This is a canary, not an acceptance run. It exercises the repaired code paths
against the real host -- the shared ``gpu-lease`` tool, IRIS's own status
endpoint, and a real unreaped process group -- but it never loads a model and
never certifies an installed revision. Its job is to show, with host evidence,
that the E1 heartbeat and the trial process-group stop survive the conditions
that aborted the paired E1 trial before it could produce a valid comparison.

Parts:

* ``iris_parity``    -- the monitor's IRIS probe reads the same endpoint the
  lease tool reads, and agrees with a direct read of it.
* ``eperm_stop``     -- the real macOS condition that failed the trial (a
  ``killpg`` that answers EPERM for an exited, unreaped leader alone in its
  group) is reproduced, and the repaired stop returns cleanly; a live leader in
  its own group is still stopped.
* ``fail_closed``    -- a missing lease, a foreign lease purpose, and an
  unanswerable IRIS all still refuse.
* ``sustained_monitor`` -- the monitor runs for a long window at the trial's
  cadence, alongside the slow ``gpu-lease status`` command, and with the
  ``gpu-lease`` binary deliberately unreachable, proving the safety path does
  not depend on that command.

Usage::

    PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=<source> python3.14 \
        scripts/canary_e1_heartbeat.py --out <receipt.json> [--seconds 90]
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path
from uuid import uuid4

from mavis import e1_gpu, runtime

IRIS_STATUS_DEFAULT = "http://127.0.0.1:8117/status"
MONITOR_CADENCE = 0.2
MONITOR_LATENCY_BOUND = 1.5
CANARY_SECONDS = 90


class _SkipMonitor(Exception):
    """Raised when the revision under canary has no bounded monitor to exercise."""


def _monotonic_call(work) -> tuple[float, BaseException | None]:
    started = time.monotonic()
    try:
        work()
    except BaseException as error:  # the canary records, it never hides
        return round(time.monotonic() - started, 3), error
    return round(time.monotonic() - started, 3), None


def _direct_session_active(url: str) -> bool | None:
    """Read the IRIS session flag without any Mavis code in the path."""
    try:
        with urllib.request.urlopen(url, timeout=1) as response:
            return bool(json.load(response).get("sessionActive"))
    except (OSError, ValueError):
        return None


def part_iris_parity() -> dict:
    url = runtime._origin("http://127.0.0.1:8117/v1") + "/status"
    probed = runtime.iris_voice_session_active(timeout=1.0)
    direct = _direct_session_active(url)
    return {
        "monitor_endpoint": url,
        "gpu_lease_default_endpoint": os.environ.get("GPU_LEASE_IRIS_STATUS",
                                                     IRIS_STATUS_DEFAULT),
        "monitor_probe": probed,
        "direct_probe": direct,
        "parity": probed is not None and probed is direct,
        "pass": url == IRIS_STATUS_DEFAULT and probed is not None and probed is direct,
    }


def _await_unreaped_exit(process: subprocess.Popen, timeout: float = 5.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        exited = runtime._child_exit_unreaped(process)
        if exited is not None:
            return exited
        time.sleep(0.05)
    raise RuntimeError("canary child did not exit within the wait window")


def _zombie_group() -> subprocess.Popen:
    """A real exited, unreaped group leader, exactly like a stopped trial server."""
    process = subprocess.Popen([sys.executable, "-c", "import sys; sys.exit(0)"],
                               start_new_session=True)
    _await_unreaped_exit(process)
    return process


def part_eperm_stop() -> dict:
    zombie = _zombie_group()
    workers = runtime._process_group_workers(zombie.pid)
    direct = "signalled"
    try:
        os.killpg(zombie.pid, signal.SIGTERM)
    except ProcessLookupError:
        direct = "ESRCH"
    except PermissionError:
        direct = "EPERM"
    latency_ms, error = _monotonic_call(
        lambda: runtime._stop_spawned_process_group(zombie, timeout=5.0)
    )
    stopped = {
        "direct_killpg_on_zombie_group": direct,
        "zombie_group_workers": sorted(workers),
        "repaired_stop_seconds": latency_ms,
        "repaired_stop_error": repr(error) if error is not None else None,
        "reaped_returncode": zombie.returncode,
    }
    live = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"],
                            start_new_session=True)
    time.sleep(0.2)
    live_latency_ms, live_error = _monotonic_call(
        lambda: runtime._stop_spawned_process_group(live, timeout=5.0)
    )
    stopped.update({
        "live_leader_stop_seconds": live_latency_ms,
        "live_leader_stop_error": repr(live_error) if live_error is not None else None,
        "live_leader_returncode": live.returncode,
    })
    stopped["pass"] = (
        error is None and zombie.returncode == 0
        and live_error is None and live.returncode is not None and live.returncode != 0
    )
    return stopped


def _expect_refusal(heartbeat, label: str) -> dict:
    _, error = _monotonic_call(heartbeat.monitor)
    return {"case": label, "raised": isinstance(error, RuntimeError),
            "error": repr(error) if error is not None else None}


def part_fail_closed(heartbeat, purpose: str) -> dict:
    cases = []
    foreign = e1_gpu._LeaseHeartbeat(purpose + ":foreign")
    cases.append(_expect_refusal(foreign, "foreign lease purpose"))
    previous = os.environ.get("MAVIS_IRIS_VOICE_STATUS")
    os.environ["MAVIS_IRIS_VOICE_STATUS"] = "http://127.0.0.1:9/v1"
    try:
        cases.append(_expect_refusal(heartbeat, "unanswerable IRIS"))
    finally:
        if previous is None:
            os.environ.pop("MAVIS_IRIS_VOICE_STATUS", None)
        else:
            os.environ["MAVIS_IRIS_VOICE_STATUS"] = previous
    cases.append({
        "case": "IRIS probe restored",
        "raised": False,
        "error": None,
        "recovered": _monotonic_call(heartbeat.monitor)[1] is None,
    })
    return {"cases": cases, "pass": all(case["raised"] for case in cases[:2])
            and all(case.get("recovered", True) for case in cases)}


def _slow_status_probe(stop: threading.Event, samples: list) -> None:
    while not stop.is_set():
        latency, error = _monotonic_call(lambda: e1_gpu._lease_command("status"))
        samples.append({"seconds": latency,
                        "error": repr(error) if error is not None else None})
        stop.wait(1.0)


def part_sustained_monitor(heartbeat, seconds: float) -> dict:
    stop = threading.Event()
    status_samples: list = []
    loader = threading.Thread(target=_slow_status_probe, args=(stop, status_samples),
                              name="canary-slow-status", daemon=True)
    loader.start()
    # The pre-repair safety path (`heartbeat(force=True)`, what `_monitored_load`
    # called) runs the whole slow command inline. Measure it for comparison.
    force_latencies = []
    for _ in range(3):
        force_latencies.append(_monotonic_call(lambda: heartbeat(force=True))[0])
        time.sleep(0.5)
    latencies: list[float] = []
    errors: list[str] = []
    iterations = 0
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        latency, error = _monotonic_call(heartbeat.monitor)
        latencies.append(latency)
        if error is not None:
            errors.append(repr(error))
        iterations += 1
        time.sleep(MONITOR_CADENCE)
    stop.set()
    loader.join()

    # The safety path must not run the slow command: break the binary, keep moving.
    saved = e1_gpu.LEASE
    e1_gpu.LEASE = Path("/nonexistent/canary/gpu-lease")
    try:
        blind_latency, blind_error = _monotonic_call(heartbeat.monitor)
    finally:
        e1_gpu.LEASE = saved

    ordered = sorted(latencies)
    status_latencies = sorted(sample["seconds"] for sample in status_samples)
    return {
        "window_seconds": seconds,
        "iterations": iterations,
        "monitor_errors": errors[:5],
        "monitor_error_count": len(errors),
        "monitor_seconds_max": ordered[-1] if ordered else None,
        "monitor_seconds_p95": ordered[int(len(ordered) * 0.95) - 1] if ordered else None,
        "monitor_seconds_min": ordered[0] if ordered else None,
        "monitor_latency_bound": MONITOR_LATENCY_BOUND,
        "prerepair_force_seconds": sorted(force_latencies),
        "lease_status_samples": len(status_samples),
        "lease_status_seconds_max": status_latencies[-1] if status_latencies else None,
        "lease_status_errors": [s["error"] for s in status_samples if s["error"]][:5],
        "monitor_without_lease_binary_seconds": blind_latency,
        "monitor_without_lease_binary_error": (
            repr(blind_error) if blind_error is not None else None
        ),
        "pass": (not errors and iterations > 0
                 and (ordered[-1] <= MONITOR_LATENCY_BOUND if ordered else False)
                 and blind_error is None),
    }


def _host_facts(checkout: Path) -> dict:
    def git(*args: str) -> str:
        return subprocess.run(["git", "-C", str(checkout), *args], capture_output=True,
                              text=True, check=False).stdout.strip()

    return {
        "canary_python": sys.version.split()[0],
        "canary_python_waitid": hasattr(os, "waitid"),
        "checkout": str(checkout),
        "revision": git("rev-parse", "HEAD"),
        "branch": git("rev-parse", "--abbrev-ref", "HEAD"),
        "worktree_dirty": bool(git("status", "--porcelain=v1")),
        "mavis_runtime_module": runtime.__file__,
        "mavis_e1_gpu_module": e1_gpu.__file__,
        "has_monitor": hasattr(e1_gpu._LeaseHeartbeat, "monitor"),
        "has_signal_spawned_group": hasattr(runtime, "_signal_spawned_group"),
        "iris_endpoint": "http://127.0.0.1:8000/v1",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--checkout", default=str(Path(__file__).resolve().parents[3]))
    parser.add_argument("--seconds", type=float, default=CANARY_SECONDS)
    args = parser.parse_args()

    checkout = Path(args.checkout).resolve()
    run_id = uuid4().hex
    receipt = {
        "schema_version": "mavis.e1-heartbeat-canary/v1",
        "run_id": run_id,
        "purpose": f"e1-canary-heartbeat:{run_id}",
        "started_epoch": time.time(),
        "host": _host_facts(checkout),
        "parts": {},
    }
    lease_holder, lease_released = None, False
    monitor_present = receipt["host"]["has_monitor"]
    try:
        if hasattr(runtime, "iris_voice_session_active"):
            receipt["parts"]["iris_parity"] = part_iris_parity()
            if not receipt["parts"]["iris_parity"]["pass"]:
                raise RuntimeError("IRIS parity failed; refusing to canary the monitor")
        else:
            receipt["parts"]["iris_parity"] = {
                "status": "skipped",
                "reason": "this revision has no iris_voice_session_active probe",
            }
        receipt["parts"]["eperm_stop"] = part_eperm_stop()

        if not monitor_present:
            receipt["parts"]["monitor_parts"] = {
                "status": "skipped",
                "reason": "this revision has no _LeaseHeartbeat.monitor to canary",
            }
            raise _SkipMonitor

        pre_acquire = e1_gpu._LeaseHeartbeat(receipt["purpose"])
        receipt["parts"]["missing_lease"] = _expect_refusal(pre_acquire, "missing lease")

        status = e1_gpu._lease_command("status")
        if "the GPU is free" not in status:
            receipt["status"] = "lease-busy"
            receipt["lease_status"] = status.splitlines()[0]
            raise SystemExit(3)
        e1_gpu._lease_command("acquire", e1_gpu.HOLDER, receipt["purpose"],
                              e1_gpu.LEASE_MINUTES)
        lease_holder = e1_gpu.HOLDER
        heartbeat = e1_gpu._LeaseHeartbeat(receipt["purpose"])
        heartbeat(force=True)
        iris_models = runtime.loaded_generation_models("http://127.0.0.1:8000/v1",
                                                       timeout=1.0)
        heartbeat.iris_endpoint = "http://127.0.0.1:8000/v1"
        heartbeat.iris_models = iris_models
        receipt["iris_models"] = iris_models

        receipt["parts"]["fail_closed"] = part_fail_closed(heartbeat, receipt["purpose"])
        receipt["parts"]["sustained_monitor"] = part_sustained_monitor(
            heartbeat, args.seconds
        )
    except _SkipMonitor:
        pass
    except SystemExit:
        raise
    except BaseException as error:
        receipt["error"] = repr(error)
    finally:
        if lease_holder is not None:
            try:
                e1_gpu._lease_command("release", lease_holder)
                lease_released = True
            except BaseException as error:
                receipt["release_error"] = repr(error)
        receipt["lease_released"] = lease_released
        receipt["finished_epoch"] = time.time()

    parts = receipt["parts"]

    def outcome(part: dict) -> str:
        if part.get("status") == "skipped":
            return "skipped"
        if part.get("pass") is True or part.get("raised") is True:
            return "pass"
        return "fail"

    outcomes = {name: outcome(part) for name, part in parts.items()}
    receipt["part_outcomes"] = outcomes
    if receipt.get("error") or "fail" in outcomes.values():
        receipt["verdict"] = "fail"
    elif "skipped" in outcomes.values():
        receipt["verdict"] = "partial"
    else:
        receipt["verdict"] = "pass"
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(receipt, indent=1, sort_keys=True) + "\n")
    print(json.dumps({key: receipt[key] for key in
                      ("run_id", "verdict", "lease_released", "error")
                      if key in receipt}, indent=1))
    print(f"receipt: {out}")
    return 1 if receipt["verdict"] == "fail" else 0


if __name__ == "__main__":
    raise SystemExit(main())

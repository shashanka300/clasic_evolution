#!/usr/bin/env python3
"""Parallel job queue: packs independent jobs onto the CPU thread budget + shared GPU instead of running them one by one.

Jobs file (default results/jobs_5090.txt), one job per line:   key=val key ... :: shell command
  id=NAME        required, unique; finished ids are recorded in results/pqueue_done.txt and skipped on restart
  threads=N      CPU threads given to the job (exported as OMP/NUMBA/OPENBLAS/MKL/VECLIB_*_THREADS and NT)
  after=a,b      start only after these ids have finished (any rc)
  needs=data     start only once  $CLASSIC_ALGOS_DATA_DIR/.ready  (default data/.ready) exists
  barrier        start only after every earlier line has finished
  exclusive      barrier + nothing else starts while it runs (latency / QPS benchmarks)
Lines starting with # are comments. Jobs launch in file order; a later job may backfill free threads unless it
is held by after/needs/barrier. Every job is resumable (tracks skip cells already in their JSONL).

Run (from the repo root, inside WSL/Linux):
  nohup python3 scripts/pqueue.py > results/pqueue.log 2>&1 &
  python3 scripts/pqueue.py --status
"""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
THREAD_VARS = ("OMP_NUM_THREADS", "NUMBA_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
               "VECLIB_MAXIMUM_THREADS", "NT")


def parse(path: Path) -> list[dict]:
    jobs = []
    for ln, raw in enumerate(path.read_text().splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if " :: " not in line:
            sys.exit(f"{path}:{ln}: missing ' :: '")
        head, cmd = line.split(" :: ", 1)
        job = {"line": ln, "cmd": cmd.strip(), "threads": 1, "after": [], "needs": None,
               "barrier": False, "exclusive": False}
        for tok in shlex.split(head):
            k, _, v = tok.partition("=")
            if k == "id":
                job["id"] = v
            elif k == "threads":
                job["threads"] = int(v)
            elif k == "after":
                job["after"] = [x for x in v.split(",") if x]
            elif k == "needs":
                job["needs"] = v
            elif k in ("barrier", "exclusive"):
                job[k] = True
                job["barrier"] = True
            else:
                sys.exit(f"{path}:{ln}: unknown key {k!r}")
        if "id" not in job:
            sys.exit(f"{path}:{ln}: missing id=")
        jobs.append(job)
    ids = [j["id"] for j in jobs]
    dup = {i for i in ids if ids.count(i) > 1}
    if dup:
        sys.exit(f"duplicate ids: {sorted(dup)}")
    for j in jobs:
        for a in j["after"]:
            if a not in ids:
                sys.exit(f"{j['id']}: after={a} is not a job id")
    return jobs


def read_done(path: Path) -> dict[str, int]:
    done = {}
    if path.exists():
        for ln in path.read_text().splitlines():
            parts = ln.split()
            if len(parts) >= 2:
                done[parts[0]] = int(parts[1])
    return done


def data_ready() -> bool:
    d = Path(os.environ.get("CLASSIC_ALGOS_DATA_DIR", ROOT / "data")).expanduser()
    return (d / ".ready").exists()


def job_core(cmd: str) -> str:
    """The python invocation of a job line (what shows in its process cmdline), e.g. '-u tracks/x/run_v2.py --stage all'."""
    return cmd.split(" >")[0].split(".venv/bin/python", 1)[-1].strip()


def find_pid(core: str) -> int | None:
    """pid of an already-running python process for this job (so a restarted queue adopts it, never duplicates it)."""
    if not core:
        return None
    for d in Path("/proc").iterdir():
        if not d.name.isdigit() or int(d.name) == os.getpid():
            continue
        try:
            argv = (d / "cmdline").read_bytes().split(b"\0")
        except OSError:
            continue
        line = " ".join(a.decode(errors="replace") for a in argv if a)
        if "python" in argv[0].decode(errors="replace") and line.endswith(core):
            return int(d.name)
    return None


class Adopted:
    """Popen-like handle for a job started by an earlier queue instance (exit code unknown: recorded as 0)."""

    def __init__(self, pid: int):
        self.pid = pid

    def poll(self):
        return None if Path(f"/proc/{self.pid}").exists() else 0


def log(msg: str) -> None:
    print(f"{time.strftime('%F %T')} {msg}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", default="results/jobs_5090.txt")
    ap.add_argument("--done", default="results/pqueue_done.txt")
    ap.add_argument("--budget", type=int, default=os.cpu_count(), help="total CPU threads to hand out")
    ap.add_argument("--max-jobs", type=int, default=9, help="cap on concurrent jobs (RAM)")
    ap.add_argument("--poll", type=float, default=15.0)
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--retry-failed", action="store_true", help="re-run ids whose recorded rc != 0")
    args = ap.parse_args()
    os.chdir(ROOT)
    jobs_path, done_path = Path(args.jobs), Path(args.done)
    jobs = parse(jobs_path)
    done = read_done(done_path)
    if args.retry_failed:
        done = {k: v for k, v in done.items() if v == 0}

    if args.status:
        running = subprocess.run(["pgrep", "-af", r"python -u (tracks|-m core|scripts)"], capture_output=True, text=True).stdout
        for j in jobs:
            core = job_core(j["cmd"])
            st = ("done" if done.get(j["id"]) == 0 else f"FAILED rc={done[j['id']]}" if j["id"] in done
                  else "RUNNING" if core and core in running else "pending")
            print(f"{j['id']:<16} {st:<14} threads={j['threads']:<3} {j['cmd'][:90]}")
        print(f"\ndata ready: {data_ready()}\nrunning processes:\n{running}")
        return

    running: dict[str, tuple[subprocess.Popen | Adopted, dict, float]] = {}
    pending = [j for j in jobs if j["id"] not in done]
    for job in list(pending):  # jobs still running from a previous queue instance
        pid = find_pid(job_core(job["cmd"]))
        if pid:
            running[job["id"]] = (Adopted(pid), job, time.time())
            pending.remove(job)
            log(f"ADOPT {job['id']} pid={pid} (already running)")
    log(f"start: {len(pending)} pending / {len(jobs)} jobs, budget={args.budget} threads, max_jobs={args.max_jobs}")
    while pending or running:
        for jid, (proc, job, t0) in list(running.items()):
            rc = proc.poll()
            if rc is None:
                continue
            del running[jid]
            done[jid] = rc
            with open(done_path, "a") as f:
                f.write(f"{jid} {rc} {time.time() - t0:.0f}s\n")
            log(f"END   {jid} rc={rc} ({(time.time() - t0) / 3600:.2f} h)")

        used = sum(j["threads"] for _, j, _ in running.values())
        exclusive_running = any(j["exclusive"] for _, j, _ in running.values())
        ready = data_ready()
        for idx, job in enumerate(list(pending)):
            if exclusive_running or len(running) >= args.max_jobs:
                break
            earlier_open = [p for p in jobs[:jobs.index(job)] if p["id"] not in done]
            if job["barrier"] and earlier_open:
                # nothing after a barrier may start ahead of it
                break
            if job["exclusive"] and running:
                break
            if any(a not in done for a in job["after"]):
                continue
            if job["needs"] == "data" and not ready:
                continue
            # a job wider than the whole budget still runs, but alone
            if used + job["threads"] > args.budget and running:
                continue
            env = dict(os.environ)
            for v in THREAD_VARS:
                env[v] = str(job["threads"])
            proc = subprocess.Popen(["sh", "-c", job["cmd"]], env=env, start_new_session=True)
            running[job["id"]] = (proc, job, time.time())
            pending.remove(job)
            used += job["threads"]
            log(f"START {job['id']} threads={job['threads']} (in use {used}/{args.budget}, {len(running)} jobs): {job['cmd']}")
            if job["exclusive"]:
                break
        time.sleep(args.poll)
    log("queue empty")


if __name__ == "__main__":
    main()

"""Optional sharing of one GPU with other jobs: a flock around GPU work, and turn-taking with one peer project.

Off by default. scripts/cuda_run.sh sources scripts/gpu_env.local.sh (git-ignored) when it exists; that file sets the
environment below for a machine whose GPU is shared:

    GOI_GPU_LOCK         the flock every GPU job holds while it computes (default /tmp/goi_gpu.lock)
    GOI_GPU_PEER=1       alternate turns with a peer project: one peer job, then one of ours (a training chunk of at
                         most ~30 min), and so on. The turn lives in GOI_GPU_TURN ("peer" name or "goi"); a goi job
                         does not take the lock while it is the peer's turn and the peer has work running or queued,
                         and hands the turn to the peer when it releases the lock. While a goi job waits it keeps
                         GOI_GPU_WAITING, so the peer's launcher can do the same.
    GOI_GPU_PEER_NAME    the peer's value in the turn file (default "peer")
    GOI_GPU_TURN         turn file (default /tmp/gpu_turn)
    GOI_GPU_WAITING      our waiting marker (default /tmp/gpu_waiting.goi)
    GOI_GPU_PEER_WAITING the peer's waiting marker, re-touched every few seconds while a peer job waits (optional)
    GOI_GPU_PEER_FLOCK   argv prefix of a peer job blocked on the lock, e.g. "flock -w 86400" (optional)
    GOI_GPU_PEER_QUEUE   the peer's queue status command; busy if its first line starts with RUNNING or has
                         "main: N queued" with N > 0 (optional)

scripts/cuda_run.sh normally wraps a whole job in the lock. Jobs launched with GOI_LATE_LOCK=1 take it themselves,
after compiling ahead of time, so a long compile never holds the card:

    compiled = jax.jit(f).lower(*args).compile()   # CPU work, no lock
    with gpu_lock.held():
        out = compiled(*args)                      # GPU work, lock held

Without GOI_LATE_LOCK the lock is already held by cuda_run.sh and held() does nothing.
"""
import fcntl
import os
import shlex
import subprocess
import time
from contextlib import contextmanager

ENV = os.environ
LOCK = ENV.get("GOI_GPU_LOCK", "/tmp/goi_gpu.lock")
PEER = ENV.get("GOI_GPU_PEER") == "1"
PEER_NAME = ENV.get("GOI_GPU_PEER_NAME", "peer")
TURN = ENV.get("GOI_GPU_TURN", "/tmp/gpu_turn")
WAITING = ENV.get("GOI_GPU_WAITING", "/tmp/gpu_waiting.goi")
PEER_WAITING = ENV.get("GOI_GPU_PEER_WAITING", "")
PEER_FLOCK = [a.encode() for a in shlex.split(ENV.get("GOI_GPU_PEER_FLOCK", ""))]
PEER_QUEUE = shlex.split(ENV.get("GOI_GPU_PEER_QUEUE", ""))


def peer_blocked():
    """True while a peer job blocks on the lock (a process whose argv starts with GOI_GPU_PEER_FLOCK)."""
    if not PEER_FLOCK:
        return False
    for pid in os.listdir("/proc"):
        if pid.isdigit():
            try:
                with open(f"/proc/{pid}/cmdline", "rb") as f:
                    if f.read().split(b"\0")[:len(PEER_FLOCK)] == PEER_FLOCK:
                        return True
            except OSError:
                pass
    return False


def peer_busy():
    """True while the peer has a GPU job running, queued or waiting for the lock (its marker file, a blocked flock, or
    its queue status). The marker is re-touched every few seconds while a peer job waits, so an older one is stale."""
    try:
        if PEER_WAITING and time.time() - os.path.getmtime(PEER_WAITING) < 15:
            return True
    except OSError:
        pass
    if peer_blocked():
        return True
    if not PEER_QUEUE:
        return False
    try:
        head = subprocess.run(PEER_QUEUE, capture_output=True, text=True, timeout=20).stdout.splitlines()[:1]
    except (OSError, subprocess.SubprocessError):
        return False
    line = head[0] if head else ""
    if line.startswith("RUNNING"):
        return True
    for part in line.split("|"):
        part = part.strip()
        if part.startswith("main:") and not part.startswith("main: 0 "):
            return True
    return False


def turn():
    try:
        with open(TURN) as f:
            return f.read().strip() or "goi"
    except OSError:
        return "goi"


def set_turn(who):
    with open(TURN, "w") as f:
        f.write(who + "\n")


def peers_turn():
    return PEER and turn() == PEER_NAME and peer_busy()


def wait_turn(grace=None):
    """Block until it is goi-jax's turn (or the peer has nothing to run). Returns seconds waited."""
    t = time.perf_counter()
    if not PEER:
        return 0.0
    open(WAITING, "w").close()
    try:
        time.sleep(float(ENV.get("GOI_LOCK_GRACE", "60")) if grace is None else grace)
        while peers_turn():
            time.sleep(15)
    finally:
        try:
            os.remove(WAITING)
        except OSError:
            pass
    return time.perf_counter() - t


@contextmanager
def held(path=LOCK):
    if ENV.get("GOI_LATE_LOCK") != "1":
        yield 0.0
        return
    with open(path, "a") as f:
        t = time.perf_counter()
        while True:
            wait_turn()
            fcntl.flock(f, fcntl.LOCK_EX)
            # Re-check once the lock is ours: goi jobs queued on the flock together would otherwise take the card back
            # to back while the peer waits (the turn was once only checked before queueing).
            if not peers_turn():
                break
            fcntl.flock(f, fcntl.LOCK_UN)
        try:
            yield time.perf_counter() - t   # seconds spent waiting for our turn and the lock
        finally:
            if PEER:
                set_turn(PEER_NAME)         # alternate: the peer's next job goes before our next one
            fcntl.flock(f, fcntl.LOCK_UN)


if __name__ == "__main__":   # used by scripts/cuda_run.sh: `python -m goi_jax.gpu_lock wait` / `... done`
    import sys
    if sys.argv[1:] == ["wait"]:
        print(f"goi turn after {wait_turn():.0f} s", flush=True)
    elif sys.argv[1:] == ["done"] and PEER:
        set_turn(PEER_NAME)

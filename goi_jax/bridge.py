"""Run the game from WSL and talk to its bridge (goi_env.client.GoiClient, protocol in plugin/Server.cs).

WSL2 forwards Windows' localhost to listeners in WSL but not the other way round, so this side listens and the game
dials in (-goi-connect). That localhost relay sometimes stops forwarding new ports (a launch then never connected in
180 s although the game was up and dialling: BepInEx/LogOutput.log showed "bridge will dial" and no "connected"), so
this side also listens on the WSL VM's own address and the game alternates between the two (-goi-host). Launching
needs WSL interop.

    python -m goi_jax.bridge          # smoke test: launch, reset, a few steps, quit
"""
import fcntl
import os
import select
import socket
import subprocess
import time
from contextlib import contextmanager
from pathlib import Path

from goi_env.client import GoiClient

GAME_DIR = Path(os.environ.get("GOI_GAME_DIR", "/mnt/c/Program Files (x86)/Steam/steamapps/common/Getting Over It"))
EXE = "GettingOverIt.exe"


@contextmanager
def _one_game_starting():
    """BepInEx rewrites its shared config as each game starts, and games that start together come up without the mod."""
    with open(Path(os.environ.get("TMPDIR", "/tmp")) / "goi-rl-start.lock", "a") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


def _vm_address():
    """The WSL VM's own IPv4 address (eth0), which Windows reaches without the localhost relay; None if unknown."""
    try:
        out = subprocess.run(["ip", "-4", "-o", "addr", "show", "eth0"], capture_output=True, text=True, timeout=5).stdout
        return out.split()[3].split("/")[0]
    except (OSError, IndexError, subprocess.SubprocessError):
        return None


def running_games():
    """PIDs of GettingOverIt.exe processes running on Windows (stale games from an earlier launch show up here)."""
    try:
        out = subprocess.run(["tasklist.exe", "/FI", f"IMAGENAME eq {EXE}", "/FO", "CSV", "/NH"], capture_output=True,
                             text=True, timeout=20).stdout
    except (OSError, subprocess.SubprocessError):
        return []
    return [int(r.split('","')[1]) for r in out.splitlines() if r.startswith(f'"{EXE}"')]


def _log_tail(n=6):
    try:
        lines = (bepinex_dir() / "LogOutput.log").read_text(errors="replace").splitlines()
    except OSError as e:
        return f"(no BepInEx log: {e})"
    keep = [ln for ln in lines if "pot position" not in ln]
    return " | ".join(keep[-n:])


def start(width=640, height=360, args=(), timeout=180.0, headless=True, **config):
    """Launch the game in RL mode and wait for it to dial in; returns (process, client). Config as in
    GoiClient.configure; args are extra launch flags (e.g. "-goi-dump"). Headless (Unity's -batchmode -nographics: no
    window, no rendering, so no frames) unless asked; RL games are always muted. The game quits when the client
    disconnects. On a timeout the error names the BepInEx log's last lines (did the plugin load, which address did
    it dial)."""
    listeners = []
    try:
        with _one_game_starting():
            lo = socket.socket()
            lo.bind(("127.0.0.1", 0))
            port = lo.getsockname()[1]
            listeners.append(lo)
            vm = _vm_address()
            if vm:
                try:
                    s = socket.socket()
                    s.bind((vm, port))
                    listeners.append(s)
                except OSError:
                    vm = None
            for s in listeners:
                s.listen(1)
            proc = subprocess.Popen([f"./{EXE}", "-goi-rl", "-goi-connect", str(port),
                                     *(("-goi-host", vm) if vm else ()), "-screen-fullscreen", "0",
                                     "-screen-width", str(width), "-screen-height", str(height),
                                     *(("-batchmode", "-nographics") if headless else ()), *args],
                                    cwd=GAME_DIR, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                                    stderr=subprocess.DEVNULL)
            try:
                ready, _, _ = select.select(listeners, [], [], timeout)
                if not ready:
                    raise TimeoutError(f"the game did not dial port {port} within {timeout:.0f} s "
                                       f"(exited: {proc.poll() is not None}); BepInEx log: {_log_tail()}")
                sock, _ = ready[0].accept()
            except BaseException:
                proc.kill()  # a game that never dialled in would never quit on its own
                stop_all()
                raise
    finally:
        for s in listeners:
            s.close()
    return proc, GoiClient(sock=sock, **config)


def stop_all():
    """Kill every running GettingOverIt.exe (the process WSL holds is only interop's proxy)."""
    subprocess.run(["taskkill.exe", "/F", "/IM", EXE], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def bepinex_dir():
    return GAME_DIR / "BepInEx"


if __name__ == "__main__":
    t = time.perf_counter()
    proc, c = start()
    try:
        obs = c.reset()
        print(f"connected and reset in {time.perf_counter() - t:.1f} s; pot at ({obs[0]:.3f}, {obs[1]:.3f})")
        t = time.perf_counter()
        for i in range(200):
            obs = c.step(1.0, 0.5)
        print(f"200 steps: {200 / (time.perf_counter() - t):.0f} steps/s; pot at ({obs[0]:.3f}, {obs[1]:.3f})")
        c.close()
    finally:
        time.sleep(2)
        if proc.poll() is None:
            stop_all()

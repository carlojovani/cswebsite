from __future__ import annotations

import os
import shutil
import signal
import subprocess
import sys
import time
from typing import Optional


DOCKER_DESKTOP_EXE = r"C:\Program Files\Docker\Docker\Docker Desktop.exe"
COMPOSE_FILE = "docker-compose.yml"  # если файл называется иначе — поменяй
COMPOSE_SERVICES = ["redis"]         # добавь сюда то, что нужно (db/redis/etc)


def py() -> str:
    return sys.executable


def is_windows() -> bool:
    return os.name == "nt"


def run(cmd: list[str], *, check: bool = True, capture: bool = False) -> subprocess.CompletedProcess:
    return subprocess.run(
        cmd,
        check=check,
        text=True,
        stdout=subprocess.PIPE if capture else None,
        stderr=subprocess.STDOUT if capture else None,
    )


def docker_exists() -> bool:
    return shutil.which("docker") is not None


def docker_ready() -> bool:
    if not docker_exists():
        return False
    try:
        # Быстрый health-check демона
        run(["docker", "info"], check=True, capture=True)
        return True
    except Exception:
        return False


def start_docker_desktop() -> None:
    # Мы НЕ создаём контейнеры здесь — только запускаем сам Docker Desktop
    if not is_windows():
        return
    if os.path.exists(DOCKER_DESKTOP_EXE):
        print("[dev_run] starting Docker Desktop...", flush=True)
        subprocess.Popen([DOCKER_DESKTOP_EXE], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    else:
        print(f"[dev_run] Docker Desktop exe not found at: {DOCKER_DESKTOP_EXE}", flush=True)


def wait_for_docker(timeout_sec: int = 90) -> None:
    t0 = time.time()
    while time.time() - t0 < timeout_sec:
        if docker_ready():
            print("[dev_run] Docker engine is ready.", flush=True)
            return
        time.sleep(1.0)
    raise RuntimeError("Docker engine is not ready (timeout).")


def compose_up_no_recreate(services: list[str]) -> None:
    # Важно: --no-recreate => НЕ пересоздаёт существующие контейнеры
    # --remove-orphans можно не использовать, чтобы не удалять “лишние” контейнеры
    if not os.path.exists(COMPOSE_FILE):
        raise FileNotFoundError(f"{COMPOSE_FILE} not found in project root")

    cmd = ["docker", "compose", "-f", COMPOSE_FILE, "up", "-d", "--no-recreate"] + services
    print("[dev_run] docker compose up (no recreate):", " ".join(cmd), flush=True)
    run(cmd, check=True, capture=False)


def start_process(name: str, cmd: list[str]) -> subprocess.Popen:
    print(f"[dev_run] starting {name}: {' '.join(cmd)}", flush=True)

    creationflags = 0
    if is_windows():
        creationflags = subprocess.CREATE_NEW_PROCESS_GROUP  # type: ignore[attr-defined]

    p = subprocess.Popen(cmd, env=os.environ.copy(), creationflags=creationflags)
    print(f"[dev_run] {name} pid={p.pid}", flush=True)
    return p


def stop_process(name: str, p: subprocess.Popen) -> None:
    if p.poll() is not None:
        return
    print(f"[dev_run] stopping {name}...", flush=True)
    try:
        if is_windows():
            p.send_signal(signal.CTRL_BREAK_EVENT)  # type: ignore[attr-defined]
        else:
            p.terminate()
    except Exception:
        pass


def main() -> int:
    procs: list[tuple[str, subprocess.Popen]] = []

    try:
        # 1) Docker
        if not docker_ready():
            start_docker_desktop()
            wait_for_docker(timeout_sec=120)

        # 2) Compose services (не пересоздаём контейнеры)
        compose_up_no_recreate(COMPOSE_SERVICES)

        # 3) Django
        procs.append(("django", start_process("django", [py(), "manage.py", "runserver", "127.0.0.1:8000"])))

        # 4) Celery (Windows: solo)
        procs.append(
            (
                "celery",
                start_process(
                    "celery",
                    [py(), "-m", "celery", "-A", "backend.celery", "worker", "-l", "info", "-P", "solo"],
                ),
            )
        )

        print("[dev_run] all services are up. Ctrl+C to stop django/celery.", flush=True)

        # loop: если что-то упало — покажем сразу
        while True:
            time.sleep(0.5)
            for name, p in procs:
                code = p.poll()
                if code is not None:
                    print(f"[dev_run] ERROR: {name} exited with code={code}", flush=True)
                    return code if code != 0 else 1

    except KeyboardInterrupt:
        print("\n[dev_run] Ctrl+C received", flush=True)
        return 0

    finally:
        # Останавливаем только локальные процессы (django/celery).
        # Контейнеры НЕ трогаем, чтобы не ломать окружение.
        for name, p in procs:
            stop_process(name, p)

        time.sleep(1.0)

        for name, p in procs:
            if p.poll() is None:
                try:
                    p.kill()
                except Exception:
                    pass

        print("[dev_run] stopped django/celery. Docker containers left running.", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())

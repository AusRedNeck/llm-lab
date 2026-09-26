"""Process-lifetime singleton guard for llm-lab trainers."""
import json
import os
import socket
import sys
from datetime import datetime, timezone
from pathlib import Path

DEFAULT_LOCK_PATH = Path(__file__).resolve().parents[1] / ".train.lock"
DEFAULT_WATCHDOG_LOCK_PATH = (
    Path(__file__).resolve().parents[1] / ".train-watchdog.lock"
)


class TrainLockError(RuntimeError):
    pass


class TrainLock:
    def __init__(self, file_obj, path, owner, metadata):
        self._file = file_obj
        self.path = path
        self.owner = owner
        self.metadata = metadata

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        self.release()

    def release(self):
        if self._file.closed:
            return
        try:
            if os.name == "nt":
                import msvcrt
                self._file.seek(0)
                msvcrt.locking(self._file.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl
                fcntl.flock(self._file.fileno(), fcntl.LOCK_UN)
        except OSError:
            pass
        finally:
            self._file.close()


def _write_owner(file_obj, owner):
    payload = {
        "pid": os.getpid(),
        "host": socket.gethostname(),
        "started_at": datetime.now(timezone.utc).isoformat(),
        "python": sys.executable,
        "owner": owner,
    }
    file_obj.seek(0)
    file_obj.truncate()
    file_obj.write((json.dumps(payload, indent=2) + "\n").encode("utf-8"))
    file_obj.flush()
    os.fsync(file_obj.fileno())
    return payload


def acquire_train_lock(path=DEFAULT_LOCK_PATH, owner=None, label="train.train"):
    """Atomically acquire the machine-wide trainer lock.

    The byte-range lock is held by the open file handle until release(), so the
    kernel releases it if the process is killed. The JSON beside it is diagnostic
    only; stale metadata never grants ownership by itself.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    file_obj = os.fdopen(
        os.open(str(path), os.O_RDWR | os.O_CREAT, 0o666), "r+b", buffering=0
    )
    if file_obj.seek(0, os.SEEK_END) == 0:
        file_obj.write(b" ")
    try:
        if os.name == "nt":
            import msvcrt
            file_obj.seek(0)
            msvcrt.locking(file_obj.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(file_obj.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError as exc:
        file_obj.close()
        try:
            owner_host = socket.gethostname()
            with open(path, "rb") as reader:
                blob = reader.read().decode("utf-8", errors="replace")
            start = blob.find("{")
            if start >= 0:
                current = json.loads(blob[start:])
                owner_host = current.get("host", "?")
        except Exception:
            pass
        raise TrainLockError(
            f"{label} singleton lock is already held on {owner_host} "
            f"(lock={path})"
        ) from exc
    try:
        metadata = _write_owner(file_obj, owner or {})
    except Exception:
        if os.name == "nt":
            import msvcrt
            file_obj.seek(0)
            msvcrt.locking(file_obj.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl
            fcntl.flock(file_obj.fileno(), fcntl.LOCK_UN)
        file_obj.close()
        raise
    return TrainLock(file_obj, path, owner or {}, metadata)


def acquire_watchdog_lock(path=DEFAULT_WATCHDOG_LOCK_PATH):
    """Serialize watchdog ticks that overlap in time."""
    return acquire_train_lock(path, owner={"role": "watchdog"}, label="watchdog")

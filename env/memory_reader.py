"""Read build-specific th06nc state without importing the reference project."""
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import struct


class MemoryReadError(RuntimeError):
    """A sample is missing, inconsistent, or invalid."""


class UnsupportedBuildError(RuntimeError):
    pass


@dataclass(frozen=True)
class Bullet:
    slot: int
    x: float
    y: float
    vx: float
    vy: float
    radius: float
    state: int
    radius_estimated: bool = False


@dataclass(frozen=True)
class Snapshot:
    frame: int
    scene: int
    wanted_scene: int
    lives: int
    bombs: int
    deaths: int
    dying: bool
    retry: bool
    player_state: int
    x: float
    y: float
    bullets: tuple = ()

    @property
    def running(self):
        # Zero reserve lives still means one playable life remains.
        return self.scene == self.wanted_scene == 2 and not self.retry

    @property
    def ready(self):
        return self.running and not self.dying and self.player_state in (0, 3)


REQUIRED = ("CALC_COUNT", "CUR_STATE", "WANT_STATE", "LIVES", "BOMBS",
            "DEATHS", "DYING", "RETRY", "PLAYER_BASE", "PLAYER_X", "BM_RVA")
DEFAULT_PROFILES = Path(__file__).resolve().parents[1] / "configs" / "memory_profiles.json"


def load_profile(exe_path, profiles_path=DEFAULT_PROFILES):
    digest = hashlib.md5(Path(exe_path).read_bytes()).hexdigest()
    profiles = json.loads(Path(profiles_path).read_text(encoding="utf-8"))
    try:
        entry = profiles["builds"][digest]
        addresses = {name: int(entry["rvas"][name], 0) for name in REQUIRED}
    except (KeyError, TypeError, ValueError) as exc:
        raise UnsupportedBuildError(f"Missing or incomplete memory profile: {digest}") from exc
    return digest, addresses


class MemoryReader:
    """The process dependency only needs read(address, size) and close()."""

    def __init__(self, process, base_address, addresses):
        self.process = process
        self.base_address = base_address
        self.addresses = addresses

    def _read(self, rva, size):
        try:
            data = self.process.read(self.base_address + rva, size)
        except OSError as exc:
            raise MemoryReadError(str(exc)) from exc
        if data is None or len(data) != size:
            raise MemoryReadError(f"Short read at RVA {rva:#x}: expected {size} bytes")
        return data

    def _value(self, name, fmt):
        return struct.unpack(fmt, self._read(self.addresses[name], struct.calcsize(fmt)))[0]

    def frame(self):
        return self._value("CALC_COUNT", "<B")

    def snapshot(self, attempts=3):
        for _ in range(attempts):
            start = self.frame()
            scene = self._value("CUR_STATE", "<i")
            wanted = self._value("WANT_STATE", "<i")
            lives = self._value("LIVES", "<B")
            bombs = self._value("BOMBS", "<B")
            deaths = self._value("DEATHS", "<i")
            dying = bool(self._value("DYING", "<B"))
            retry = self._value("RETRY", "<B") == 255
            playable_scene = scene == wanted == 2 and not retry
            if playable_scene:
                state = self._read(self.addresses["PLAYER_BASE"] + 0x7898, 1)[0]
                x, y = struct.unpack("<ff", self._read(self.addresses["PLAYER_X"], 8))
                bullets = self._bullets()
            else:
                # Player/pool storage may be stale while a scene is being loaded.
                state, x, y, bullets = 0, 0.0, 0.0, ()
            end = self.frame()
            if start != end:
                continue
            if not (math.isfinite(x) and math.isfinite(y)):
                raise MemoryReadError("Non-finite player position")
            if playable_scene and not (-64 <= x <= 448 and -64 <= y <= 512):
                raise MemoryReadError("Player position is outside the expected field")
            if deaths < 0 or state > 3:
                raise MemoryReadError("Invalid player state or death count")
            return Snapshot(end, scene, wanted, lives, bombs, deaths, dying,
                            retry, state, x, y, bullets)
        raise MemoryReadError("Game advanced during every sample; retry acquisition")

    def _bullets(self):
        # Layout taken from the reference build; verify it on each new build.
        stride = 0x620
        raw = self._read(self.addresses["BM_RVA"] + 8, 640 * stride)
        result = []
        for slot in range(640):
            offset = slot * stride
            state = struct.unpack_from("<H", raw, offset + 0x44)[0]
            if state not in (1, 2, 3, 4):
                continue
            x, y = struct.unpack_from("<ff", raw, offset + 0x30)
            vx, vy = struct.unpack_from("<ff", raw, offset + 8)
            w, h = struct.unpack_from("<ff", raw, offset + 0x5F4)
            if not all(math.isfinite(v) for v in (x, y, vx, vy)):
                raise MemoryReadError(f"Non-finite bullet data in slot {slot}")
            radius = min(w, h) / 2 if math.isfinite(w) and math.isfinite(h) else 0
            estimated = not 0 < radius <= 32
            if estimated:
                # Match the reference reader for missing hitbox dimensions.
                # This is an estimate, not a guarantee of collision geometry.
                radius = 3.0
            result.append(Bullet(slot, x, y, vx, vy, radius, state, estimated))
        return tuple(result)

    def close(self):
        self.process.close()

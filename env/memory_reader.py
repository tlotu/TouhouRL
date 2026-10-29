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
class Enemy:
    slot: int
    x: float
    y: float
    hp: int
    flags1: int
    flags2: int
    flags3: int

    def record(self):
        return {"slot": self.slot, "x": self.x, "y": self.y, "hp": self.hp,
                "is_boss": bool(self.flags2 & 0x08),
                "collidable": bool(self.flags2 & 0x02),
                "visible": not bool(self.flags3 & 0x08),
                "flags1": self.flags1, "flags2": self.flags2, "flags3": self.flags3}


@dataclass(frozen=True)
class ResultMenu:
    active: bool
    selection: int = 0
    phase: int = 0
    outcome_flag: int = 0

    @property
    def outcome(self):
        if self.active and self.phase == 1:
            return "success" if self.outcome_flag == 1 else "failure"
        return None


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
    enemies: tuple = ()
    result_menu: ResultMenu | None = None

    @property
    def running(self):
        # Zero reserve lives still means one playable life remains.
        return (self.scene == self.wanted_scene == 2 and not self.retry
                and not (self.result_menu and self.result_menu.active))

    @property
    def ready(self):
        return self.running and not self.dying and self.player_state in (0, 3)


REQUIRED = ("CALC_COUNT", "CUR_STATE", "WANT_STATE", "LIVES", "BOMBS",
            "DEATHS", "DYING", "RETRY", "PLAYER_BASE", "PLAYER_X", "BM_RVA", "EM_RVA")
DEFAULT_PROFILES = Path(__file__).resolve().parents[1] / "configs" / "memory_profiles.json"


def load_profile(exe_path, profiles_path=DEFAULT_PROFILES):
    digest = hashlib.md5(Path(exe_path).read_bytes()).hexdigest()
    profiles = json.loads(Path(profiles_path).read_text(encoding="utf-8"))
    try:
        entry = profiles["builds"][digest]
        addresses = {name: int(entry["rvas"][name], 0) for name in REQUIRED}
        for optional in ("GUI", "SPELL_RESULT_ACTIVE_CANDIDATE", "SPELL_RESULT_MENU_CANDIDATE"):
            if optional in entry["rvas"]:
                addresses[optional] = int(entry["rvas"][optional], 0)
        if entry.get("spell_result_verified") is True:
            addresses["SPELL_RESULT_ACTIVE"] = addresses["SPELL_RESULT_ACTIVE_CANDIDATE"]
            addresses["SPELL_RESULT_MENU"] = addresses["SPELL_RESULT_MENU_CANDIDATE"]
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
            menu = self._result_menu() if scene == wanted == 2 else None
            playable_scene = scene == wanted == 2 and not retry and not (menu and menu.active)
            if playable_scene:
                state = self._read(self.addresses["PLAYER_BASE"] + 0x7898, 1)[0]
                x, y = struct.unpack("<ff", self._read(self.addresses["PLAYER_X"], 8))
                bullets = self._bullets()
                enemies = self._enemies()
            else:
                # Player/pool storage may be stale while a scene is being loaded.
                state, x, y, bullets = 0, 0.0, 0.0, ()
                enemies = ()
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
                            retry, state, x, y, bullets, enemies, menu)
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

    def _result_menu(self):
        if "SPELL_RESULT_ACTIVE" not in self.addresses:
            return None
        active = self._value("SPELL_RESULT_ACTIVE", "<B")
        if active not in (0, 1):
            raise MemoryReadError("Invalid spell-result active flag")
        if not active:
            return ResultMenu(False)
        raw = self._read(self.addresses["SPELL_RESULT_MENU"] + 0xB40, 0x260)
        selection = struct.unpack_from("<i", raw, 4)[0]
        phase = struct.unpack_from("<i", raw, 0x254)[0]
        outcome = raw[0x258]
        if phase not in range(6) or outcome not in (0, 1) or (phase == 1 and selection not in range(3)):
            raise MemoryReadError("Invalid spell-result menu fields")
        return ResultMenu(True, selection, phase, outcome)

    def _enemies(self):
        # Reference EnemyManager: an inline pool, not a pointer chain.
        stride = 0x10B0
        raw = self._read(self.addresses["EM_RVA"] + 8, 256 * stride)
        result = []
        for slot in range(256):
            offset = slot * stride
            flags1, flags2, flags3 = struct.unpack_from("<BBB", raw, offset + 0xBC)
            if not flags1 & 0x80:
                continue
            x, y = struct.unpack_from("<ff", raw, offset + 0xB0)
            hp = struct.unpack_from("<i", raw, offset + 0x234)[0]
            if not math.isfinite(x) or not math.isfinite(y):
                raise MemoryReadError(f"Non-finite enemy position in slot {slot}")
            # Keep offscreen, invisible, non-collidable, and nonpositive-HP
            # occupied slots for diagnosis. None alone proves a spell clear.
            result.append(Enemy(slot, x, y, hp, flags1, flags2, flags3))
        return tuple(result)

    def close(self):
        self.process.close()

"""Baseline-only candidate movement fields; diagnostic, never policy input."""
import math
import struct

from .memory_reader import MemoryReadError

BASELINE_BUILD = "b0cc28ec904e9efce2439ad857c938b5"


class MovementProbe:
    """Offsets from reference tools/offsets.py and verify_focus.py, unverified here."""
    def __init__(self, reader, build):
        if build != BASELINE_BUILD or reader.addresses.get("PLAYER_BASE") != 0x4FF3A0:
            raise ValueError("Movement candidates are available only for the baseline build")
        self.reader = reader

    def capture(self, expected_frame):
        record = {"verified": False, "source": "reference movement candidates"}
        try:
            start = self.reader.frame()
            mask = struct.unpack("<H", self.reader._read(0xA6EC60, 2))[0]
            raw = self.reader._read(self.reader.addresses["PLAYER_BASE"] + 0x785C, 20)
            speeds = struct.unpack_from("<ffff", raw, 4)
            end = self.reader.frame()
            record.update(available=True, frame_before=start, frame_after=end,
                          same_transition_frame=start == end == expected_frame,
                          input_mask=mask, slow_input_bit=bool(mask & 4), focus_raw=raw[0],
                          speed_parameters={name: value if math.isfinite(value) else None
                                            for name, value in zip(
                                                ("fast", "slow", "diagonal_fast_axis", "diagonal_slow_axis"), speeds)},
                          plausible=raw[0] in (0, 1, 4) and all(math.isfinite(v) and 0 < v <= 32 for v in speeds))
        except MemoryReadError as exc:
            record.update(available=False, error=str(exc))
        return record


def annotate_movement(row, probe, controls):
    if row.get("event") == "transition" and row["info"].get("player_x") is not None:
        row = {**row, "movement_diagnostic": {
            "post_step_cleanup": bool(row.get("terminated") or row.get("truncated")),
            "requested_slow": controls.slow, "owned_keys": sorted(controls.held),
            "os_shift": controls.keyboard.shift_state(),
            "game_candidates": probe.capture(row["info"]["frame"])}}
    return row

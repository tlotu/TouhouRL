"""Small labeled memory captures; no inferred menu meanings or writes."""
import struct

from .memory_reader import MemoryReadError


def capture_regions(reader):
    """Only inspect known state neighborhoods and the optional GUI anchor."""
    regions = {}
    def capture(name, address, size):
        try:
            data = reader.process.read(address, size)
            if data is None or len(data) != size:
                raise MemoryReadError("Unreadable or short region")
            regions[name] = {"address": hex(address), "size": size, "hex": data.hex()}
            return data
        except OSError as exc:
            regions[name] = {"address": hex(address), "size": size, "error": str(exc)}
        except MemoryReadError as exc:
            regions[name] = {"address": hex(address), "size": size, "error": str(exc)}
        return None

    base, a = reader.base_address, reader.addresses
    capture("status", base + a["LIVES"] - 0x100, 0x400)
    capture("player_state", base + a["PLAYER_BASE"] + 0x7700, 0x300)
    if "SPELL_RESULT_ACTIVE_CANDIDATE" in a:
        capture("spell_result_active_candidate", base + a["SPELL_RESULT_ACTIVE_CANDIDATE"], 1)
    if "SPELL_RESULT_MENU_CANDIDATE" in a:
        # Derived from this exact build's menu update/initializer references.
        capture("spell_result_state_candidate", base + a["SPELL_RESULT_MENU_CANDIDATE"] + 0xB40, 0x260)
    if "GUI" in a:
        raw = capture("gui_anchor", base + a["GUI"], 8)
        if raw:
            pointer = struct.unpack("<Q", raw)[0]
            # The GUI anchor's pointee layout is not verified. Read failures
            # are evidence, not permission to scan other process regions.
            if 0x10000 <= pointer < 0x0000800000000000:
                capture("gui_pointee", pointer, 0x2000)
            else:
                regions["gui_pointee"] = {"error": "GUI anchor is not a plausible user pointer"}
    return regions


def result_candidates(regions):
    """Expose unverified scalar values for human validation, never for input."""
    active = regions.get("spell_result_active_candidate", {})
    state = regions.get("spell_result_state_candidate", {})
    if "hex" not in active or "hex" not in state:
        return {"available": False, "verified": False}
    raw = bytes.fromhex(state["hex"])
    flag = bytes.fromhex(active["hex"])
    if len(flag) != 1 or len(raw) != 0x260:
        return {"available": False, "verified": False}
    return {"available": True, "verified": False, "active": flag[0],
            "selection": struct.unpack_from("<i", raw, 4)[0],
            "phase": struct.unpack_from("<i", raw, 0x254)[0],
            "outcome_flag": raw[0x258]}


def stable_differences(left, right, limit=64):
    """Find bytes stable within each label but different across labels.

    Returned offsets are region-relative candidates, not verified fields.
    Different builds or mixed labels are not comparable.
    """
    if len(left) < 2 or len(right) < 2:
        raise ValueError("At least two captures per label are required")
    builds = {row["build_md5"] for row in left + right}
    if len(builds) != 1:
        raise ValueError("Cannot compare different executable builds")
    for group in (left, right):
        if len({row["label"] for row in group}) != 1:
            raise ValueError("Each input must contain a single label")
    if left[0]["label"] == right[0]["label"]:
        raise ValueError("Use distinct labels")
    output = []
    names = set.intersection(*(set(row["regions"]) for row in left + right))
    for name in sorted(names):
        # Absolute pointer values are relocation artifacts, not menu evidence.
        if name == "gui_anchor":
            continue
        records = [row["regions"][name] for row in left + right]
        if any("hex" not in record for record in records):
            continue
        data = [bytes.fromhex(record["hex"]) for record in records]
        if len({len(value) for value in data}) != 1:
            continue
        ldata, rdata = data[:len(left)], data[len(left):]
        candidates = []
        count = 0
        for offset in range(len(data[0])):
            lv, rv = ldata[0][offset], rdata[0][offset]
            if lv != rv and all(d[offset] == lv for d in ldata) and all(d[offset] == rv for d in rdata):
                count += 1
                if len(candidates) < limit:
                    candidates.append({"offset": hex(offset), "left": lv, "right": rv})
        output.append({"region": name, "candidate_count": count, "candidates": candidates})
    return output

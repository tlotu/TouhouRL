"""Unverified reference laser layout, isolated from policy observations."""
import base64
import math
import struct

from .memory_reader import SnapshotRaceError, UnsupportedBuildError

BUILD = 'b0cc28ec904e9efce2439ad857c938b5'
MANAGER_RVA = 0x3EC2A0
POOL_OFFSET, STRIDE, COUNT = 0xF5018, 0x298, 64
LAYOUT = 'reference-laser-layout-candidate-v1'
PHASE_HINTS = {0: 'warning', 1: 'firing', 2: 'fading', 3: 'tail'}


def require_layout(reader, build):
    if build != BUILD or reader.addresses.get('BM_RVA') != MANAGER_RVA:
        raise UnsupportedBuildError('Laser probe requires the validated build and matching manager RVA')


def decode_pool(raw):
    if len(raw) != STRIDE * COUNT:
        raise ValueError('Invalid laser pool size')
    rows = []
    for slot in range(COUNT):
        data = raw[slot*STRIDE:(slot+1)*STRIDE]
        state, phase = data[0x274], data[0x290]
        if state == 0:
            continue
        fields = {name: struct.unpack_from('<f', data, offset)[0] for name, offset in
            {'width':4, 'start_offset':0x10, 'end_offset':0x14, 'angle':0x260,
             'speed':0x26C, 'x':0x278, 'y':0x27C, 'z':0x280, 'length_limit':0x288}.items()}
        issues = [f'nonfinite_{k}' for k,v in fields.items() if not math.isfinite(v)]
        fields = {k:v if math.isfinite(v) else None for k,v in fields.items()}
        if phase not in PHASE_HINTS:
            issues.append('unknown_phase')
        if fields['width'] is not None and not 0 <= fields['width'] <= 2048:
            issues.append('width_out_of_probe_range')
        if any(v is not None and abs(v)>100000 for v in fields.values()):
            issues.append('geometry_out_of_probe_range')
        segment = None
        if not issues:
            x,y,a = fields['x'],fields['y'],fields['angle']
            segment = [[x+math.cos(a)*fields[k],y+math.sin(a)*fields[k]]
                       for k in ('start_offset','end_offset')]
        rows.append({'slot':slot, 'state_raw':state, 'phase_raw':phase,
            'phase_hint':PHASE_HINTS.get(phase,'unknown'), 'verified':False,
            **fields, 'timer_raw':struct.unpack_from('<i',data,0xC)[0],
            'warning_frames_raw':struct.unpack_from('<i',data,0x270)[0],
            'firing_frames_raw':struct.unpack_from('<i',data,0x268)[0],
            'fading_frames_raw':struct.unpack_from('<i',data,0x18)[0],
            'segment_candidate':segment, 'issues':issues,
            'raw_head_hex':data[:0x20].hex(), 'raw_tail_hex':data[0x250:].hex()})
    return rows


def capture(reader, *, attempts=6, raw_pool=False):
    """Bracket normal state and the candidate pool with the byte frame counter."""
    for attempt in range(attempts):
        before = reader.frame()
        try:
            sample = reader.snapshot()
        except SnapshotRaceError:
            continue
        raw = reader._read(MANAGER_RVA+POOL_OFFSET, STRIDE*COUNT) if sample.running else None
        after = reader.frame()
        if before != sample.frame or before != after:
            continue
        lasers = decode_pool(raw) if raw is not None else []
        record = {'frame':after, 'acquisition_attempts':attempt+1,
            'pool_sampled':raw is not None, 'candidate_count':len(lasers), 'lasers':lasers}
        if raw_pool and raw is not None:
            record['pool_base64'] = base64.b64encode(raw).decode('ascii')
        return sample,record
    raise SnapshotRaceError('Laser pool crossed frames on every bounded attempt')

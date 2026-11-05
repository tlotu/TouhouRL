"""Candidate layout decoding, frame coherence and read-only command wiring."""
import base64
import contextlib
import io
import json
import math
from pathlib import Path
import struct
import tempfile
import unittest
from unittest.mock import patch

from env.laser_probe import BUILD, MANAGER_RVA, STRIDE, COUNT, capture, decode_pool, require_layout
from env.memory_reader import SnapshotRaceError, UnsupportedBuildError, MemoryReadError
from tests.fakes import sample, FakeClock


def pool(phase=1, width=12.):
    data=bytearray(STRIDE*COUNT)
    for off,value in {4:width,0x10:10.,0x14:110.,0x260:math.pi/2,
                      0x26C:0.,0x278:192.,0x27C:100.,0x280:0.,0x288:100.}.items():
        struct.pack_into('<f',data,off,value)
    for off,value in {0xC:3,0x270:20,0x268:60,0x18:15}.items():
        struct.pack_into('<i',data,off,value)
    data[0x274]=1;data[0x290]=phase
    return bytes(data)


class Reader:
    addresses={'BM_RVA':MANAGER_RVA}
    def __init__(self, frames=None, state=None):
        self.frames=iter(frames or [10]*100)
        self.state=state or sample()
        self.reads=[];self.closed=False
    def frame(self):return next(self.frames)
    def snapshot(self):return self.state
    def _read(self,address,size):
        self.reads.append((address,size));return pool()
    def close(self):self.closed=True


class LaserProbeTests(unittest.TestCase):
    def test_segment_uses_origin_plus_both_offsets_and_keeps_raw(self):
        row=decode_pool(pool())[0]
        self.assertEqual(row['phase_hint'],'firing')
        self.assertFalse(row['verified'])
        self.assertAlmostEqual(row['segment_candidate'][0][0],192.,places=4)
        self.assertAlmostEqual(row['segment_candidate'][0][1],110.,places=4)
        self.assertAlmostEqual(row['segment_candidate'][1][1],210.,places=4)
        self.assertEqual(row['timer_raw'],3)
        self.assertTrue(row['raw_head_hex'])
        self.assertEqual(decode_pool(bytes(STRIDE*COUNT)),[])

    def test_unknown_phase_and_nonfinite_geometry_remain_diagnostic(self):
        row=decode_pool(pool(phase=99,width=float('nan')))[0]
        self.assertEqual(row['phase_raw'],99)
        self.assertIsNone(row['width'])
        self.assertIsNone(row['segment_candidate'])
        self.assertIn('unknown_phase',row['issues'])
        json.dumps(row,allow_nan=False)
        with self.assertRaises(ValueError):decode_pool(b'')

    def test_build_and_manager_guards(self):
        reader=Reader();require_layout(reader,BUILD)
        with self.assertRaises(UnsupportedBuildError):require_layout(reader,'other')
        reader.addresses={'BM_RVA':0}
        with self.assertRaises(UnsupportedBuildError):require_layout(reader,BUILD)

    def test_frame_races_retry_bounded_and_raw_pool_is_recoverable(self):
        reader=Reader(frames=[10,11,10,10])
        state,row=capture(reader,raw_pool=True)
        self.assertEqual(row['acquisition_attempts'],2)
        self.assertEqual(base64.b64decode(row['pool_base64']),pool())
        with self.assertRaises(SnapshotRaceError):capture(Reader(frames=[10,11]*6))
        class Broken(Reader):
            def _read(self,*args):raise MemoryReadError('short read')
        with self.assertRaises(MemoryReadError):capture(Broken())

    def test_menus_skip_pool_and_do_not_reuse_stale_lasers(self):
        reader=Reader(state=sample(scene=1,wanted_scene=1))
        _,row=capture(reader,raw_pool=True)
        self.assertFalse(row['pool_sampled'])
        self.assertEqual(row['lasers'],[])
        self.assertNotIn('pool_base64',row)
        self.assertFalse(reader.reads)

    def test_read_only_cli_closes_reader_and_logs_candidates(self):
        from scripts.probe_lasers import main
        reader=Reader();clock=FakeClock()
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent)as directory:
            output=Path(directory)/'laser.jsonl'
            args=['probe_lasers','--spell','young demon lord','--difficulty','Lunatic',
                  '--seconds','1','--interval','.5','--delay','0','--raw-pool','--output',str(output)]
            with patch('sys.argv',args),patch('scripts.probe_lasers.connect',return_value=(reader,1,BUILD)), \
                    patch('scripts.probe_lasers.time.perf_counter',clock), \
                    patch('scripts.probe_lasers.time.sleep',clock.sleep), \
                    patch('env.windows.WindowsKeyboard',side_effect=AssertionError('No input')), \
                    contextlib.redirect_stdout(io.StringIO()),contextlib.redirect_stderr(io.StringIO()):
                main()
            rows=[json.loads(l)for l in output.read_text().splitlines()]
            self.assertTrue(rows[0]['read_only'])
            self.assertFalse(rows[0]['verified'])
            self.assertEqual(rows[-1]['samples'],2)
            self.assertEqual(rows[-1]['peak_laser_candidates'],1)
            self.assertTrue(reader.closed)
            self.assertTrue(output.with_suffix('.console.log').exists())

    def test_cli_invalid_options_do_not_connect(self):
        from scripts.probe_lasers import main
        args=['probe_lasers','--spell','test','--difficulty','Lunatic',
              '--interval','0','--output','unused']
        with patch('sys.argv',args),patch('scripts.probe_lasers.connect')as connect, \
                contextlib.redirect_stderr(io.StringIO()),self.assertRaises(SystemExit):main()
        connect.assert_not_called()

    def test_wrong_build_closes_reader_before_pool_read(self):
        from scripts.probe_lasers import main
        reader=Reader()
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent)as directory:
            args=['probe_lasers','--spell','test','--difficulty','Normal','--output',str(Path(directory)/'out.jsonl')]
            with patch('sys.argv',args),patch('scripts.probe_lasers.connect',return_value=(reader,1,'other')), \
                    contextlib.redirect_stdout(io.StringIO()),contextlib.redirect_stderr(io.StringIO()), \
                    self.assertRaises(UnsupportedBuildError):main()
        self.assertTrue(reader.closed)
        self.assertFalse(reader.reads)

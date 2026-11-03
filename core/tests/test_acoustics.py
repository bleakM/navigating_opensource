"""Small waveform-only regression; no stored human recording is required."""
import math,sys,unittest
from pathlib import Path
import numpy as np
sys.path[:0]=[str(Path(__file__).resolve().parents[1]/'src'),str(Path(__file__).resolve().parents[2]/'interaction/src')]
from room_audio import source_level,active_rms,reliable_direction
from acoustics import render_audio


class AcousticTests(unittest.TestCase):
    def test_shared_gain_reaches_target_without_clipping(self):
        wave=np.random.default_rng(18).uniform(-.02,.02,16000).astype(np.float32)
        adjusted,meta=source_level(wave)
        self.assertAlmostEqual(20*np.log10(active_rms(adjusted)),-14.,places=4)
        self.assertLessEqual(np.max(np.abs(adjusted)),.921)

    def test_silence_and_invalid_channels_rejected(self):
        with self.assertRaises(ValueError):source_level(np.zeros(16000,np.float32))
        with self.assertRaises(ValueError):reliable_direction(np.zeros((16000,4)),16000)

    def test_four_channel_direction_uses_waveform(self):
        # Speech-frame gating requires quiet intervals; continuous noise should fail.
        signal=np.r_[np.zeros(8000),np.random.default_rng(19).normal(0,.03,16000),np.zeros(8000)]
        for angle in (0.,math.pi/2,math.pi):
            with self.subTest(angle=angle):
                source=[7+2*math.cos(angle),6+2*math.sin(angle),1.35]
                wave=render_audio(signal,16000,source,[7,6,1.35],max_order=0)
                result=reliable_direction(wave,16000)
                error=abs(math.atan2(math.sin(result['azimuth_rad']-angle),math.cos(result['azimuth_rad']-angle)))
                self.assertLessEqual(error,math.radians(3.))
                self.assertFalse(result['range_estimated'])


if __name__=='__main__':unittest.main()

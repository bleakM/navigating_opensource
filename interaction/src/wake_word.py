"""Pretrained sherpa-onnx keyword spotting. Input is audio, never a fixture label."""
from pathlib import Path
import numpy as np
from scipy.signal import resample_poly
import sherpa_onnx


class WakeWord:
    def __init__(self, directory='/workspace/models/kws-zh'):
        folder = Path(directory)
        def model(part):
            return str(next(folder.glob(part+'*.int8.onnx')))
        self.spotter = sherpa_onnx.KeywordSpotter(
            tokens=str(folder/'tokens.txt'), encoder=model('encoder'),
            decoder=model('decoder'), joiner=model('joiner'),
            keywords_file='/workspace/interaction/config/keywords-zh.txt',
            num_threads=2, keywords_score=1.0, keywords_threshold=.25)

    def detect(self, waveform, sample_rate):
        mono = np.asarray(waveform, dtype=np.float32)
        if mono.ndim != 1 or not np.isfinite(mono).all() or len(mono) < 1024:
            raise ValueError('Invalid mono audio')
        from math import gcd
        divisor = gcd(int(sample_rate), 16000)
        mono = resample_poly(mono, 16000//divisor, int(sample_rate)//divisor)
        stream = self.spotter.create_stream()
        found = []
        for chunk in np.array_split(np.r_[mono, np.zeros(8000,dtype=np.float32)],
                                     max(1,(len(mono)+8000)//1600)):
            stream.accept_waveform(16000, chunk.astype(np.float32))
            while self.spotter.is_ready(stream):
                self.spotter.decode_stream(stream)
                keyword = self.spotter.get_result(stream)
                if keyword:
                    found.append(keyword)
                    self.spotter.reset_stream(stream)
        return found

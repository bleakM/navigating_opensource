"""Silero speech endpoints; transport blocks never define an utterance boundary."""
import time
import numpy as np
import sherpa_onnx
from common import ROOT

RATE = 16000


def vad():
    config = sherpa_onnx.VadModelConfig()
    config.silero_vad.model = str(ROOT/'models/silero-vad.onnx')
    config.silero_vad.min_silence_duration = .6
    config.silero_vad.min_speech_duration = .25
    config.silero_vad.max_speech_duration = 7.
    config.sample_rate = RATE
    config.num_threads = 1
    return sherpa_onnx.VoiceActivityDetector(config, buffer_size_in_seconds=20)


def normalized(audio):
    audio = np.asarray(audio, np.float32)
    rms = float(np.sqrt(np.mean(audio**2))) if len(audio) else 0.
    peak = float(np.max(np.abs(audio))) if len(audio) else 0.
    gain = min(10., .06/max(rms, 1e-6), .95/max(peak, 1e-6))
    return audio*gain


class UtteranceStream:
    def __init__(self):
        self.detector = vad()
        self.pending = np.empty(0, np.float32)
        self.original = np.empty(0, np.float32)
        self.offset = 0
        self.total = 0
        self.updated = time.monotonic()
        self.sequence = 0

    def feed(self, audio, *, flush=False):
        self.updated = time.monotonic()
        self.original = np.r_[self.original, audio]
        self.total += len(audio)
        self.pending = np.r_[self.pending, normalized(audio)]
        while len(self.pending) >= 512:
            self.detector.accept_waveform(self.pending[:512])
            self.pending = self.pending[512:]
        if flush:
            if len(self.pending):
                self.detector.accept_waveform(np.pad(self.pending, (0, 512-len(self.pending))))
                self.pending = np.empty(0, np.float32)
            self.detector.flush()
        segments=[]
        while not self.detector.empty():
            segment=self.detector.front
            start=max(self.offset, segment.start-int(.08*RATE))
            end=min(self.total, segment.start+len(segment.samples)+int(.08*RATE))
            segments.append(self.original[start-self.offset:end-self.offset].copy())
            self.detector.pop()
        # Bounded original-sample ring, independent of how long mic is open.
        keep=12*RATE
        if len(self.original)>keep:
            dropped=len(self.original)-keep
            self.original=self.original[dropped:]
            self.offset+=dropped
        return segments


def speech_segments(audio):
    stream=UtteranceStream()
    # Identical transport cadence for enrollment and live verification.
    segments=[]
    for at in range(0,len(audio),8000):
        segments.extend(stream.feed(audio[at:at+8000]))
    segments.extend(stream.feed(np.empty(0,np.float32),flush=True))
    return segments

"""Pretrained CAMPPlus speaker vectors and YuNet/SFace FACE-only vectors.

Names/actor IDs label enrollment galleries, never substitute for inferred identity.
Thresholds are conservative provisional gates, not calibrated probabilities.
"""
import io,threading
import cv2,numpy as np,soundfile as sf,sherpa_onnx
from scipy.signal import resample_poly
from common import ROOT
from speech import speech_segments,normalized


def unit(vector):
    v=np.asarray(vector,dtype=np.float32).reshape(-1)
    if not np.isfinite(v).all() or np.linalg.norm(v)<1e-8:raise ValueError('无有效识别特征')
    return v/np.linalg.norm(v)


def audio_bytes(data):
    audio,fs=sf.read(io.BytesIO(data),dtype='float32')
    if audio.ndim==2:audio=audio.mean(axis=1)
    if fs!=16000:
        from math import gcd
        g=gcd(fs,16000);audio=resample_poly(audio,16000//g,fs//g)
    if not np.isfinite(audio).all() or len(audio)>16000*16:raise ValueError('录音无效或超过16秒')
    return audio


class Speaker:
    def __init__(self):
        self.lock=threading.Lock()
        config=sherpa_onnx.SpeakerEmbeddingExtractorConfig(model=str(ROOT/'models/campplus-zh-en.onnx'),num_threads=2,provider='cpu')
        self.extractor=sherpa_onnx.SpeakerEmbeddingExtractor(config)

    def analyze(self,audio):
        audio=np.asarray(audio,np.float32)
        if audio.ndim!=1 or len(audio)<8000:raise ValueError('有效语音太短，请完整说“小北小北”')
        if np.mean(np.abs(audio)>.99)>.08:raise ValueError('录音过载，请离麦克风稍远一些')
        segments=[s for s in speech_segments(audio) if len(s)>=6400]
        if not segments:raise ValueError('未检测到清晰人声，请检查麦克风或靠近一些')
        speech=np.concatenate(segments)
        if len(speech)<11200:raise ValueError('有效语音不足，请完整说“小北小北”并稍作停顿')
        quality={'duration_s':len(audio)/16000,'speech_s':len(speech)/16000,
                 'rms_dbfs':round(20*np.log10(max(float(np.sqrt(np.mean(audio**2))),1e-8)),1),
                 'clipping_ratio':float(np.mean(np.abs(audio)>.99)), 'utterances':len(segments)}
        return segments,quality

    def extract(self,speech):
        with self.lock:
            stream=self.extractor.create_stream();stream.accept_waveform(16000,normalized(speech));stream.input_finished()
            if not self.extractor.is_ready(stream):raise ValueError('有效语音不足，请再录一次')
            return unit(self.extractor.compute(stream))

    def embed(self,audio):
        segments,_=self.analyze(audio)
        return self.extract(np.concatenate(segments))

    def profile(self,audio,previous=None):
        segments,quality=self.analyze(audio)
        vectors=[self.extract(s) for s in segments if len(s)>=11200]
        if not vectors:vectors=[self.extract(np.concatenate(segments))]
        old=previous.get('samples',[]) if isinstance(previous,dict) else []
        vectors=[unit(v) for v in old]+vectors
        vectors=vectors[-8:]
        centre=unit(np.mean(vectors,axis=0))
        return {'version':2,'preprocessing':'silero-v5-contiguous','centroid':centre.tolist(),
                'samples':[v.tolist() for v in vectors],'sample_count':len(vectors),'quality':quality}


def match(vector,gallery,threshold=.62,margin=.08):
    scores=sorted(((float(unit(vector)@unit(v['centroid'] if isinstance(v,dict) else v)),name) for name,v in gallery.items()),reverse=True)
    if not scores:return None,0.,{}
    score,name=scores[0];second=scores[1][0] if len(scores)>1 else -1
    return (name if score>=threshold and score-second>=margin else None),score,dict((n,s) for s,n in scores)


class Faces:
    def __init__(self):
        self.detector=cv2.FaceDetectorYN_create(str(ROOT/'models/yunet.onnx'),'',(320,320),.75,.3,5000)
        self.recognizer=cv2.FaceRecognizerSF_create(str(ROOT/'models/sface.onnx'),'')
        self.lock=threading.Lock()

    def detect(self,image):
        with self.lock:
            self.detector.setInputSize((image.shape[1],image.shape[0]))
            _,faces=self.detector.detect(image)
            rows=[]
            for face in ([] if faces is None else faces):
                x,y,w,h=face[:4]
                if min(w,h)<18:continue
                aligned=self.recognizer.alignCrop(image,face)
                rows.append({'box':face[:4].tolist(),'embedding':unit(self.recognizer.feature(aligned)).tolist(),'crop':aligned})
            return rows

    def identify(self,image,gallery):
        output=[]
        for face in self.detect(image):
            name,score,scores=match(face['embedding'],gallery,threshold=.40,margin=.07)
            output.append({**face,'identity':name,'score':score,'scores':scores})
        return output

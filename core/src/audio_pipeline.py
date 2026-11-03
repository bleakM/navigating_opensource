"""Bounded local audio storage and transport; no simulator identity in ROS audio."""
import hashlib,io,re,time,uuid
import numpy as np
import soundfile as sf
from common import atomic,write_json

RATE=16000
REQUEST=re.compile(r'^[0-9a-f]{32}$')


def clip_path(data,identity):
    return data/'voice-clips'/(hashlib.sha256(identity.encode()).hexdigest()+'.wav')


def wav_bytes(audio):
    out=io.BytesIO()
    sf.write(out,np.asarray(audio,np.float32),RATE,format='WAV',subtype='PCM_16')
    return out.getvalue()


def save_clip(data,identity,audio):
    audio=np.asarray(audio,np.float32)
    if audio.ndim!=1 or not 0<len(audio)<=RATE*16 or not np.isfinite(audio).all():
        raise ValueError('绑定录音须为有效的16秒以内单声道语音')
    path=clip_path(data,identity)
    atomic(path,wav_bytes(audio));path.chmod(0o600)


def enqueue_audio(data,audio,slot,*,monitor=False):
    folder=data/'audio-inbox';folder.mkdir(exist_ok=True)
    if len(list(folder.glob('*.npz')))>=3:raise ValueError('声音处理繁忙，请稍等')
    request_id=uuid.uuid4().hex
    # The same silent tail is used for live speech and stored-clip playback.
    speech=np.r_[np.asarray(audio,np.float32),np.zeros(8000,np.float32)]
    packet=io.BytesIO()
    np.savez(packet,audio=speech,slot=np.array(slot,np.int32),request_id=np.array(request_id),monitor=np.array(monitor))
    atomic(folder/(str(time.time_ns())+'.npz'),packet.getvalue())
    return request_id


def write_monitor(data,request_id,wave,meta):
    """Copy first receiver channel after propagation/limiting, never the dry source.

    PCM16 introduces at most one quantization step. There is no monitor AGC.
    Keep only eight recent clips, expiring after ten minutes.
    """
    if not REQUEST.fullmatch(request_id):raise ValueError('无效音频请求')
    folder=data/'audio-monitor';folder.mkdir(exist_ok=True)
    for p in folder.glob('*'):
        if p.is_file() and time.time()-p.stat().st_mtime>600:p.unlink(missing_ok=True)
    old=sorted(folder.glob('*.wav'),key=lambda p:p.stat().st_mtime,reverse=True)
    for p in old[7:]:
        p.unlink(missing_ok=True);p.with_suffix('.json').unlink(missing_ok=True)
    path=folder/(request_id+'.wav')
    atomic(path,wav_bytes(wave[:,0]));path.chmod(0o600)
    write_json(path.with_suffix('.json'),{**meta,'status':'ready','channel':0,'channels_in_robot':4,
        'duration_s':len(wave)/RATE,'created_at':time.time(),'monitor_gain':1.})

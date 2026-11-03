"""Workbench acoustic adapter. Frozen baseline acoustics are left untouched.

Digital levels are relative simulation settings, NOT calibrated physical dB SPL.
Source level is controlled once; distance, portal loss and fixed receiver noise
remain audible. All microphones share gains, preserving phase/time differences.
"""
import time
from collections import OrderedDict
import numpy as np
import pyroomacoustics as pra
from scipy.signal import butter,sosfilt,fftconvolve
from shapely.geometry import LineString
from acoustics import MICROPHONES,estimate_direction
from portal_acoustics import ApartmentAcoustics

SOURCE_TARGET_DBFS=-14.

def dbfs(audio):
    a=np.asarray(audio,dtype=float)
    return float(20*np.log10(max(float(np.sqrt(np.mean(a*a))) if a.size else 0.,1e-8)))


def active_rms(audio):
    a=np.asarray(audio,float)
    if len(a)<320:return float(np.sqrt(np.mean(a*a))) if len(a) else 0.
    frames=a[:len(a)//320*320].reshape(-1,320)
    energy=np.mean(frames*frames,axis=1)
    active=energy[energy>max(float(energy.max())*.01,1e-12)]
    return float(np.sqrt(np.mean(active))) if len(active) else 0.


def source_level(audio,target_dbfs=SOURCE_TARGET_DBFS,max_gain_db=30.):
    a=np.asarray(audio,np.float32)
    if a.ndim!=1 or len(a)<1024 or not np.isfinite(a).all():raise ValueError('声源音频无效')
    rms=active_rms(a);peak=float(np.max(np.abs(a)))
    if rms<10**(-50/20):raise ValueError('原始人声过小（有效语音低于−50 dBFS），请靠近麦克风重录')
    gain=min(10**(max_gain_db/20),10**(target_dbfs/20)/rms,.92/max(peak,1e-8))
    return a*gain,{'input_active_dbfs':20*np.log10(rms),'source_active_dbfs':20*np.log10(rms*gain),
        'source_gain_db':20*np.log10(gain),'source_target_dbfs':target_dbfs,
        'source_peak_limited':bool(gain>=.92/max(peak,1e-8)*.9999),
        'level_units':'relative digital dBFS; not calibrated dB SPL'}


class WorkbenchAcoustics(ApartmentAcoustics):
    def __init__(self,scene,noise_dbfs=-62.,max_order=2,source_target_dbfs=SOURCE_TARGET_DBFS,
                 wall_loss_db=14.,wall_lowpass_hz=1500.):
        super().__init__(scene)
        self.noise_dbfs=noise_dbfs;self.max_order=max_order;self.rirs=OrderedDict()
        self.source_target_dbfs=source_target_dbfs
        self.wall_loss_db=wall_loss_db;self.wall_lowpass_hz=wall_lowpass_hz

    def render(self,speech,fs,source,centre,yaw=0.,seed=0):
        started=time.perf_counter()
        signal,levels=source_level(speech,target_dbfs=self.source_target_dbfs)
        original_signal=signal.copy()
        source,centre=np.asarray(source,float),np.asarray(centre,float)
        route,length=self.path(source,centre);blocked=len(route)>2
        secondary=source.copy()
        if blocked:
            secondary[:2]=route[-2]
            secondary[:2]+=.015*(centre[:2]-secondary[:2])/max(np.linalg.norm(centre[:2]-secondary[:2]),1e-6)
            signal=sosfilt(butter(2,3500,fs=fs,output='sos'),signal)
        c,s=np.cos(yaw),np.sin(yaw)
        microphones=MICROPHONES@np.array([[c,-s,0],[s,c,0],[0,0,1]]).T+centre
        # Exact geometry keys: rounded receiver poses can corrupt array TDOA.
        key=(fs,*secondary,*centre,float(yaw))
        hit=key in self.rirs
        if hit:
            rir=self.rirs.pop(key);self.rirs[key]=rir
        else:
            room=pra.Room.from_corners(np.asarray(self.free.exterior.coords[:-1]).T,
                fs=fs,max_order=self.max_order,materials=pra.Material(.55))
            room.extrude(self.scene['wall']['height'],materials=pra.Material(.55))
            room.add_microphone_array(microphones.T);room.add_source(secondary);room.compute_rir()
            rir=[np.asarray(r[0],float) for r in room.rir]
            self.rirs[key]=rir
            while len(self.rirs)>8:self.rirs.popitem(last=False)
        channels=[fftconvolve(signal,r) for r in rir]
        wave=np.column_stack([np.pad(a,(0,max(map(len,channels))-len(a))) for a in channels])
        # pyroomacoustics' RIR has approximately 1/r amplitude (checked with a
        # 1 m free-field impulse). Keep that common reference and distance loss;
        # do not normalize individual recordings or microphone channels.
        if blocked:
            last=float(np.linalg.norm(centre-secondary))
            wave*=last/max(float(length),last,.1)*10**(-6/20)
            extra=max(0,length-np.linalg.norm(centre[:2]-secondary[:2]))
            wave=np.pad(wave,((round(extra/343*fs),0),(0,0)))
        transmitted_db=None;wall_count=0
        if blocked and self.wall_loss_db is not None:
            # Parallel, frequency-dependent wall transmission, not a portal
            # coordinate secretly given to perception. Count actual solid wall
            # intervals on the direct acoustic ray; door openings are already cut.
            intersection=LineString([source[:2],centre[:2]]).intersection(self.solids)
            pieces=list(intersection.geoms) if hasattr(intersection,'geoms') else [intersection]
            wall_count=sum(p.length>.005 for p in pieces)
            if wall_count:
                direct_key=('transmission',fs,*source,*centre,float(yaw))
                if direct_key in self.rirs:
                    direct_rir=self.rirs.pop(direct_key);self.rirs[direct_key]=direct_rir
                else:
                    direct=pra.AnechoicRoom(fs=fs,dim=3)
                    direct.add_microphone_array(microphones.T);direct.add_source(source);direct.compute_rir()
                    direct_rir=[np.asarray(r[0],float) for r in direct.rir]
                    self.rirs[direct_key]=direct_rir
                    while len(self.rirs)>8:self.rirs.popitem(last=False)
                through=original_signal
                for _ in range(wall_count):
                    through=sosfilt(butter(2,self.wall_lowpass_hz,fs=fs,output='sos'),through)
                through*=10**(-self.wall_loss_db*wall_count/20)
                direct_channels=[fftconvolve(through,r) for r in direct_rir]
                length_out=max(len(wave),max(map(len,direct_channels)))
                transmitted=np.column_stack([np.pad(a,(0,length_out-len(a))) for a in direct_channels])
                wave=np.pad(wave,((0,length_out-len(wave)),(0,0)))+transmitted
                transmitted_db=dbfs(transmitted)
        clean_db=dbfs(wave)
        noise=np.random.default_rng(seed).normal(0,10**(self.noise_dbfs/20),wave.shape)
        wave+=noise
        peak=float(np.max(np.abs(wave)));limiter=min(1.,.98/max(peak,1e-8))
        wave=(wave*limiter).astype(np.float32)
        return wave,{**levels,'receiver_dbfs':dbfs(wave),'noise_dbfs':self.noise_dbfs+20*np.log10(limiter),
            'snr_db':clean_db-self.noise_dbfs,'limiter_gain_db':20*np.log10(limiter),
            'direct_blocked':blocked,'portal_count':max(0,len(route)-2),'path_length_m':length,
            'wall_intersections':wall_count,'wall_transmission_dbfs':transmitted_db,
            'wall_loss_db_per_crossing':self.wall_loss_db,'wall_lowpass_hz_per_crossing':self.wall_lowpass_hz,
            'propagation':f'nonconvex ISM order {self.max_order} + approximate portal path + frequency-dependent direct wall transmission; uncalibrated partition-wall model',
            'rir_cache_hit':hit,'render_seconds':time.perf_counter()-started}


def reliable_direction(wave,fs):
    """Speech-gated SRP-PHAT with independent time-window agreement.

    Geometry, sender identity and source coordinates are deliberately absent.
    PHAT gives quiet/noise frames equal phase weight; remove these frames before
    invoking the existing pyroomacoustics estimator, not after finding a peak.
    """
    wave=np.asarray(wave,np.float32)
    if (wave.ndim!=2 or wave.shape[1]!=4 or len(wave)<1024
            or not np.isfinite(wave).all() or fs!=16000):
        raise ValueError('声向需要有效的16kHz四通道音频')
    spectrum=pra.transform.stft.analysis(wave,512,256).transpose(2,1,0)
    bins=np.fft.rfftfreq(512,1/fs)
    band=(bins>=300)&(bins<=3500)
    energy=np.mean(np.abs(spectrum[:,band,:])**2,axis=(0,1))
    # Shared gate across all microphones; no channel-dependent normalization.
    noise=float(np.percentile(energy,15));peak=float(np.percentile(energy,95))
    keep=energy>max(noise*3.2,peak*.045,1e-12)
    frames=np.flatnonzero(keep)
    if len(frames)<10 or dbfs(wave)<-57:
        raise ValueError('有效语音或信噪比不足，本次不执行转向，请再次呼叫')

    def locate(indices,frequencies=(300,3500)):
        doa=pra.doa.SRP(MICROPHONES.T,fs=fs,nfft=512,c=343.,num_src=1,
                        dim=2,n_grid=360,mode='far')
        doa.locate_sources(spectrum[:,:,indices],freq_range=list(frequencies))
        values=np.asarray(doa.grid.values,float)
        return values,float(doa.azimuth_recon[0])

    values,azimuth=locate(frames)
    estimates=[locate(part)[1] for part in np.array_split(frames,3) if len(part)>=3]
    errors=np.abs(np.arctan2(np.sin(np.asarray(estimates)-azimuth),np.cos(np.asarray(estimates)-azimuth)))
    spread=float(np.max(errors))
    contrast=float((values.max()-np.median(values))/max(abs(values.max()),1e-12))
    if contrast<.08 or len(estimates)<3 or spread>np.deg2rad(20):
        raise ValueError('不同语音片段测向不一致，本次不执行转向，请再次呼叫')
    # Report alternative angular peaks instead of turning a peak into certainty.
    angles=np.arange(360)*2*np.pi/360
    ranked=np.argsort(values)[::-1];peaks=[]
    for i in ranked:
        angle=float(angles[i]);delta=lambda a:abs(np.arctan2(np.sin(angle-a),np.cos(angle-a)))
        if all(delta(p['azimuth_rad'])>np.deg2rad(35) for p in peaks):
            peaks.append({'azimuth_rad':float(np.arctan2(np.sin(angle),np.cos(angle))),
                          'relative_score':float((values[i]-values.min())/max(np.ptp(values),1e-12))})
        if len(peaks)==3:break
    spectral=[]
    for band_range in ((300,1200),(1200,3500)):
        band_values,angle=locate(frames,band_range)
        band_contrast=float((band_values.max()-np.median(band_values))/max(abs(band_values.max()),1e-12))
        spectral.append({'band_hz':list(band_range),'azimuth_rad':float(np.arctan2(np.sin(angle),np.cos(angle))),
            'peak_contrast':band_contrast})
    return {'azimuth_rad':float(np.arctan2(np.sin(azimuth),np.cos(azimuth))),
        'peak_contrast':contrast,'method':'pyroomacoustics.SRP-PHAT + shared speech-frame gate + 3-window consensus',
        'input':'four-channel waveform','range_estimated':False,
        'window_count':len(estimates),'window_spread_deg':float(np.rad2deg(spread)),
        'active_frame_count':len(frames),'total_frame_count':len(energy),
        'half_angle_deg':max(20.,float(np.rad2deg(spread))+10.),
        'angular_candidates':peaks,'frequency_candidates':spectral,'scores_are_probabilities':False}

"""Audio synthesis and SRP-PHAT are separate APIs: estimator never gets source poses."""
import numpy as np
import pyroomacoustics as pra

MICROPHONES = np.array([[.04,.04,0],[-.04,.04,0],[-.04,-.04,0],[.04,-.04,0]])


def render_audio(speech, sample_rate, source, centre, yaw=0., max_order=0, snr_db=None):
    """Simulation side only: 14x12x2.6m shoebox, not a multiroom wall model."""
    c,s = np.cos(yaw),np.sin(yaw)
    rotation = np.array([[c,-s,0],[s,c,0],[0,0,1]])
    positions = MICROPHONES @ rotation.T + np.asarray(centre)
    room = pra.ShoeBox([14,12,2.6],fs=sample_rate,max_order=max_order,
                       materials=pra.Material(.45))
    room.c = 343.
    room.add_microphone_array(positions.T)
    room.add_source(source,signal=speech)
    room.simulate(snr=snr_db)
    signals = room.mic_array.signals.T
    signals /= max(float(np.abs(signals).max()),1e-8)
    return signals.astype(np.float32)


def estimate_direction(signals, sample_rate):
    """Four channels in array coordinates; azimuth0=forward, +pi/2=left."""
    signals = np.asarray(signals,dtype=float)
    if (signals.ndim!=2 or signals.shape[1]!=4 or signals.shape[0]<1024
            or not np.isfinite(signals).all() or np.sqrt(np.mean(signals**2))<1e-4):
        raise ValueError('Need finite, non-silent four-channel audio')
    spectrum = pra.transform.stft.analysis(signals,512,256).transpose(2,1,0)
    doa = pra.doa.SRP(MICROPHONES.T,fs=sample_rate,nfft=512,c=343.,num_src=1,
                     dim=2,n_grid=360,mode='far')
    doa.locate_sources(spectrum,freq_range=[300,min(3500,sample_rate*.45)])
    azimuth = float(doa.azimuth_recon[0])
    # Contrast is a diagnostic, not a calibrated probability or wall-aware confidence.
    contrast = float((doa.grid.values.max()-np.median(doa.grid.values)) /
                     max(abs(doa.grid.values.max()),1e-12))
    return {'azimuth_rad':float(np.arctan2(np.sin(azimuth),np.cos(azimuth))),
            'peak_contrast':contrast,'method':'pyroomacoustics.SRP-PHAT',
            'input':'four-channel waveform','range_estimated':False}
